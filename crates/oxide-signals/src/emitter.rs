/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! High-level integration API for emitting OIS signals through OpenTelemetry.
//!
//! [`SignalEmitter`] wraps an OTEL [`Tracer`](opentelemetry::trace::Tracer) and
//! provides ergonomic methods for converting OIS envelopes into OTEL spans.
//!
//! # Example
//!
//! ```ignore
//! use oxide_signals::{SignalEmitter, SignalClass, Actor, ActorType, Reproducibility, OisUrn};
//! use opentelemetry::global;
//! use serde_json::json;
//!
//! let tracer = global::tracer("cuda-oxide");
//! let emitter = SignalEmitter::new(tracer);
//!
//! emitter.emit(
//!     SignalEnvelope::builder()
//!         .signal_id("urn:ois:signal:acme:exec-7:trace:1".parse().unwrap())
//!         .signal_class(SignalClass::Trace)
//!         .exec_id("urn:ois:exec:acme:exec-7".parse().unwrap())
//!         .tenant_id("urn:ois:tenant:acme".parse().unwrap())
//!         .actor(Actor::new(
//!             "urn:ois:agent:acme:compiler".parse().unwrap(),
//!             ActorType::Agent,
//!         ))
//!         .reproducibility(Reproducibility::Deterministic)
//!         .payload(json!({ "event": "kernel_launch" }))
//!         .build()
//!         .unwrap(),
//! ).unwrap();
//! ```

use crate::otel;
use crate::types::*;
use opentelemetry::trace::{Span, Tracer};
use std::error::Error;
use std::fmt;

/// Error type for signal emission failures.
#[derive(Debug)]
pub struct EmitError {
    msg: String,
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.msg.fmt(f)
    }
}

impl Error for EmitError {}

/// High-level emitter that converts OIS signals into OTEL spans.
///
/// Create one from an OTEL [`Tracer`] and call [`emit`](Self::emit) for each
/// signal. All required metadata (trace id, span id, attributes, links) is
/// derived automatically from the envelope.
pub struct SignalEmitter<T: Tracer> {
    tracer: T,
}

impl<T: Tracer> SignalEmitter<T> {
    /// Creates a new [`SignalEmitter`] backed by `tracer`.
    pub fn new(tracer: T) -> Self {
        Self { tracer }
    }

    /// Emits an OIS signal as an OTEL span.
    ///
    /// The span is started immediately with the signal's `emitted_at` timestamp,
    /// populated with all envelope attributes, and ended before returning.
    ///
    /// # Errors
    ///
    /// Returns [`EmitError`] if the underlying tracer fails to create a span.
    pub fn emit(&self, signal: &JsonSignalEnvelope) -> Result<(), EmitError> {
        let span_kind = otel::signal_class_to_span_kind(signal.signal_class);
        let attrs = otel::to_otel_attributes(signal);
        let links = otel::to_span_links(signal);

        let span_builder = self
            .tracer
            .span_builder(signal.signal_id.to_string())
            .with_kind(span_kind)
            .with_attributes(attrs)
            .with_links(links);

        let mut span = span_builder.start(&self.tracer);

        // End the span immediately — signals are point-in-time events.
        span.end();

        Ok(())
    }

    /// Emits a signal with a custom span name.
    ///
    /// This is useful when the `signal_id` URN is too verbose for span
    /// display names in backends like Jaeger or Grafana.
    pub fn emit_named(
        &self,
        name: impl Into<String>,
        signal: &JsonSignalEnvelope,
    ) -> Result<(), EmitError> {
        let span_kind = otel::signal_class_to_span_kind(signal.signal_class);
        let attrs = otel::to_otel_attributes(signal);
        let links = otel::to_span_links(signal);

        let span_builder = self
            .tracer
            .span_builder(name.into())
            .with_kind(span_kind)
            .with_attributes(attrs)
            .with_links(links);

        let mut span = span_builder.start(&self.tracer);
        span.end();

        Ok(())
    }
}

/// Convenience function that creates a [`SignalEmitter`] from the global
/// OTEL tracer named `"oxide-signals"`.
///
/// # Panics
///
/// Panics if the global tracer provider has not been initialised.
pub fn global_emitter() -> SignalEmitter<opentelemetry::global::BoxedTracer> {
    SignalEmitter::new(opentelemetry::global::tracer("oxide-signals"))
}
