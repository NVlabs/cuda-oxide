/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! OIS (Oxide Instrumentation Signal) canonical envelope types with
//! OpenTelemetry integration.
//!
//! This crate provides:
//! 1. The Rust type system for the canonical OIS signal envelope defined by
//!    `https://nvidia.com/oxide/ois/v1.0/schema/envelope/canonical-signal-envelope.schema.json`.
//! 2. An OpenTelemetry mapping layer so that OIS signals can be emitted through
//!    any OTEL-compatible exporter.
//!
//! # Quick start
//!
//! ```
//! use oxide_signals::{SignalEnvelope, SignalClass, Actor, ActorType, Reproducibility, OisUrn};
//! use chrono::Utc;
//! use serde_json::json;
//!
//! let signal = SignalEnvelope::builder()
//!     .signal_id("urn:ois:signal:acme:exec-42:trace:1".parse().unwrap())
//!     .signal_class(SignalClass::Trace)
//!     .exec_id("urn:ois:exec:acme:exec-42".parse().unwrap())
//!     .tenant_id("urn:ois:tenant:acme".parse().unwrap())
//!     .actor(Actor::new(
//!         "urn:ois:agent:acme:compiler-v1".parse().unwrap(),
//!         ActorType::Agent,
//!     ))
//!     .reproducibility(Reproducibility::Deterministic)
//!     .payload(json!({ "event": "kernel_launch", "grid": [4, 1, 1] }))
//!     .build()
//!     .unwrap();
//! ```

pub mod emitter;
pub mod otel;
pub mod types;

pub use emitter::{EmitError, SignalEmitter, global_emitter};
pub use types::*;

#[cfg(test)]
mod tests;
