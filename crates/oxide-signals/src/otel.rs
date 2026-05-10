/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! OpenTelemetry integration for OIS signals.
//!
//! This module maps the canonical OIS signal envelope onto the OpenTelemetry
//! data model so that cuda-oxide observability can be exported through any
//! OTEL collector (stdout, OTLP, Jaeger, etc.).
//!
//! # Mapping rules
//!
//! | OIS field            | OTEL construct                     |
//! |----------------------|------------------------------------|
//! | `exec_id`            | `TraceId` (execution = trace)     |
//! | `signal_id`          | `SpanId` or log record id         |
//! | `tenant_id`          | Resource attribute                 |
//! | `actor`              | Resource attributes                |
//! | `signal_class`       | Span kind / log severity           |
//! | `emitted_at`         | Timestamp                          |
//! | `payload`            | Span attributes / log body         |
//! | `reproducibility`    | Span/log attribute `ois.repro`     |
//! | `retention_class`    | Span/log attribute `ois.retention` |
//! | `channel`            | Span/log attribute `ois.channel`   |
//! | `signature`          | Span/log attribute `ois.sig.alg`   |
//! | `lineage_refs`       | Span links                         |

use crate::types::*;
use opentelemetry::KeyValue;
use opentelemetry::trace::{SpanId, TraceId};
use serde_json::Value;
use std::hash::{Hash, Hasher};

/// Derives an OTEL [`TraceId`] from an OIS URN.
///
/// The URN is hashed to a 128-bit value so that every execution context
/// gets a stable, deterministic OTEL trace identifier.
pub fn urn_to_trace_id(urn: &OisUrn) -> TraceId {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    urn.as_str().hash(&mut hasher);
    let h1 = hasher.finish();
    urn.as_str().bytes().for_each(|b| b.hash(&mut hasher));
    let h2 = hasher.finish();
    let bytes: [u8; 16] = [
        (h1 >> 56) as u8,
        (h1 >> 48) as u8,
        (h1 >> 40) as u8,
        (h1 >> 32) as u8,
        (h1 >> 24) as u8,
        (h1 >> 16) as u8,
        (h1 >> 8) as u8,
        h1 as u8,
        (h2 >> 56) as u8,
        (h2 >> 48) as u8,
        (h2 >> 40) as u8,
        (h2 >> 32) as u8,
        (h2 >> 24) as u8,
        (h2 >> 16) as u8,
        (h2 >> 8) as u8,
        h2 as u8,
    ];
    TraceId::from_bytes(bytes)
}

/// Derives an OTEL [`SpanId`] from an OIS URN.
///
/// The URN is hashed to a 64-bit value so that every signal gets a stable,
/// deterministic OTEL span identifier.
pub fn urn_to_span_id(urn: &OisUrn) -> SpanId {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    urn.as_str().hash(&mut hasher);
    let h = hasher.finish();
    let bytes: [u8; 8] = [
        (h >> 56) as u8,
        (h >> 48) as u8,
        (h >> 40) as u8,
        (h >> 32) as u8,
        (h >> 24) as u8,
        (h >> 16) as u8,
        (h >> 8) as u8,
        h as u8,
    ];
    SpanId::from_bytes(bytes)
}

/// Converts an OIS signal envelope into a flat list of OTEL [`KeyValue`] attributes.
///
/// All required and optional scalar fields are included. The payload is
/// flattened as `ois.payload.<key>` attributes when it is a JSON object.
pub fn to_otel_attributes(signal: &JsonSignalEnvelope) -> Vec<KeyValue> {
    let mut attrs = vec![
        KeyValue::new("ois.signal_id", signal.signal_id.to_string()),
        KeyValue::new("ois.signal_class", signal.signal_class.to_string()),
        KeyValue::new("ois.signal_version", signal.signal_version.clone()),
        KeyValue::new("ois.exec_id", signal.exec_id.to_string()),
        KeyValue::new("ois.tenant_id", signal.tenant_id.to_string()),
        KeyValue::new("ois.actor.id", signal.actor.id.to_string()),
        KeyValue::new("ois.actor.type", signal.actor.actor_type.to_string()),
        KeyValue::new("ois.reproducibility", signal.reproducibility.to_string()),
        KeyValue::new("ois.emitted_at", signal.emitted_at.to_rfc3339()),
    ];

    if let Some(ref captured) = signal.captured_at {
        attrs.push(KeyValue::new("ois.captured_at", captured.to_rfc3339()));
    }
    if let Some(ref retention) = signal.retention_class {
        attrs.push(KeyValue::new("ois.retention_class", retention.to_string()));
    }
    if let Some(ref channel) = signal.channel {
        attrs.push(KeyValue::new("ois.channel", channel.to_string()));
    }
    if let Some(ref schema_uri) = signal.schema_uri {
        attrs.push(KeyValue::new("ois.schema_uri", schema_uri.clone()));
    }
    if let Some(ref sig) = signal.signature {
        attrs.push(KeyValue::new("ois.signature.alg", sig.alg.to_string()));
        attrs.push(KeyValue::new(
            "ois.signature.key_ref",
            sig.key_ref.to_string(),
        ));
    }
    if let Some(ref lineage) = signal.lineage_refs {
        let refs: Vec<String> = lineage.iter().map(|u| u.to_string()).collect();
        attrs.push(KeyValue::new("ois.lineage_refs", refs.join(",")));
    }

    if let Value::Object(map) = &signal.payload {
        for (k, v) in map {
            attrs.push(KeyValue::new(
                format!("ois.payload.{}", k),
                serde_json::to_string(v).unwrap_or_default(),
            ));
        }
    } else {
        attrs.push(KeyValue::new(
            "ois.payload",
            serde_json::to_string(&signal.payload).unwrap_or_default(),
        ));
    }

    attrs
}

/// Resource-level attributes extracted from an OIS signal.
///
/// These should be attached to the OTEL `Resource` of the provider so that
/// every span/log emitted by the process carries the execution and tenant
/// context.
pub fn to_resource_attributes(signal: &JsonSignalEnvelope) -> Vec<KeyValue> {
    vec![
        KeyValue::new("ois.exec_id", signal.exec_id.to_string()),
        KeyValue::new("ois.tenant_id", signal.tenant_id.to_string()),
        KeyValue::new("ois.actor.id", signal.actor.id.to_string()),
        KeyValue::new("ois.actor.type", signal.actor.actor_type.to_string()),
    ]
}

/// Span links derived from `lineage_refs`.
///
/// Each lineage reference becomes a [`SpanLink`](opentelemetry::trace::Link)
/// back to the parent signal, allowing OTEL backends to reconstruct the
/// causal graph.
pub fn to_span_links(signal: &JsonSignalEnvelope) -> Vec<opentelemetry::trace::Link> {
    signal
        .lineage_refs
        .as_ref()
        .map(|refs| {
            refs.iter()
                .map(|urn| {
                    let span_context = opentelemetry::trace::SpanContext::new(
                        urn_to_trace_id(urn),
                        urn_to_span_id(urn),
                        opentelemetry::trace::TraceFlags::default(),
                        false,
                        opentelemetry::trace::TraceState::default(),
                    );
                    opentelemetry::trace::Link::new(span_context, Vec::new(), 0)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// OTEL [`SpanKind`](opentelemetry::trace::SpanKind) inferred from the
/// signal class.
///
/// | OIS class | SpanKind |
/// |-----------|----------|
/// | `trace`   | `Internal` |
/// | `metric`  | `Internal` |
/// | `log`     | `Internal` |
/// | `stream`  | `Consumer` |
/// | `decision`| `Server`   |
/// | `policy`  | `Server`   |
/// | `receipt` | `Producer` |
/// | `evidence`| `Producer` |
/// | `state`   | `Internal` |
pub fn signal_class_to_span_kind(class: SignalClass) -> opentelemetry::trace::SpanKind {
    use opentelemetry::trace::SpanKind;
    match class {
        SignalClass::Trace => SpanKind::Internal,
        SignalClass::Metric => SpanKind::Internal,
        SignalClass::Log => SpanKind::Internal,
        SignalClass::Stream => SpanKind::Consumer,
        SignalClass::Decision => SpanKind::Server,
        SignalClass::Policy => SpanKind::Server,
        SignalClass::Receipt => SpanKind::Producer,
        SignalClass::Evidence => SpanKind::Producer,
        SignalClass::State => SpanKind::Internal,
    }
}
