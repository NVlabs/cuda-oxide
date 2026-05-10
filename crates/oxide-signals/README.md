# oxide-signals

OIS (Oxide Instrumentation Signal) canonical envelope types with OpenTelemetry integration for [cuda-oxide](https://github.com/NVlabs/cuda-oxide).

## Overview

`oxide-signals` provides a typed, versioned envelope for observability data emitted by `cuda-oxide` tooling. Every signal — whether a kernel trace, compilation metric, policy decision, or audit log — carries the same identity, provenance, and lifecycle metadata, enabling downstream collectors to correlate, filter, and verify signals without parsing opaque payloads.

Signals are natively mapped to the [OpenTelemetry](https://opentelemetry.io/) data model so they can be exported through any OTEL-compatible collector (OTLP, Jaeger, stdout, etc.).

## Quick start

```rust
use oxide_signals::{SignalEnvelope, SignalClass, Actor, ActorType, Reproducibility, OisUrn};
use chrono::Utc;
use serde_json::json;

let signal = SignalEnvelope::builder()
    .signal_id("urn:ois:signal:acme:exec-42:trace:1".parse().unwrap())
    .signal_class(SignalClass::Trace)
    .exec_id("urn:ois:exec:acme:exec-42".parse().unwrap())
    .tenant_id("urn:ois:tenant:acme".parse().unwrap())
    .actor(Actor::new(
        "urn:ois:agent:acme:compiler-v1".parse().unwrap(),
        ActorType::Agent,
    ))
    .reproducibility(Reproducibility::Deterministic)
    .payload(json!({ "event": "kernel_launch", "grid": [4, 1, 1] }))
    .build()
    .unwrap();

// Serialize to JSON
let json = signal.to_json().unwrap();
```

## Architecture

- **`types`** — Canonical OIS envelope types (`SignalEnvelope`, `Actor`, `Signature`, `OisUrn`, etc.) with serde and JSON Schema support via `schemars`.
- **`otel`** — OpenTelemetry mapping layer that converts OIS signals into OTEL traces, spans, attributes, and links.

## Signal classes

| Class | OTEL mapping | Typical use |
|-------|-------------|-------------|
| `trace` | `SpanKind::Internal` | Kernel launch, memory copy |
| `metric` | `SpanKind::Internal` | GPU utilization, throughput |
| `log` | `SpanKind::Internal` | Compiler diagnostics |
| `decision` | `SpanKind::Server` | Policy engine verdicts |
| `receipt` | `SpanKind::Producer` | Build artifact receipts |
| `evidence` | `SpanKind::Producer` | SAST / attestation evidence |
| `stream` | `SpanKind::Consumer` | Async data streams |
| `state` | `SpanKind::Internal` | Runtime state snapshots |
| `policy` | `SpanKind::Server` | Governance rule evaluations |

## Schema

See [`SCHEMA.md`](SCHEMA.md) for the full canonical envelope specification.

## License

Apache-2.0 — see the workspace `LICENSE-APACHE` file.
