# OIS (Oxide Instrumentation Signal) Canonical Envelope — v1.0

**Namespace:** `urn:ois:`  
**Schema Base URI:** `https://nvidia.com/oxide/ois/v1.0/schema/`  
**Status:** Stable — every emitted signal MUST validate against this envelope.

---

## Purpose

The OIS canonical envelope provides a unified, versioned wrapper for all observability data produced by `cuda-oxide` and related NVIDIA tooling. It guarantees that traces, metrics, logs, decisions, evidence, and policy signals carry identical identity, provenance, and lifecycle metadata regardless of payload shape.

By standardizing on this envelope, downstream collectors can:

- Correlate signals across execution boundaries via `exec_id`.
- Attribute every signal to a specific tenant and actor.
- Enforce retention and reproducibility policies without parsing opaque payloads.
- Verify authenticity through embedded cryptographic signatures.
- Reconstruct causal graphs via `lineage_refs`.

---

## Envelope Structure

| Field             | Type                  | Required | Description                                                                                      |
| ----------------- | --------------------- | -------- | ------------------------------------------------------------------------------------------------ |
| `signal_id`       | `string` (URN)        | **Yes**  | Unique identifier for this signal. Format: `urn:ois:signal:{tenant}:{exec}:{class}:{seq}`        |
| `signal_class`    | `enum`                | **Yes**  | One of: `trace`, `decision`, `policy`, `receipt`, `evidence`, `state`, `stream`, `metric`, `log` |
| `signal_version`  | `string`              | **Yes**  | Fixed `"1.0"` — the envelope schema version                                                      |
| `schema_uri`      | `string` (URI)        | No       | URI of the payload-specific JSON schema                                                          |
| `exec_id`         | `string` (URN)        | **Yes**  | Execution context URN. Format: `urn:ois:exec:{tenant}:{id}`                                      |
| `tenant_id`       | `string` (URN)        | **Yes**  | Tenant / organisation URN. Format: `urn:ois:tenant:{name}`                                       |
| `emitted_at`      | `string` (RFC3339)    | **Yes**  | Wall-clock time when the signal was emitted                                                      |
| `captured_at`     | `string` (RFC3339)    | No       | Wall-clock time when the signal was captured (may differ from `emitted_at` in async pipelines)   |
| `actor`           | `object`              | **Yes**  | See [Actor](#actor) below                                                                        |
| `reproducibility` | `enum`                | **Yes**  | One of: `exact`, `deterministic`, `attested`, `explanatory`, `best_effort`, `lossy_capped`       |
| `retention_class` | `enum`                | No       | One of: `ephemeral`, `standard`, `regulated`, `legal_hold`, `sovereign_archive`                  |
| `channel`         | `enum`                | No       | One of: `control`, `data`, `proof`, `recall`, `evidence`                                         |
| `payload`         | `object`              | **Yes**  | Arbitrary JSON object — shape defined by `schema_uri`                                            |
| `signature`       | `object`              | No       | See [Signature](#signature) below                                                                |
| `lineage_refs`    | `array<string>` (URN) | No       | Parent signal URNs that this signal causally depends on                                          |

---

## Actor

| Field            | Type                  | Required | Description                                                |
| ---------------- | --------------------- | -------- | ---------------------------------------------------------- |
| `id`             | `string` (URN)        | **Yes**  | Actor URN. Format: `urn:ois:{actor-type}:{tenant}:{id}`    |
| `type`           | `enum`                | **Yes**  | One of: `human`, `agent`, `service`, `runtime`, `pipeline` |
| `delegate_chain` | `array<string>` (URN) | No       | Ordered chain of delegation (e.g. human → agent → runtime) |

---

## Signature

| Field       | Type           | Required | Description                                                                |
| ----------- | -------------- | -------- | -------------------------------------------------------------------------- |
| `alg`       | `enum`         | **Yes**  | One of: `ed25519`, `ecdsa-p256`, `dilithium3`, `hybrid-ed25519-dilithium3` |
| `value`     | `string`       | **Yes`   | Base64-encoded signature bytes                                             |
| `key_ref`   | `string` (URN) | **Yes**  | Reference to the signing key URN                                           |
| `rekor_uri` | `string` (URI) | No       | Link to the Rekor transparency-log entry                                   |

---

## URN Schemes

All OIS URNs share the prefix `urn:ois:` and use colon-separated segments:

```text
urn:ois:signal:{tenant}:{exec}:{class}:{seq}
urn:ois:exec:{tenant}:{id}
urn:ois:tenant:{name}
urn:ois:agent:{tenant}:{id}
urn:ois:key:{tenant}:{id}
```

---

## Reproducibility Tiers

| Tier            | Meaning                                                             |
| --------------- | ------------------------------------------------------------------- |
| `exact`         | Bit-for-bit reproducible given identical inputs                     |
| `deterministic` | Same inputs yield same outputs, but environment must be pinned      |
| `attested`      | Result attested by a trusted party; not necessarily reproducible    |
| `explanatory`   | Human-readable rationale provided; no mechanical reproducibility    |
| `best_effort`   | Attempted reproducibility with known gaps                           |
| `lossy_capped`  | Bounded, acceptable information loss; exact reproduction impossible |

---

## OpenTelemetry Mapping

The `oxide-signals` crate maps every OIS envelope onto the OpenTelemetry data model:

- `exec_id` → `TraceId` (execution = distributed trace)
- `signal_id` → `SpanId` (individual signal = span)
- `actor`, `tenant_id` → Resource attributes
- `payload` → Span attributes (`ois.payload.*`) or log body
- `lineage_refs` → Span links
- `signal_class` → `SpanKind` (`Internal`, `Server`, `Producer`, `Consumer`)

See [`src/otel.rs`](src/otel.rs) for the conversion logic.

---

## Version History

| Version | Date       | Changes                |
| ------- | ---------- | ---------------------- |
| `1.0`   | 2026-05-10 | Initial stable release |
