use super::*;
use serde_json::json;

#[test]
fn urn_validates_prefix() {
    assert!(OisUrn::new("urn:ois:tenant:acme").is_ok());
    assert!(OisUrn::new("urn:ois:signal:tenant:exec:trace:1").is_ok());
}

#[test]
fn urn_rejects_invalid_prefix() {
    assert!(OisUrn::new("http://example.com").is_err());
    assert!(OisUrn::new("urn:other:thing").is_err());
    assert!(OisUrn::new("").is_err());
}

#[test]
fn urn_roundtrip_string() {
    let urn = OisUrn::new("urn:ois:exec:42").unwrap();
    let s: String = urn.into();
    assert_eq!(s, "urn:ois:exec:42");
}

#[test]
fn urn_display() {
    let urn = OisUrn::new("urn:ois:test").unwrap();
    assert_eq!(format!("{}", urn), "urn:ois:test");
}

#[test]
fn urn_from_str() {
    let urn: OisUrn = "urn:ois:foo".parse().unwrap();
    assert_eq!(urn.as_str(), "urn:ois:foo");
}

#[test]
fn urn_serde_roundtrip() {
    let urn = OisUrn::new("urn:ois:tenant:x").unwrap();
    let json = serde_json::to_string(&urn).unwrap();
    assert_eq!(json, "\"urn:ois:tenant:x\"");
    let back: OisUrn = serde_json::from_str(&json).unwrap();
    assert_eq!(urn, back);
}

#[test]
fn envelope_builder_roundtrip() {
    let envelope = SignalEnvelope::builder()
        .signal_id("urn:ois:signal:acme:exec-7:trace:1".parse().unwrap())
        .signal_class(SignalClass::Trace)
        .exec_id("urn:ois:exec:acme:exec-7".parse().unwrap())
        .tenant_id("urn:ois:tenant:acme".parse().unwrap())
        .actor(Actor::new(
            "urn:ois:agent:acme:compiler".parse().unwrap(),
            ActorType::Agent,
        ))
        .reproducibility(Reproducibility::Deterministic)
        .payload(json!({ "grid": [2, 1, 1], "block": [128, 1, 1] }))
        .build()
        .unwrap();

    let json_str = envelope.to_json().unwrap();
    let restored: JsonSignalEnvelope = JsonSignalEnvelope::from_json(&json_str).unwrap();

    assert_eq!(envelope.signal_id, restored.signal_id);
    assert_eq!(envelope.signal_class, restored.signal_class);
    assert_eq!(envelope.signal_version, restored.signal_version);
    assert_eq!(envelope.exec_id, restored.exec_id);
    assert_eq!(envelope.tenant_id, restored.tenant_id);
    assert_eq!(envelope.actor, restored.actor);
    assert_eq!(envelope.reproducibility, restored.reproducibility);
    assert_eq!(envelope.payload, restored.payload);
}

#[test]
fn envelope_builder_missing_field_fails() {
    let result = JsonSignalEnvelope::builder()
        .signal_id("urn:ois:signal:t:e:trace:1".parse().unwrap())
        // missing signal_class, exec_id, tenant_id, actor, reproducibility, payload
        .build();
    assert!(result.is_err());
}

#[test]
fn envelope_builder_invalid_version_fails() {
    let result = JsonSignalEnvelope::builder()
        .signal_id("urn:ois:signal:t:e:trace:1".parse().unwrap())
        .signal_class(SignalClass::Trace)
        .exec_id("urn:ois:exec:t:e".parse().unwrap())
        .tenant_id("urn:ois:tenant:t".parse().unwrap())
        .actor(Actor::new(
            "urn:ois:actor:a".parse().unwrap(),
            ActorType::Agent,
        ))
        .reproducibility(Reproducibility::Exact)
        .payload(json!({}))
        .signal_version("2.0")
        .build();
    assert!(matches!(result, Err(ValidationError::InvalidVersion(_))));
}

#[test]
fn envelope_with_optional_fields() {
    let envelope = SignalEnvelope::builder()
        .signal_id("urn:ois:signal:acme:exec-7:metric:3".parse().unwrap())
        .signal_class(SignalClass::Metric)
        .exec_id("urn:ois:exec:acme:exec-7".parse().unwrap())
        .tenant_id("urn:ois:tenant:acme".parse().unwrap())
        .actor(Actor::new(
            "urn:ois:agent:acme:metricsd".parse().unwrap(),
            ActorType::Service,
        ))
        .reproducibility(Reproducibility::Exact)
        .payload(json!({ "gpu_util": 0.87 }))
        .schema_uri("https://nvidia.com/oxide/ois/v1.0/schema/metric.schema.json".to_string())
        .retention_class(RetentionClass::Standard)
        .channel(Channel::Data)
        .signature(Signature {
            alg: SignatureAlg::Ed25519,
            value: "deadbeef".to_string(),
            key_ref: "urn:ois:key:acme:signing-01".parse().unwrap(),
            rekor_uri: None,
        })
        .lineage_refs(vec!["urn:ois:signal:acme:exec-7:trace:1".parse().unwrap()])
        .build()
        .unwrap();

    let json_str = envelope.to_json().unwrap();
    let restored: JsonSignalEnvelope = JsonSignalEnvelope::from_json(&json_str).unwrap();

    assert_eq!(envelope.schema_uri, restored.schema_uri);
    assert_eq!(envelope.retention_class, restored.retention_class);
    assert_eq!(envelope.channel, restored.channel);
    assert_eq!(envelope.signature, restored.signature);
    assert_eq!(envelope.lineage_refs, restored.lineage_refs);
}

#[test]
fn actor_delegate_chain_roundtrip() {
    let actor = Actor::new("urn:ois:agent:acme:main".parse().unwrap(), ActorType::Agent)
        .with_delegate_chain(vec![
            "urn:ois:agent:acme:sub-1".parse().unwrap(),
            "urn:ois:agent:acme:sub-2".parse().unwrap(),
        ]);

    let json = serde_json::to_string(&actor).unwrap();
    let restored: Actor = serde_json::from_str(&json).unwrap();
    assert_eq!(actor, restored);
}

#[test]
fn all_enum_variants_serialise_correctly() {
    assert_eq!(
        serde_json::to_string(&SignalClass::Trace).unwrap(),
        "\"trace\""
    );
    assert_eq!(
        serde_json::to_string(&SignalClass::Decision).unwrap(),
        "\"decision\""
    );
    assert_eq!(
        serde_json::to_string(&ActorType::Pipeline).unwrap(),
        "\"pipeline\""
    );
    assert_eq!(
        serde_json::to_string(&Reproducibility::LossyCapped).unwrap(),
        "\"lossy_capped\""
    );
    assert_eq!(
        serde_json::to_string(&RetentionClass::SovereignArchive).unwrap(),
        "\"sovereign_archive\""
    );
    assert_eq!(
        serde_json::to_string(&Channel::Evidence).unwrap(),
        "\"evidence\""
    );
    assert_eq!(
        serde_json::to_string(&SignatureAlg::HybridEd25519Dilithium3).unwrap(),
        "\"hybrid-ed25519-dilithium3\""
    );
}

#[test]
fn schemars_generates_schema() {
    let schema = schemars::schema_for!(JsonSignalEnvelope);
    let _json = serde_json::to_string_pretty(&schema).unwrap();
    // Smoke-test: ensure the schema serialises without panic.
}

#[test]
fn otel_urn_to_trace_id_is_stable() {
    let urn = OisUrn::new("urn:ois:exec:acme:build-42").unwrap();
    let id1 = otel::urn_to_trace_id(&urn);
    let id2 = otel::urn_to_trace_id(&urn);
    assert_eq!(id1, id2);
}

#[test]
fn otel_urn_to_span_id_is_stable() {
    let urn = OisUrn::new("urn:ois:signal:acme:build-42:trace:1").unwrap();
    let id1 = otel::urn_to_span_id(&urn);
    let id2 = otel::urn_to_span_id(&urn);
    assert_eq!(id1, id2);
}

#[test]
fn otel_attributes_contain_required_fields() {
    let envelope = SignalEnvelope::builder()
        .signal_id("urn:ois:signal:acme:exec-7:trace:1".parse().unwrap())
        .signal_class(SignalClass::Trace)
        .exec_id("urn:ois:exec:acme:exec-7".parse().unwrap())
        .tenant_id("urn:ois:tenant:acme".parse().unwrap())
        .actor(Actor::new(
            "urn:ois:agent:acme:compiler".parse().unwrap(),
            ActorType::Agent,
        ))
        .reproducibility(Reproducibility::Deterministic)
        .payload(json!({ "kernel": "matmul", "grid": [4, 1, 1] }))
        .build()
        .unwrap();

    let attrs = otel::to_otel_attributes(&envelope);
    let keys: Vec<String> = attrs.iter().map(|kv| kv.key.as_str().to_string()).collect();

    assert!(keys.contains(&"ois.signal_id".to_string()));
    assert!(keys.contains(&"ois.signal_class".to_string()));
    assert!(keys.contains(&"ois.exec_id".to_string()));
    assert!(keys.contains(&"ois.tenant_id".to_string()));
    assert!(keys.contains(&"ois.actor.id".to_string()));
    assert!(keys.contains(&"ois.reproducibility".to_string()));
    assert!(keys.contains(&"ois.payload.kernel".to_string()));
}

#[test]
fn otel_resource_attributes_focus_on_identity() {
    let envelope = SignalEnvelope::builder()
        .signal_id("urn:ois:signal:acme:exec-7:trace:1".parse().unwrap())
        .signal_class(SignalClass::Trace)
        .exec_id("urn:ois:exec:acme:exec-7".parse().unwrap())
        .tenant_id("urn:ois:tenant:acme".parse().unwrap())
        .actor(Actor::new(
            "urn:ois:agent:acme:compiler".parse().unwrap(),
            ActorType::Agent,
        ))
        .reproducibility(Reproducibility::Deterministic)
        .payload(json!({}))
        .build()
        .unwrap();

    let attrs = otel::to_resource_attributes(&envelope);
    let keys: Vec<String> = attrs.iter().map(|kv| kv.key.as_str().to_string()).collect();

    assert!(keys.contains(&"ois.exec_id".to_string()));
    assert!(keys.contains(&"ois.tenant_id".to_string()));
    assert!(!keys.contains(&"ois.signal_id".to_string()));
}

#[test]
fn otel_span_kind_mapping() {
    assert_eq!(
        otel::signal_class_to_span_kind(SignalClass::Trace),
        opentelemetry::trace::SpanKind::Internal
    );
    assert_eq!(
        otel::signal_class_to_span_kind(SignalClass::Stream),
        opentelemetry::trace::SpanKind::Consumer
    );
    assert_eq!(
        otel::signal_class_to_span_kind(SignalClass::Decision),
        opentelemetry::trace::SpanKind::Server
    );
    assert_eq!(
        otel::signal_class_to_span_kind(SignalClass::Receipt),
        opentelemetry::trace::SpanKind::Producer
    );
}
