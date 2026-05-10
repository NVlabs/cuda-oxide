/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! OIS (Oxide Instrumentation Signal) canonical envelope types.
//!
//! This module provides the Rust type system for the canonical OIS signal envelope
//! defined by `https://nvidia.com/oxide/ois/v1.0/schema/envelope/canonical-signal-envelope.schema.json`.
//! Every emitted signal MUST validate against this schema.

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// An OIS URN string that is guaranteed to start with `urn:ois:`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "String", into = "String")]
#[schemars(description = "OIS URN — must start with 'urn:ois:'")]
pub struct OisUrn(String);

impl OisUrn {
    /// Creates a new `OisUrn` after validating the prefix.
    pub fn new(s: impl Into<String>) -> Result<Self, ValidationError> {
        let s = s.into();
        if s.starts_with("urn:ois:") {
            Ok(Self(s))
        } else {
            Err(ValidationError::InvalidUrnPrefix(s))
        }
    }

    /// Returns the underlying string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for OisUrn {
    type Err = ValidationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<String> for OisUrn {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<OisUrn> for String {
    fn from(value: OisUrn) -> Self {
        value.0
    }
}

impl fmt::Display for OisUrn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Errors that can arise while constructing or validating an OIS envelope.
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    /// The provided string does not start with `urn:ois:`.
    #[error("URN must start with 'urn:ois:', got: {0}")]
    InvalidUrnPrefix(String),
    /// A required envelope field was not supplied.
    #[error("required field missing: {0}")]
    MissingField(&'static str),
    /// The signal version is not `"1.0"`.
    #[error("signal_version must be '1.0', got: {0}")]
    InvalidVersion(String),
}

/// Classification of an OIS signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(description = "Signal class")]
#[allow(missing_docs)]
pub enum SignalClass {
    Trace,
    Decision,
    Policy,
    Receipt,
    Evidence,
    State,
    Stream,
    Metric,
    Log,
}

impl fmt::Display for SignalClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SignalClass::Trace => "trace",
            SignalClass::Decision => "decision",
            SignalClass::Policy => "policy",
            SignalClass::Receipt => "receipt",
            SignalClass::Evidence => "evidence",
            SignalClass::State => "state",
            SignalClass::Stream => "stream",
            SignalClass::Metric => "metric",
            SignalClass::Log => "log",
        };
        f.write_str(s)
    }
}

/// The type of actor that emitted a signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(description = "Actor type")]
#[allow(missing_docs)]
pub enum ActorType {
    Human,
    Agent,
    Service,
    Runtime,
    Pipeline,
}

impl fmt::Display for ActorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ActorType::Human => "human",
            ActorType::Agent => "agent",
            ActorType::Service => "service",
            ActorType::Runtime => "runtime",
            ActorType::Pipeline => "pipeline",
        };
        f.write_str(s)
    }
}

/// Actor metadata for an OIS signal.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[schemars(description = "Actor")]
pub struct Actor {
    /// Actor URN.
    pub id: OisUrn,
    /// Actor type (human, agent, service, runtime, pipeline).
    #[serde(rename = "type")]
    pub actor_type: ActorType,
    /// Ordered delegation chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delegate_chain: Option<Vec<OisUrn>>,
}

impl Actor {
    /// Constructs a new `Actor` without a delegate chain.
    pub fn new(id: OisUrn, actor_type: ActorType) -> Self {
        Self {
            id,
            actor_type,
            delegate_chain: None,
        }
    }

    /// Sets the delegate chain.
    pub fn with_delegate_chain(mut self, chain: Vec<OisUrn>) -> Self {
        self.delegate_chain = Some(chain);
        self
    }
}

/// Reproducibility tier of a signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(description = "Reproducibility")]
#[allow(missing_docs)]
pub enum Reproducibility {
    Exact,
    Deterministic,
    Attested,
    Explanatory,
    BestEffort,
    LossyCapped,
}

impl fmt::Display for Reproducibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Reproducibility::Exact => "exact",
            Reproducibility::Deterministic => "deterministic",
            Reproducibility::Attested => "attested",
            Reproducibility::Explanatory => "explanatory",
            Reproducibility::BestEffort => "best_effort",
            Reproducibility::LossyCapped => "lossy_capped",
        };
        f.write_str(s)
    }
}

/// Retention classification governing lifecycle and compliance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(description = "Retention class")]
#[allow(missing_docs)]
pub enum RetentionClass {
    Ephemeral,
    Standard,
    Regulated,
    LegalHold,
    SovereignArchive,
}

impl fmt::Display for RetentionClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            RetentionClass::Ephemeral => "ephemeral",
            RetentionClass::Standard => "standard",
            RetentionClass::Regulated => "regulated",
            RetentionClass::LegalHold => "legal_hold",
            RetentionClass::SovereignArchive => "sovereign_archive",
        };
        f.write_str(s)
    }
}

/// Communication channel over which the signal is carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(description = "Channel")]
#[allow(missing_docs)]
pub enum Channel {
    Control,
    Data,
    Proof,
    Recall,
    Evidence,
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Channel::Control => "control",
            Channel::Data => "data",
            Channel::Proof => "proof",
            Channel::Recall => "recall",
            Channel::Evidence => "evidence",
        };
        f.write_str(s)
    }
}

/// Cryptographic signature attached to a signal.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[schemars(description = "Signature")]
pub struct Signature {
    /// Signature algorithm.
    pub alg: SignatureAlg,
    /// Base64-encoded signature bytes.
    pub value: String,
    /// Reference to the signing key URN.
    pub key_ref: OisUrn,
    /// Optional Rekor transparency-log entry URI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rekor_uri: Option<String>,
}

/// Supported signature algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[schemars(description = "Signature algorithm")]
#[allow(missing_docs)]
pub enum SignatureAlg {
    Ed25519,
    EcdsaP256,
    Dilithium3,
    HybridEd25519Dilithium3,
}

impl fmt::Display for SignatureAlg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SignatureAlg::Ed25519 => "ed25519",
            SignatureAlg::EcdsaP256 => "ecdsa-p256",
            SignatureAlg::Dilithium3 => "dilithium3",
            SignatureAlg::HybridEd25519Dilithium3 => "hybrid-ed25519-dilithium3",
        };
        f.write_str(s)
    }
}

/// The canonical OIS signal envelope.
///
/// `Payload` is generic so that downstream crates can use strongly-typed
/// payloads while still conforming to the envelope schema. Use
/// `JsonSignalEnvelope` when you need an untyped / JSON-object payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(description = "Canonical Signal Envelope")]
pub struct SignalEnvelope<Payload = serde_json::Value> {
    /// Unique signal URN.
    pub signal_id: OisUrn,
    /// Signal classification.
    pub signal_class: SignalClass,
    /// Envelope schema version (always `"1.0"`).
    #[serde(default = "default_version")]
    pub signal_version: String,
    /// URI of the payload-specific JSON schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_uri: Option<String>,
    /// Execution context URN.
    pub exec_id: OisUrn,
    /// Tenant / organisation URN.
    pub tenant_id: OisUrn,
    /// Wall-clock time when the signal was emitted (RFC3339).
    pub emitted_at: DateTime<Utc>,
    /// Wall-clock time when the signal was captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured_at: Option<DateTime<Utc>>,
    /// Actor that emitted the signal.
    pub actor: Actor,
    /// Reproducibility tier.
    pub reproducibility: Reproducibility,
    /// Retention classification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention_class: Option<RetentionClass>,
    /// Communication channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<Channel>,
    /// Arbitrary payload object.
    pub payload: Payload,
    /// Cryptographic signature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<Signature>,
    /// Parent signal URNs (causal lineage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage_refs: Option<Vec<OisUrn>>,
}

fn default_version() -> String {
    "1.0".to_string()
}

/// Type alias for a signal envelope whose payload is an arbitrary JSON object.
pub type JsonSignalEnvelope = SignalEnvelope<serde_json::Value>;

impl<Payload> SignalEnvelope<Payload> {
    /// Returns a builder for constructing a `SignalEnvelope`.
    pub fn builder() -> SignalEnvelopeBuilder<Payload> {
        SignalEnvelopeBuilder {
            signal_id: None,
            signal_class: None,
            signal_version: None,
            schema_uri: None,
            exec_id: None,
            tenant_id: None,
            emitted_at: None,
            captured_at: None,
            actor: None,
            reproducibility: None,
            retention_class: None,
            channel: None,
            payload: None,
            signature: None,
            lineage_refs: None,
        }
    }
}

impl JsonSignalEnvelope {
    /// Serialises the envelope to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialises an envelope from a JSON string.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Builder for [`SignalEnvelope`].
pub struct SignalEnvelopeBuilder<Payload = serde_json::Value> {
    signal_id: Option<OisUrn>,
    signal_class: Option<SignalClass>,
    signal_version: Option<String>,
    schema_uri: Option<String>,
    exec_id: Option<OisUrn>,
    tenant_id: Option<OisUrn>,
    emitted_at: Option<DateTime<Utc>>,
    captured_at: Option<DateTime<Utc>>,
    actor: Option<Actor>,
    reproducibility: Option<Reproducibility>,
    retention_class: Option<RetentionClass>,
    channel: Option<Channel>,
    payload: Option<Payload>,
    signature: Option<Signature>,
    lineage_refs: Option<Vec<OisUrn>>,
}

#[allow(missing_docs)]
impl<Payload> SignalEnvelopeBuilder<Payload> {
    pub fn signal_id(mut self, id: OisUrn) -> Self {
        self.signal_id = Some(id);
        self
    }

    pub fn signal_class(mut self, class: SignalClass) -> Self {
        self.signal_class = Some(class);
        self
    }

    pub fn signal_version(mut self, version: impl Into<String>) -> Self {
        self.signal_version = Some(version.into());
        self
    }

    pub fn schema_uri(mut self, uri: impl Into<String>) -> Self {
        self.schema_uri = Some(uri.into());
        self
    }

    pub fn exec_id(mut self, id: OisUrn) -> Self {
        self.exec_id = Some(id);
        self
    }

    pub fn tenant_id(mut self, id: OisUrn) -> Self {
        self.tenant_id = Some(id);
        self
    }

    pub fn emitted_at(mut self, at: DateTime<Utc>) -> Self {
        self.emitted_at = Some(at);
        self
    }

    pub fn captured_at(mut self, at: DateTime<Utc>) -> Self {
        self.captured_at = Some(at);
        self
    }

    pub fn actor(mut self, actor: Actor) -> Self {
        self.actor = Some(actor);
        self
    }

    pub fn reproducibility(mut self, repro: Reproducibility) -> Self {
        self.reproducibility = Some(repro);
        self
    }

    pub fn retention_class(mut self, rc: RetentionClass) -> Self {
        self.retention_class = Some(rc);
        self
    }

    pub fn channel(mut self, ch: Channel) -> Self {
        self.channel = Some(ch);
        self
    }

    pub fn payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn signature(mut self, sig: Signature) -> Self {
        self.signature = Some(sig);
        self
    }

    pub fn lineage_refs(mut self, refs: Vec<OisUrn>) -> Self {
        self.lineage_refs = Some(refs);
        self
    }
}

impl<Payload> SignalEnvelopeBuilder<Payload> {
    /// Consumes the builder and returns a [`SignalEnvelope`].
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::MissingField`] if any required field was not
    /// supplied, or [`ValidationError::InvalidVersion`] if an explicit
    /// `signal_version` was provided and is not `"1.0"`.
    pub fn build(self) -> Result<SignalEnvelope<Payload>, ValidationError> {
        let signal_version = self.signal_version.unwrap_or_else(|| "1.0".to_string());
        if signal_version != "1.0" {
            return Err(ValidationError::InvalidVersion(signal_version));
        }

        Ok(SignalEnvelope {
            signal_id: self
                .signal_id
                .ok_or(ValidationError::MissingField("signal_id"))?,
            signal_class: self
                .signal_class
                .ok_or(ValidationError::MissingField("signal_class"))?,
            signal_version,
            schema_uri: self.schema_uri,
            exec_id: self
                .exec_id
                .ok_or(ValidationError::MissingField("exec_id"))?,
            tenant_id: self
                .tenant_id
                .ok_or(ValidationError::MissingField("tenant_id"))?,
            emitted_at: self.emitted_at.unwrap_or_else(Utc::now),
            captured_at: self.captured_at,
            actor: self.actor.ok_or(ValidationError::MissingField("actor"))?,
            reproducibility: self
                .reproducibility
                .ok_or(ValidationError::MissingField("reproducibility"))?,
            retention_class: self.retention_class,
            channel: self.channel,
            payload: self
                .payload
                .ok_or(ValidationError::MissingField("payload"))?,
            signature: self.signature,
            lineage_refs: self.lineage_refs,
        })
    }
}
