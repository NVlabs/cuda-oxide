/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attributes belonging to the LLVM dialect.

use std::fmt::Display;
use std::hash::{Hash, Hasher};

use combine::parser::char::spaces;
use combine::{Parser, choice};
use thiserror::Error;

use pliron::attribute::Attribute;
use pliron::builtin::attr_interfaces::{FloatAttr, TypedAttrInterface};
use pliron::builtin::attributes::IntegerAttr;
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::derive::{attr_interface_impl, format, pliron_attr};
use pliron::location::Located;
use pliron::parsable::{IntoParseResult, Parsable};
use pliron::printable::Printable;
use pliron::result::Result;
use pliron::r#type::{TypeObj, Typed};
use pliron::utils::apfloat::{self, Float, GetSemantics};
use pliron::{impl_printable_for_display, input_error, verify_err_noloc};

use crate::types::HalfType;

#[pliron_attr(name = "llvm.half_attr", format = "$0", verifier = "succ")]
#[derive(PartialEq, Clone, Debug)]
pub struct FPHalfAttr(pub apfloat::Half);

impl FPHalfAttr {
    pub fn from_bits(bits: u16) -> Self {
        Self(<apfloat::Half as Float>::from_bits(bits as u128))
    }

    pub fn to_bits(&self) -> u16 {
        self.0.to_bits() as u16
    }
}

impl Hash for FPHalfAttr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_bits().hash(state);
    }
}

impl Typed for FPHalfAttr {
    fn get_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        HalfType::get(ctx).into()
    }
}

#[attr_interface_impl]
impl TypedAttrInterface for FPHalfAttr {
    fn get_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        HalfType::get(ctx).into()
    }
}

#[attr_interface_impl]
impl FloatAttr for FPHalfAttr {
    fn get_inner(&self) -> &dyn apfloat::DynFloat {
        &self.0
    }

    fn build_from(&self, df: Box<dyn apfloat::DynFloat>) -> Box<dyn FloatAttr> {
        let df = df
            .downcast::<apfloat::Half>()
            .expect("Expected a half precision float");
        Box::new(Self(*df))
    }

    fn get_semantics(&self) -> apfloat::Semantics {
        Self::get_semantics_static()
    }

    fn get_semantics_static() -> apfloat::Semantics
    where
        Self: Sized,
    {
        <apfloat::Half as GetSemantics>::get_semantics()
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct FastmathFlags: u32 {
        const NNAN     = 1 << 1;
        const NINF     = 1 << 2;
        const NSZ      = 1 << 3;
        const ARCP     = 1 << 4;
        const CONTRACT = 1 << 5;
        const AFN      = 1 << 6;
        const REASSOC  = 1 << 7;
        const FAST     = Self::NNAN.bits() | Self::NINF.bits() | Self::NSZ.bits() | Self::ARCP.bits() | Self::CONTRACT.bits() | Self::AFN.bits() | Self::REASSOC.bits();
    }
}

/// Integer overflow flags for arithmetic operations.
/// The description below is from LLVM's
/// [release notes](https://releases.llvm.org/2.6/docs/ReleaseNotes.html)
/// that added the flags.
/// "nsw" and "nuw" bits indicate that the operation is guaranteed to not overflow
/// (in the signed or unsigned case, respectively). This gives the optimizer more information
///  and can be used for things like C signed integer values, which are undefined on overflow.
#[pliron_attr(name = "llvm.integer_overlflow_flags", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Default, Hash)]
pub struct IntegerOverflowFlagsAttr {
    pub nsw: bool,
    pub nuw: bool,
}

#[pliron_attr(name = "llvm.fast_math_flags", verifier = "succ")]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct FastmathFlagsAttr(pub FastmathFlags);

impl Default for FastmathFlagsAttr {
    fn default() -> Self {
        Self(FastmathFlags::empty())
    }
}

impl From<FastmathFlags> for FastmathFlagsAttr {
    fn from(value: FastmathFlags) -> Self {
        Self(value)
    }
}

impl From<FastmathFlagsAttr> for FastmathFlags {
    fn from(attr: FastmathFlagsAttr) -> Self {
        attr.0
    }
}

impl Display for FastmathFlagsAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<")?;
        bitflags::parser::to_writer(&self.0, &mut *f)?;
        write!(f, ">")
    }
}

impl_printable_for_display!(FastmathFlagsAttr);

#[derive(Debug, Error)]
#[error("Error parsing fastmath flags: {0}")]
pub struct FastmathFlagParseErr(pub bitflags::parser::ParseError);

impl Parsable for FastmathFlagsAttr {
    type Arg = ();

    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        _arg: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        let pos = state_stream.loc();
        let allowed_chars = combine::choice!(
            combine::parser::char::space().map(|c| c.to_string()),
            combine::parser::char::alpha_num().map(|c| c.to_string()),
            combine::parser::char::char('|').map(|c: char| c.to_string())
        );

        let (parsed, _): (Vec<String>, _) = combine::between(
            combine::parser::char::char('<').with(spaces()),
            spaces().with(combine::parser::char::char('>')),
            combine::many(allowed_chars),
        )
        .parse_stream(state_stream)
        .into_result()?;
        let parsed_string = parsed.concat();

        let (fast_math_flags, _) = bitflags::parser::from_str::<FastmathFlags>(&parsed_string)
            .map_err(|e| input_error!(pos.clone(), FastmathFlagParseErr(e)))
            .into_parse_result()?;

        Ok(Self(fast_math_flags)).into_parse_result()
    }
}

#[pliron_attr(name = "llvm.icmp_predicate", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ICmpPredicateAttr {
    EQ,
    NE,
    SLT,
    SLE,
    SGT,
    SGE,
    ULT,
    ULE,
    UGT,
    UGE,
}

#[pliron_attr(name = "llvm.fcmp_predicate", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum FCmpPredicateAttr {
    False,
    OEQ,
    OGT,
    OGE,
    OLT,
    OLE,
    ONE,
    ORD,
    UEQ,
    UGT,
    UGE,
    ULT,
    ULE,
    UNE,
    UNO,
    True,
}

/// An index for a GEP can be either a constant or an SSA operand.
/// Contrary to its name, this isn't an [Attribute][pliron::attribute::Attribute].
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
#[format]
pub enum GepIndexAttr {
    /// This GEP index is a raw u32 compile time constant
    Constant(u32),
    /// This GEP Index is the SSA value in the containing
    /// [Operation](pliron::operation::Operation)s `operands[idx]`
    OperandIdx(usize),
}

#[pliron_attr(
    name = "llvm.gep_indices",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct GepIndicesAttr(pub Vec<GepIndexAttr>);

/// An attribute that contains a list of case values for a switch operation.
#[pliron_attr(name = "llvm.case_values", format = "`[` vec($0, CharSpace(`,`)) `]`")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CaseValuesAttr(pub Vec<IntegerAttr>);

#[derive(Debug, Error)]
#[error("Case values must be of the same type, but found different types: {0} and {1}")]
pub struct CaseValuesAttrVerifyErr(pub String, pub String);

impl Verify for CaseValuesAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.0.windows(2).try_for_each(|pair| {
            pair[0].verify(ctx)?;
            if pair[0].get_type() == pair[1].get_type() {
                Ok(())
            } else {
                verify_err_noloc!(CaseValuesAttrVerifyErr(
                    pair[0].get_type().disp(ctx).to_string(),
                    pair[1].get_type().disp(ctx).to_string()
                ))
            }
        })
    }
}

#[pliron_attr(name = "llvm.linkage", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LinkageAttr {
    ExternalLinkage,
    AvailableExternallyLinkage,
    LinkOnceAnyLinkage,
    LinkOnceODRLinkage,
    LinkOnceODRAutoHideLinkage,
    WeakAnyLinkage,
    WeakODRLinkage,
    AppendingLinkage,
    InternalLinkage,
    PrivateLinkage,
    DLLImportLinkage,
    DLLExportLinkage,
    ExternalWeakLinkage,
    GhostLinkage,
    CommonLinkage,
    LinkerPrivateLinkage,
    LinkerPrivateWeakLinkage,
}

// =============================================================================
// Atomic Attributes
// =============================================================================

/// LLVM atomic memory ordering.
///
/// Maps directly to LLVM IR ordering keywords.
#[pliron_attr(name = "llvm.atomic_ordering", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LlvmAtomicOrdering {
    Monotonic,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// LLVM syncscope for atomic operations.
///
/// Maps to LLVM IR `syncscope("...")` qualifiers:
///
/// | Variant | LLVM IR                      | PTX scope |
/// |---------|------------------------------|-----------|
/// | System  | (default, no syncscope)      | `.sys`    |
/// | Device  | `syncscope("device")`        | `.gpu`    |
/// | Block   | `syncscope("block")`         | `.cta`    |
#[pliron_attr(name = "llvm.sync_scope", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LlvmSyncScope {
    System,
    Device,
    Block,
}

/// LLVM atomicrmw operation kind.
///
/// Maps directly to LLVM IR `atomicrmw` operation keywords.
#[pliron_attr(name = "llvm.atomic_rmw_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LlvmAtomicRmwKind {
    Add,
    Sub,
    Xchg,
    And,
    Or,
    Xor,
    Max,
    Min,
    UMax,
    UMin,
    FAdd,
}

pub fn register(ctx: &mut Context) {
    FPHalfAttr::register(ctx);
    IntegerOverflowFlagsAttr::register(ctx);
    ICmpPredicateAttr::register(ctx);
    FCmpPredicateAttr::register(ctx);
    GepIndicesAttr::register(ctx);
    CaseValuesAttr::register(ctx);
    InsertExtractValueIndicesAttr::register(ctx);
    AlignmentAttr::register(ctx);
    GlobalAddressSpaceAttr::register(ctx);
    FastmathFlagsAttr::register(ctx);
    LinkageAttr::register(ctx);
    LlvmAtomicOrdering::register(ctx);
    LlvmSyncScope::register(ctx);
    LlvmAtomicRmwKind::register(ctx);
}

#[pliron_attr(
    name = "llvm.insert_extract_value_indices",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct InsertExtractValueIndicesAttr(pub Vec<u32>);

/// LLVM alignment attribute for memory operations and globals.
#[pliron_attr(name = "llvm.align", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct AlignmentAttr(pub u32);

/// LLVM global address-space attribute.
#[pliron_attr(name = "llvm.global_addrspace", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct GlobalAddressSpaceAttr(pub u32);

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use pliron::{
        location,
        parsable::{self, state_stream_from_iterator},
    };

    use super::*;

    #[test]
    fn test_fastmath_flags_attr_empty() {
        let flags = FastmathFlags::empty();
        assert_eq!(flags.bits(), 0);

        let ctx = &mut Context::default();
        let flags_attr: FastmathFlagsAttr = flags.into();
        expect!["<>"].assert_eq(&flags_attr.disp(ctx).to_string());

        let input = "<>";
        let mut state_stream = state_stream_from_iterator(
            input.chars(),
            parsable::State::new(ctx, location::Source::InMemory),
        );
        let (parsed, _) = FastmathFlagsAttr::parse(&mut state_stream, ()).unwrap();
        assert_eq!(parsed, flags_attr);
    }

    #[test]
    fn test_fastmath_flags_attr_set_flags() {
        let mut flags = FastmathFlags::empty();
        flags |= FastmathFlags::NNAN | FastmathFlags::NINF;
        assert!(flags.contains(FastmathFlags::NNAN));
        assert!(flags.contains(FastmathFlags::NINF));
        assert!(!flags.contains(FastmathFlags::NSZ));
    }

    #[test]
    fn test_fastmath_flags_attr_fmt() {
        let ctx = &Context::default();
        let flags: FastmathFlagsAttr = (FastmathFlags::NNAN | FastmathFlags::ARCP).into();
        expect!["<NNAN | ARCP>"].assert_eq(&flags.disp(ctx).to_string());
    }

    #[test]
    fn test_fastmath_flags_attr_fmt_fast() {
        let ctx = &Context::default();
        let flags: FastmathFlagsAttr = FastmathFlags::FAST.into();
        expect!["<NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC>"]
            .assert_eq(&flags.disp(ctx).to_string());
    }

    #[test]
    fn test_fastmath_flags_attr_parse_valid() {
        let ctx = &mut Context::default();

        let input = "<NNAN | ARCP>";
        let mut state_stream = state_stream_from_iterator(
            input.chars(),
            parsable::State::new(ctx, location::Source::InMemory),
        );
        let (parsed, _) = FastmathFlagsAttr::parse(&mut state_stream, ()).unwrap();
        assert!(parsed.0.contains(FastmathFlags::NNAN));
        assert!(parsed.0.contains(FastmathFlags::ARCP));
    }

    // Test input with FAST flag set
    #[test]
    fn test_fastmath_flags_attr_parse_fast() {
        let ctx = &mut Context::default();

        let input = "<FAST>";
        let mut state_stream = state_stream_from_iterator(
            input.chars(),
            parsable::State::new(ctx, location::Source::InMemory),
        );
        let (parsed, _) = FastmathFlagsAttr::parse(&mut state_stream, ()).unwrap();
        assert!(parsed.0.contains(FastmathFlags::FAST));

        // FAST also means all the other flags.
        assert!(parsed.0.contains(FastmathFlags::NNAN));
        assert!(parsed.0.contains(FastmathFlags::NINF));
        assert!(parsed.0.contains(FastmathFlags::NSZ));
        assert!(parsed.0.contains(FastmathFlags::ARCP));
        assert!(parsed.0.contains(FastmathFlags::CONTRACT));
        assert!(parsed.0.contains(FastmathFlags::REASSOC));
    }

    #[test]
    fn test_fastmath_flags_attr_parse_invalid() {
        let ctx = &mut Context::default();
        let input = "<INVALIDFLAG>";
        let state_stream = state_stream_from_iterator(
            input.chars(),
            parsable::State::new(ctx, location::Source::InMemory),
        );
        match FastmathFlagsAttr::parser(()).parse(state_stream) {
            Ok((parsed, _)) => {
                panic!("Expected error, but got: {}", parsed);
            }
            Err(e) => {
                expect![[r#"
                    Parse error at line: 1, column: 1
                    Error parsing fastmath flags: unrecognized named flag
                "#]]
                .assert_eq(&e.to_string());
            }
        }
    }
}
