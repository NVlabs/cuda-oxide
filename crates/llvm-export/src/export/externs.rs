/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device extern declaration types for FFI with external LTOIR.

use std::fmt::Write;

/// A device-extern type that preserves pointer pointees and address spaces.
///
/// The lowered module uses opaque pointers, so this separate type description
/// is needed to emit declarations such as `declare void @f(float*)` for
/// pre-Blackwell libNVVM. Unsupported extern signatures are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceExternType {
    /// Only valid as a function result.
    Void,
    /// A signless LLVM integer. Signedness is not part of an LLVM integer type.
    Integer(u32),
    /// An integer parameter that must be sign-extended to the given width.
    ///
    /// On NVPTX, i8/i16 signed values are promoted to i32 with `signext`.
    /// The width stored here is the promoted width (e.g., 32), not the
    /// original source width.
    SignExtInteger(u32),
    /// An integer parameter that must be zero-extended to the given width.
    ///
    /// On NVPTX, u8/u16/bool values are promoted to i32 with `zeroext`.
    /// The width stored here is the promoted width (e.g., 32), not the
    /// original source width.
    ZeroExtInteger(u32),
    Float16,
    Float32,
    Float64,
    /// A pointer with an exact pointee and NVVM address space.
    Pointer {
        pointee: Box<DeviceExternType>,
        address_space: u32,
    },
    /// A fixed-size array. Arrays are supported as pointer pointees; passing an
    /// array by value is not yet supported.
    Array {
        element: Box<DeviceExternType>,
        len: u64,
    },
}

impl DeviceExternType {
    pub fn pointer_to(pointee: DeviceExternType, address_space: u32) -> Self {
        Self::Pointer {
            pointee: Box::new(pointee),
            address_space,
        }
    }

    pub fn pointer_parts(&self) -> Option<(&DeviceExternType, u32)> {
        match self {
            Self::Pointer {
                pointee,
                address_space,
            } => Some((pointee, *address_space)),
            _ => None,
        }
    }

    /// Return the integer width for any integer variant (plain, signext, zeroext).
    pub fn integer_width(&self) -> Option<u32> {
        match self {
            Self::Integer(bits) | Self::SignExtInteger(bits) | Self::ZeroExtInteger(bits) => {
                Some(*bits)
            }
            _ => None,
        }
    }

    /// True when this legacy pointer is already the internal byte-pointer type.
    pub(crate) fn is_canonical_byte_pointer(&self) -> bool {
        matches!(
            self,
            Self::Pointer { pointee, .. }
                if matches!(pointee.as_ref(), Self::Integer(8))
        )
    }

    pub(crate) fn contains_float16(&self) -> bool {
        match self {
            Self::Float16 => true,
            Self::Pointer { pointee, .. } => pointee.contains_float16(),
            Self::Array { element, .. } => element.contains_float16(),
            _ => false,
        }
    }

    /// Write the LLVM IR type string for this type (without ABI attributes).
    ///
    /// For `SignExtInteger` and `ZeroExtInteger`, this writes just the `iN`
    /// type. The `signext`/`zeroext` attributes are emitted separately by the
    /// declaration emitter.
    ///
    /// With modern LLVM syntax, pointer pointees are omitted while address
    /// spaces are retained.
    pub(crate) fn write_llvm(
        &self,
        output: &mut String,
        legacy_typed_pointers: bool,
    ) -> Result<(), String> {
        match self {
            Self::Void => write!(output, "void").unwrap(),
            Self::Integer(bits) | Self::SignExtInteger(bits) | Self::ZeroExtInteger(bits)
                if *bits > 0 =>
            {
                write!(output, "i{bits}").unwrap()
            }
            Self::Integer(_) | Self::SignExtInteger(_) | Self::ZeroExtInteger(_) => {
                return Err("device-extern integer width must be non-zero".to_string());
            }
            Self::Float16 => write!(output, "half").unwrap(),
            Self::Float32 => write!(output, "float").unwrap(),
            Self::Float64 => write!(output, "double").unwrap(),
            Self::Pointer {
                pointee,
                address_space,
            } => {
                if matches!(pointee.as_ref(), Self::Void) {
                    return Err(
                        "device-extern pointer cannot have LLVM `void` as its pointee; use i8"
                            .to_string(),
                    );
                }
                if legacy_typed_pointers {
                    pointee.write_llvm(output, true)?;
                    if *address_space != 0 {
                        write!(output, " addrspace({address_space})").unwrap();
                    }
                    write!(output, "*").unwrap();
                } else if *address_space == 0 {
                    write!(output, "ptr").unwrap();
                } else {
                    write!(output, "ptr addrspace({address_space})").unwrap();
                }
            }
            Self::Array { element, len } => {
                if matches!(element.as_ref(), Self::Void) {
                    return Err("device-extern array element cannot be `void`".to_string());
                }
                write!(output, "[{len} x ").unwrap();
                element.write_llvm(output, legacy_typed_pointers)?;
                write!(output, "]").unwrap();
            }
        }
        Ok(())
    }

    /// Write the LLVM IR type string with any required ABI attribute prefix.
    ///
    /// For `SignExtInteger(32)` this writes `i32 signext`, for
    /// `ZeroExtInteger(32)` this writes `i32 zeroext`. Plain types write
    /// just the type.
    ///
    /// The attribute is written *after* the type for parameter and return
    /// positions in LLVM IR declarations:
    ///   `declare void @f(i32 signext %a)`
    pub(crate) fn write_llvm_with_attr(
        &self,
        output: &mut String,
        legacy_typed_pointers: bool,
    ) -> Result<(), String> {
        match self {
            Self::SignExtInteger(bits) if *bits > 0 => {
                write!(output, "i{bits} signext").unwrap();
            }
            Self::ZeroExtInteger(bits) if *bits > 0 => {
                write!(output, "i{bits} zeroext").unwrap();
            }
            _ => self.write_llvm(output, legacy_typed_pointers)?,
        }
        Ok(())
    }

    pub(crate) fn llvm_string(&self, legacy_typed_pointers: bool) -> Result<String, String> {
        let mut output = String::new();
        self.write_llvm(&mut output, legacy_typed_pointers)?;
        Ok(output)
    }
}

/// An external device function declaration (for linking with external LTOIR).
///
/// These declarations are emitted as LLVM `declare` statements and resolved
/// at link time by nvJitLink when linking with external LTOIR (e.g., CCCL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceExternDecl {
    /// The export name (e.g., "cub_block_reduce_sum").
    pub export_name: String,

    /// Function parameter types, including pointer pointees and address spaces.
    pub param_types: Vec<DeviceExternType>,

    /// Return type.
    pub return_type: DeviceExternType,

    /// NVVM attributes for this function.
    pub attrs: DeviceExternAttrs,
}

/// NVVM attributes for device extern declarations.
///
/// NOTE: These attributes are currently **not emitted** to the LLVM IR output.
/// When linking LTOIR via nvJitLink, the external library's LTOIR already contains
/// proper attributes (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
/// nvJitLink uses the definition's attributes during LTO, making attributes on
/// declarations redundant.
///
/// This struct is retained for potential future use or for debugging/inspection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DeviceExternAttrs {
    /// Function is convergent (all threads must execute together).
    pub is_convergent: bool,

    /// Function is pure (no side effects). Maps to LLVM `readnone`.
    pub is_pure: bool,

    /// Function is read-only (only reads memory). Maps to LLVM `readonly`.
    pub is_readonly: bool,
}

/// Trait for types that can be converted to [`DeviceExternDecl`].
///
/// This allows mir-importer to pass its own DeviceExternDecl type
/// without llvm-export depending on mir-importer.
pub trait AsDeviceExtern {
    fn as_device_extern(&self) -> DeviceExternDecl;
}

impl AsDeviceExtern for DeviceExternDecl {
    fn as_device_extern(&self) -> DeviceExternDecl {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signext_integer_write_llvm() {
        let ty = DeviceExternType::SignExtInteger(32);
        assert_eq!(ty.llvm_string(false).unwrap(), "i32");

        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "i32 signext");
    }

    #[test]
    fn zeroext_integer_write_llvm() {
        let ty = DeviceExternType::ZeroExtInteger(32);
        assert_eq!(ty.llvm_string(false).unwrap(), "i32");

        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "i32 zeroext");
    }

    #[test]
    fn plain_integer_write_llvm_with_attr_has_no_attr() {
        let ty = DeviceExternType::Integer(32);
        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "i32");
    }

    #[test]
    fn float16_write_llvm_with_attr() {
        let ty = DeviceExternType::Float16;
        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "half");
    }

    #[test]
    fn integer_width_returns_bits_for_all_integer_variants() {
        assert_eq!(DeviceExternType::Integer(64).integer_width(), Some(64));
        assert_eq!(
            DeviceExternType::SignExtInteger(32).integer_width(),
            Some(32)
        );
        assert_eq!(
            DeviceExternType::ZeroExtInteger(32).integer_width(),
            Some(32)
        );
        assert_eq!(DeviceExternType::Float32.integer_width(), None);
    }

    #[test]
    fn signext_and_zeroext_are_distinct() {
        let s = DeviceExternType::SignExtInteger(32);
        let z = DeviceExternType::ZeroExtInteger(32);
        let p = DeviceExternType::Integer(32);
        assert_ne!(s, z);
        assert_ne!(s, p);
        assert_ne!(z, p);
    }
}
