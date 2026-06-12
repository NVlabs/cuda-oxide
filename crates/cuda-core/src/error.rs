/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA driver error types and result conversion utilities.
//!
//! [`DriverError`] wraps a raw `CUresult` code and implements `Display`,
//! [`Debug`], and [`Error`](std::error::Error) by querying the driver for
//! human-readable descriptions via `cuGetErrorName` / `cuGetErrorString`.
//!
//! [`IntoResult`] converts raw return tuples from the bindings into
//! idiomatic `Result<T, DriverError>`.

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// A CUDA driver error wrapping a raw [`CUresult`](cuda_bindings::CUresult) code.
///
/// The inner value is public so callers can match on specific
/// `cudaError_enum_*` constants when needed. Prefer the `Display` impl for
/// user-facing messages -- it calls into the driver to produce a description.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DriverError(pub cuda_bindings::CUresult);

impl DriverError {
    /// Shared formatting helper for both `Display` and [`Debug`].
    ///
    /// Attempts to resolve the error string via the driver; falls back to a
    /// placeholder if `cuGetErrorString` itself fails (e.g., driver not
    /// initialized).
    fn _fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self.error_string() {
            Ok(err_str) => formatter
                .debug_tuple("DriverError")
                .field(&self.0)
                .field(&err_str)
                .finish(),
            Err(_) => formatter
                .debug_tuple("DriverError")
                .field(&self.0)
                .field(&"<cuGetErrorString failed>")
                .finish(),
        }
    }

    /// Returns the short symbolic name for this error code (e.g.,
    /// `"CUDA_ERROR_INVALID_VALUE"`).
    ///
    /// Calls `cuGetErrorName`. Returns `Err` if the driver cannot resolve the
    /// code (e.g., it is from a newer driver version).
    pub fn error_name(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGetErrorName(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }

    /// Returns a human-readable description of this error code.
    ///
    /// Calls `cuGetErrorString`. The returned `CStr` is owned by the driver
    /// and valid for the lifetime of the loaded driver library.
    pub fn error_string(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGetErrorString(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }
}

impl Display for DriverError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self._fmt(f)
    }
}

impl std::fmt::Debug for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> fmt::Result {
        self._fmt(f)
    }
}

impl error::Error for DriverError {}

/// Converts a raw CUDA driver return value into `Result<T, DriverError>`.
///
/// Implemented for `CUresult` (void-returning calls) and for
/// `(CUresult, MaybeUninit<T>)` (calls that output a value through a pointer).
pub trait IntoResult<T> {
    /// Converts `self` into a `Result`, mapping `CUDA_SUCCESS` to `Ok`.
    fn result(self) -> Result<T, DriverError>
    where
        Self: Sized;
}

/// Converts a bare `CUresult` into `Result<(), DriverError>`.
impl IntoResult<()> for cuda_bindings::CUresult {
    fn result(self) -> Result<(), DriverError> {
        match self {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(()),
            _ => Err(DriverError(self)),
        }
    }
}

/// Converts a `(CUresult, MaybeUninit<T>)` pair into `Result<T, DriverError>`.
///
/// On success the `MaybeUninit` is assumed initialized and unwrapped. On
/// failure the uninitialized value is discarded safely.
impl<T> IntoResult<T> for (cuda_bindings::CUresult, MaybeUninit<T>) {
    fn result(self) -> Result<T, DriverError> {
        match self.0 {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(unsafe { self.1.assume_init() }),
            _ => Err(DriverError(self.0)),
        }
    }
}
