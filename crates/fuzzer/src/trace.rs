/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! FNV-1a 64-bit trace state shared between the CPU oracle and the GPU runs.
//!
//! The trace is a single `u64` global. Every interesting intermediate value is
//! folded into it byte-by-byte. The hash is the program's fingerprint: if both
//! backends are correct, the CPU `u64` and the GPU `u64` are equal.
//!
//! The state starts at zero because cuda-oxide currently only supports
//! zero-initialized device statics. Each run must call [`trace_reset`] before
//! executing the program, and [`trace_finish`] after.
//!
//! All trace functions are marked `#[inline]` so their MIR is encoded in the
//! `fuzzer` rlib and reachable to cuda-oxide's MIR collector when the smoke
//! example is compiled for the device. The `static mut RL_TRACE` already
//! prevents the optimizer from constant-folding the trace state away, so we
//! don't need `#[inline]` to keep the byte mixers separate.

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

static mut RL_TRACE: u64 = 0;

/// Reset the trace state to the FNV-1a 64-bit offset basis.
#[inline]
pub fn trace_reset() {
    unsafe {
        RL_TRACE = FNV_OFFSET;
    }
}

/// Read out the current trace state.
#[inline]
pub fn trace_finish() -> u64 {
    unsafe { RL_TRACE }
}

#[inline]
fn trace_write_byte(byte: u8) {
    unsafe {
        RL_TRACE = (RL_TRACE ^ byte as u64).wrapping_mul(FNV_PRIME);
    }
}

#[inline]
fn trace_write_u8(val: u8) {
    trace_write_byte(val);
}

#[inline]
fn trace_write_i8(val: i8) {
    trace_write_u8(val as u8);
}

#[inline]
fn trace_write_u16(val: u16) {
    trace_write_u8((val & 0xff) as u8);
    trace_write_u8(((val >> 8) & 0xff) as u8);
}

#[inline]
fn trace_write_i16(val: i16) {
    trace_write_u16(val as u16);
}

#[inline]
fn trace_write_u32(val: u32) {
    trace_write_u8((val & 0xff) as u8);
    trace_write_u8(((val >> 8) & 0xff) as u8);
    trace_write_u8(((val >> 16) & 0xff) as u8);
    trace_write_u8(((val >> 24) & 0xff) as u8);
}

#[inline]
fn trace_write_i32(val: i32) {
    trace_write_u32(val as u32);
}

#[inline]
fn trace_write_u64(val: u64) {
    trace_write_u32((val & 0xffff_ffff) as u32);
    trace_write_u32((val >> 32) as u32);
}

#[inline]
fn trace_write_i64(val: i64) {
    trace_write_u64(val as u64);
}

#[inline]
fn trace_write_u128(val: u128) {
    trace_write_u64((val & 0xffff_ffff_ffff_ffff) as u64);
    trace_write_u64((val >> 64) as u64);
}

#[inline]
fn trace_write_i128(val: i128) {
    trace_write_u128(val as u128);
}

#[inline]
fn trace_write_usize(val: usize) {
    trace_write_u64(val as u64);
}

#[inline]
fn trace_write_isize(val: isize) {
    trace_write_u64(val as u64);
}

#[inline]
fn trace_write_bool(val: bool) {
    trace_write_u8(val as u8);
}

#[inline]
fn trace_write_char(val: char) {
    trace_write_u32(val as u32);
}

/// Floats are folded as their bit patterns, so the trace stays an exact
/// comparison. A tolerance here would compare something other than the value
/// the backend produced, and the mismatches worth finding are the ones a
/// tolerance hides.
///
/// The bits agree for every non-NaN value the two backends produce. NaN
/// payload bits are not pinned down by Rust, so both writers canonicalize a
/// NaN to the quiet-NaN bit pattern before hashing, as Cranelift's fuzzgen
/// does. A payload divergence therefore cannot produce a false MISMATCH,
/// while a NaN against a non-NaN still hashes differently and remains a real
/// signal.
#[inline]
fn trace_write_f32(val: f32) {
    let bits = if val.is_nan() {
        f32::NAN.to_bits()
    } else {
        val.to_bits()
    };
    trace_write_u32(bits);
}

#[inline]
fn trace_write_f64(val: f64) {
    let bits = if val.is_nan() {
        f64::NAN.to_bits()
    } else {
        val.to_bits()
    };
    trace_write_u64(bits);
}

/// Scalar values that can be folded into the trace.
pub trait TraceValue {
    fn trace_write(self);
}

macro_rules! impl_trace_value {
    ($($ty:ty => $writer:ident),* $(,)?) => {
        $(
            impl TraceValue for $ty {
                #[inline]
                fn trace_write(self) {
                    $writer(self);
                }
            }
        )*
    };
}

impl_trace_value! {
    bool => trace_write_bool,
    i8 => trace_write_i8,
    i16 => trace_write_i16,
    i32 => trace_write_i32,
    i64 => trace_write_i64,
    i128 => trace_write_i128,
    isize => trace_write_isize,
    u8 => trace_write_u8,
    u16 => trace_write_u16,
    u32 => trace_write_u32,
    u64 => trace_write_u64,
    u128 => trace_write_u128,
    usize => trace_write_usize,
    char => trace_write_char,
    f32 => trace_write_f32,
    f64 => trace_write_f64,
}

/// Aggregates of scalar values that can be folded into the trace in one call.
///
/// `dump_var` accepts any `TraceDump`. We provide implementations for `()` and
/// tuples up to arity 5, which matches the largest argument bundle rustlantis
/// emits today after we prune unit values from its `dump_var` calls.
pub trait TraceDump {
    fn trace_dump(self);
}

impl TraceDump for () {
    #[inline]
    fn trace_dump(self) {}
}

impl<A: TraceValue> TraceDump for (A,) {
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
    }
}

impl<A: TraceValue, B: TraceValue> TraceDump for (A, B) {
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
        self.1.trace_write();
    }
}

impl<A: TraceValue, B: TraceValue, C: TraceValue> TraceDump for (A, B, C) {
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
        self.1.trace_write();
        self.2.trace_write();
    }
}

impl<A: TraceValue, B: TraceValue, C: TraceValue, D: TraceValue> TraceDump for (A, B, C, D) {
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
        self.1.trace_write();
        self.2.trace_write();
        self.3.trace_write();
    }
}

impl<A: TraceValue, B: TraceValue, C: TraceValue, D: TraceValue, E: TraceValue> TraceDump
    for (A, B, C, D, E)
{
    #[inline]
    fn trace_dump(self) {
        self.0.trace_write();
        self.1.trace_write();
        self.2.trace_write();
        self.3.trace_write();
        self.4.trace_write();
    }
}

/// Fold a value into the trace.
///
/// Generic over any `TraceDump`, so a single call site handles every supported
/// argument shape. Generated programs typically materialize a tuple local with
/// the values to dump, then pass it here.
#[inline]
pub fn dump_var<T: TraceDump>(value: T) {
    value.trace_dump();
}
