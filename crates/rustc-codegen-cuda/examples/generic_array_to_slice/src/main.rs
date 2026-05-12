//! Regression test for `&Container -> &[T]` slice construction via
//! `unsafe { slice::from_raw_parts(self as *const _ as *const T, N) }`
//! on a struct whose Rust-level layout has multiple fields but whose
//! runtime memory layout is a contiguous `[T; N]`. Real-world surface:
//! `generic_array::GenericArray<T, N>`, used pervasively across the
//! `RustCrypto` / `elliptic-curve` / `k256` stack.
//!
//! ## Pre-fix wall
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Verification failed for '<…as_bytes…>':
//!        Compilation error: verification failed.
//!        MirInsertFieldOp on slice field 0: pointee mir.struct
//!          <Container, …> does not match slice element
//!          builtin.integer ui8
//!        Failed operation:
//!          op….. = mir.insert_field (slice_undef, ptr_to_struct) [0]
//! ```
//!
//! ## What triggers it
//!
//! A struct whose Rust-level layout is "a recursive tower of nested
//! sub-structs and `PhantomData` markers" (typenum-encoded) but whose
//! runtime memory layout is identical to `[T; N]`. The struct's
//! `Deref::deref` impl (or any `as_slice`-style helper) does:
//!
//! ```rust,ignore
//! impl Container {
//!     fn as_slice(&self) -> &[u8] {
//!         unsafe {
//!             core::slice::from_raw_parts(
//!                 self as *const Self as *const u8,
//!                 SIZE,
//!             )
//!         }
//!     }
//! }
//! ```
//!
//! The MIR for that body is, roughly:
//!
//! ```text
//! _2 = Cast(PtrToPtr, _1: *const Container, *const u8)
//! _3 = aggregate (RawPtr [u8]) (_2, SIZE)
//! ```
//!
//! cuda-oxide's `PtrToPtr` cast lowering does not actually retype the
//! pointer's pointee in the dialect-mir IR — both source and dest
//! stay `MirPtrType<MirStructType<Container, …>>`. The downstream
//! slice aggregate then tries `mir.insert_field [0]` of a
//! `*Container` into a `MirSliceType<u8>`, and the dialect verifier
//! correctly rejects the type mismatch.
//!
//! ## What landed
//!
//! Synthesize a `MirCastOp::PtrToPtr` in the
//! `Aggregate(RawPtr(slice<T>))` handler when the data-pointer
//! operand's pointee type doesn't already match the slice element
//! type. The cast retypes the pointer to `*T` (preserving mutability
//! and address space) before it's `insert_field`'d into the slice
//! fat pointer.
//!
//! This is exactly what an opaque-pointer LLVM backend (rust-cuda's
//! path) does implicitly — `insertvalue { ptr, i64 }` doesn't carry
//! pointee type info, so the source pointer's prior pointee is
//! erased silently. Our typed dialect (`MirPtrType<T>`) preserves
//! the pointee, so we have to issue the retyping cast explicitly to
//! satisfy the slice's `MirSliceType<T>` element constraint. The
//! cast lowers to a downstream `bitcast ptr to ptr` in opaque-pointer
//! LLVM IR (effectively a no-op but type-correct in our dialect).
//!
//! Why this gap exists: rustc's MIR optimizer elides no-op pointer
//! casts (`self as *const Self as *const T`) when both source and
//! dest are `*const _` shapes, then feeds the un-retyped source
//! pointer straight into `Aggregate(RawPtr(slice<T>))`. The
//! optimizer is correct under opaque-pointer semantics; we observe
//! the elision because we look at MIR after this optimization runs
//! and we maintain typed pointers.
//!
//! Note on the broader implication: this fix also helps
//! `~/vanity-miner-rs/`'s `k256` chain past the
//! `to_encoded_point().as_bytes()` wall, since
//! `generic_array::Deref::deref` uses exactly the
//! `from_raw_parts(self as _, N)` pattern this test exercises. There
//! may be additional walls deeper in the `k256` /
//! `elliptic-curve` / `generic_array` type tree (typenum-encoded
//! `UInt`/`UTerm`/`B0`/`B1` types appear in many code paths), each
//! resolved one at a time.
//!
//! Originally surfaced from `~/vanity-miner-rs/`:
//! `logic::secp256k1_derive_public_key` calling
//! `public_key.to_encoded_point(true).as_bytes()` in
//! `<EncodedPoint<UInt<UInt<UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>,
//! B0>, B0>, B0>> as ToEncodedPoint<Secp256k1>>::as_bytes`.
//!
//! ## Build with
//!
//!     cargo oxide run generic_array_to_slice
//!
//! Expected: kernel runs and emits the XOR-fold of each input
//! 32-byte window.
//!
//! ## What this example is NOT
//!
//! - Not about `Iterator::Item` / `Mul::Output` /
//!   `GenericSequence::Sequence` aliases (those are alias-arm
//!   territory and have separate reproducers).
//! - Not about `Drop` glue (separate reproducer).
//! - Not about safe `as_slice` paths (`[T; N]::as_slice` works fine
//!   because the source pointer is already typed as `*[T; N]`).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Minimal stand-in for `generic_array::GenericArray<u8, U33>`: a
/// struct whose Rust-level layout is "two halves + a marker" but
/// whose runtime memory layout is a contiguous `[u8; 32]`. The
/// `as_slice` impl below uses the same `from_raw_parts(self as
/// *const _ as *const u8, N)` pattern `GenericArray::Deref::deref`
/// uses internally.
#[repr(C)]
pub struct Container {
    pub head: Half,
    pub tail: Half,
    pub _marker: Marker,
}

#[repr(C)]
pub struct Half {
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub d: u8,
    pub e: u8,
    pub f: u8,
    pub g: u8,
    pub h: u8,
    pub i: u8,
    pub j: u8,
    pub k: u8,
    pub l: u8,
    pub m: u8,
    pub n: u8,
    pub o: u8,
    pub p: u8,
}

pub struct Marker;

impl Container {
    /// Trigger: pointer-cast + `slice::from_raw_parts`. The MIR has
    /// `Cast(PtrToPtr, *Container, *u8)` followed by a slice
    /// aggregate. cuda-oxide's `PtrToPtr` lowering doesn't retype
    /// the pointer's pointee, so the slice aggregate's data-pointer
    /// field gets a `*Container` where it expects `*u8`, and the
    /// dialect verifier rejects the mismatch.
    #[inline(never)]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(self as *const Self as *const u8, 32)
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Fill a `Container` from input bytes, then call `as_slice` and
    /// XOR-fold its bytes into the output slot.
    #[kernel]
    pub fn container_xor(input: &[u8], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i + 32 <= input.len()
        {
            // Build a Container from 32 input bytes.
            let head = Half {
                a: input[i + 0], b: input[i + 1], c: input[i + 2], d: input[i + 3],
                e: input[i + 4], f: input[i + 5], g: input[i + 6], h: input[i + 7],
                i: input[i + 8], j: input[i + 9], k: input[i + 10], l: input[i + 11],
                m: input[i + 12], n: input[i + 13], o: input[i + 14], p: input[i + 15],
            };
            let tail = Half {
                a: input[i + 16], b: input[i + 17], c: input[i + 18], d: input[i + 19],
                e: input[i + 20], f: input[i + 21], g: input[i + 22], h: input[i + 23],
                i: input[i + 24], j: input[i + 25], k: input[i + 26], l: input[i + 27],
                m: input[i + 28], n: input[i + 29], o: input[i + 30], p: input[i + 31],
            };
            let container = Container { head, tail, _marker: Marker };

            // The trigger.
            let bytes = container.as_slice();
            let mut acc = 0u8;
            for &b in bytes {
                acc ^= b;
            }
            *slot = acc;
        }
    }
}

fn main() {
    println!("=== generic_array_to_slice ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u8> = (0..(N + 32) as u32).map(|n| (n & 0xff) as u8).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .container_xor(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    println!("output: {:?}", result);
    println!("SUCCESS: Container::as_slice codegen'd to PTX");
}
