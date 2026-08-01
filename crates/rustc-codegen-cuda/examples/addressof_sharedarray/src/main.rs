/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static shared-memory access through `llvm.addressof` (guards issue #54).
//!
//! The kernel does `OUTPUT_NORM[0] = OUTPUT_NORM[0] * weight` on a static
//! `SharedArray<f32, 1>`. Before the fix in PR #55, the llvm-export textual exporter
//! gave the `addressof @__shared_mem_N` result a `%vN` SSA name even though
//! `addressof` is virtual in textual LLVM IR (it has no instruction form,
//! only a symbol reference at use sites). When the use printed before the
//! addressof's block, the GEP referenced a `%vN` no instruction defined and
//! libNVVM rejected the IR.
//!
//! The same kernel verifies that exposing the address of the first static
//! shared allocation does not turn its valid shared-space offset zero into a
//! null Rust address. Named-space pointers must become CUDA generic pointers
//! before pointer-to-integer conversion, and the exposed integer must cast
//! back into the shared space through the generic space, so the
//! expose/recover round trip stores through the recovered pointer.
//!
//! It also constructs and matches a direct enum payload containing that shared
//! pointer. The enum's physical storage must stay 64-bit even when modern NVVM
//! uses a 32-bit representation for semantic shared-space pointers.
//!
//! This example launches the kernel through `cuda_host::ltoir::load_kernel_module`,
//! which compiles the cuda-oxide-emitted NVVM IR via libNVVM and links the
//! cubin via nvJitLink. A dangling SSA reference in the `.ll` would fail at
//! libNVVM's verifier before the kernel could run, so a regression of #54
//! is now a hard runtime failure instead of a silent build artifact.
//!
//! Run: `cargo oxide run addressof_sharedarray`

#![allow(static_mut_refs)]
#![allow(clippy::assign_op_pattern)] // Expanded assignment preserves the addressof repro CFG.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, device, kernel, thread};
use cuda_host::{cuda_module, ltoir};

enum SharedPointerPayload {
    Empty,
    Pointer(*mut SharedArray<f32, 1>),
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    #[device]
    fn shared_pointer_enum_address(use_pointer: bool, pointer: *mut SharedArray<f32, 1>) -> usize {
        let payload = if use_pointer {
            SharedPointerPayload::Pointer(pointer)
        } else {
            SharedPointerPayload::Empty
        };

        match payload {
            SharedPointerPayload::Empty => 0,
            SharedPointerPayload::Pointer(extracted) => extracted.addr(),
        }
    }

    #[kernel]
    pub fn sharedarray_late_use(seed: f32, mut out: DisjointSlice<f32>) {
        static mut OUTPUT_NORM: SharedArray<f32, 1> = SharedArray::UNINIT;

        if thread::index_1d().get() == 0 {
            unsafe {
                OUTPUT_NORM[0] = seed;
                let weight = repro_weight();
                // Issue #54 repro shape: load addressof[0], multiply, store.
                OUTPUT_NORM[0] = OUTPUT_NORM[0] * weight;
                *out.get_unchecked_mut(0) = OUTPUT_NORM[0];

                // The first static shared allocation has local shared offset
                // zero, but its CUDA generic address must not be null.
                let raw = &raw mut OUTPUT_NORM;
                let raw_address = raw.addr();
                *out.get_unchecked_mut(1) = if raw.is_null() || raw_address == 0 {
                    0.0
                } else {
                    1.0
                };

                // A direct enum payload keeps the pointer's shared-space
                // semantics while using target-stable generic physical
                // storage. The runtime condition keeps construction,
                // discriminant inspection, and extraction observable.
                let enum_address = shared_pointer_enum_address(seed != 0.0, raw);
                *out.get_unchecked_mut(2) = if enum_address == raw_address {
                    1.0
                } else {
                    0.0
                };

                // The exposed integer is a generic address, so casting it
                // back into the shared space must recover the original
                // pointer (inttoptr to generic, then cvta back to shared).
                // Write through the recovered pointer and observe the store
                // through the original allocation.
                // The recovered pointer must equal the original as a value,
                // not merely reach the same memory: hardware masks the
                // shared-window base out of st.shared addresses, so a wrong
                // pointer value can still store to the right slot.
                let round_trip = recover_shared_pointer(raw_address);
                (&mut (*round_trip))[0] = OUTPUT_NORM[0] + 1.0;
                *out.get_unchecked_mut(3) = if core::ptr::eq(round_trip, raw)
                    && OUTPUT_NORM[0] == seed * repro_weight() + 1.0
                {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }

    #[inline(never)]
    #[device]
    fn repro_weight() -> f32 {
        3.0
    }

    /// Recover a shared pointer from its exposed integer address.
    ///
    /// `#[inline(never)]` keeps the `inttoptr` in a function that cannot
    /// see the matching `ptrtoint`, so LLVM's InferAddressSpaces cannot
    /// fold the pair away and the lowering's own re-entry rule is what
    /// executes.
    #[inline(never)]
    #[device]
    fn recover_shared_pointer(address: usize) -> *mut SharedArray<f32, 1> {
        address as *mut SharedArray<f32, 1>
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== addressof_sharedarray (issue #54 regression) ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Forces the cuda-oxide-emitted `.ll` through libNVVM + nvJitLink.
    // A dangling SSA reference in the IR would fail libNVVM's verifier here.
    let raw_module = ltoir::load_kernel_module(&ctx, "addressof_sharedarray")?;
    let module = kernels::from_module(raw_module).expect("typed module init failed");

    let cfg = LaunchConfig::for_num_elems(1);
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 4)?;
    let seed: f32 = 7.0;

    // SAFETY: one thread is launched, matching the kernel's three-element output
    // access and fixed shared-memory use.
    unsafe { module.sharedarray_late_use(stream.as_ref(), cfg, seed, &mut out) }?;

    let result = out.to_host_vec(&stream)?;
    let expected: f32 = 21.0; // seed * repro_weight() == 7.0 * 3.0

    if (result[0] - expected).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: got {}, expected {expected}",
            result[0]
        );
        std::process::exit(1);
    }
    if (result[1] - 1.0).abs() >= f32::EPSILON {
        eprintln!("FAIL addressof_sharedarray: shared offset zero exposed as null");
        std::process::exit(1);
    }
    if (result[2] - 1.0).abs() >= f32::EPSILON {
        eprintln!("FAIL addressof_sharedarray: shared pointer enum did not round-trip");
        std::process::exit(1);
    }
    if (result[3] - 1.0).abs() >= f32::EPSILON {
        eprintln!(
            "FAIL addressof_sharedarray: integer address did not cast back to the shared pointer"
        );
        std::process::exit(1);
    }
    println!(
        "PASS addressof_sharedarray: seed={seed}, result={}, shared address is non-null, shared pointer enum round-tripped, integer address cast back to the shared pointer",
        result[0]
    );
    Ok(())
}
