/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for assigning through a 2-level `Index -> Field` place, e.g.
//! `arr[i].field = value` where `arr` is a local array of structs.
//!
//! The statement translator's 2-level assignment-projection match enumerated
//! `(Deref, Field)`, `(Field, Field)`, `(Deref, Index)`, `(Index, Index)`, and
//! `(Field, Index)`, but had no arm for `(Index, Field)` / `(ConstantIndex,
//! Field)`. Writing `arr[i].field = v` therefore failed to lower:
//!
//! ```text
//! 2-level projection Index(_) -> Field(_, _) not yet implemented for assignment
//! ```
//!
//! The fix adds the missing arm, delegating to `store_through_place_address` —
//! the same address-walk-and-store helper the sibling index arms and the 3+
//! projection fallback already use.
//!
//! The kernel exercises both the runtime `Index -> Field` form (a `for i in
//! 0..N` loop writing `arr[i].a`/`arr[i].b`) and the `ConstantIndex -> Field`
//! form (`arr[0].a`), then reduces the array so the host can verify the writes
//! actually landed.
//!
//! Usage:
//!   cargo oxide run index_field_assign

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[derive(Copy, Clone)]
    struct Cell {
        a: u64,
        b: u64,
    }

    /// Each thread fills a local `[Cell; 4]` by assigning to `arr[i].a` /
    /// `arr[i].b` (the `Index -> Field` place the fix enables), bumps `arr[0].a`
    /// (a `ConstantIndex -> Field` write), then writes the array's reduction to
    /// `out[tid]`.
    #[kernel]
    pub fn fill_and_sum(mut out: DisjointSlice<u64>, n: u32, fill: u32) {
        let tid = thread::index_1d().get();
        if tid >= n as usize {
            return;
        }

        let mut arr = [Cell { a: 0, b: 0 }; 4];
        // Runtime `Index -> Field` assignments. `fill` is a kernel argument, so
        // the optimizer cannot unroll the loop into constant indices — this keeps
        // `arr[i]` a genuine runtime `Index` projection (the form that blocked the
        // seeding/chain kernels), distinct from the `ConstantIndex` write below.
        let m = (fill as usize).min(4);
        for i in 0..m {
            // Arrays use built-in indexing: `arr[k]` lowers to a MIR
            // bounds-check `Assert` terminator plus a direct `Index` projection
            // on the place (`[T; N]` indexing never desugars to an
            // `IndexMut::index_mut` call). `k` is a runtime value, so these
            // stores exercise the genuine 2-level `(Index, Field)` shape. With
            // `m == 4`, `i & 3 == i`.
            let k = i & 3;
            arr[k].a = (tid as u64).wrapping_add(i as u64);
            arr[k].b = (tid as u64).wrapping_mul(i as u64 + 1);
        }
        // `ConstantIndex -> Field` assignment.
        arr[0].a = arr[0].a.wrapping_add(100);

        let mut s = 0u64;
        // Indexed reads are deliberate: `arr[j].a` exercises the 2-level
        // `Index -> Field` read projection, mirroring the writes above.
        #[allow(clippy::needless_range_loop)]
        for j in 0..4usize {
            s = s.wrapping_add(arr[j].a).wrapping_add(arr[j].b);
        }

        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = s;
        }
    }
}

fn main() {
    println!("=== index_field_assign ===");
    const N: usize = 256;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its writes.
    // `fill = 4` runs the runtime loop over all four cells (kept opaque to the
    // optimizer so `arr[i]` stays a runtime Index projection).
    unsafe { module.fill_and_sum(&stream, cfg, &mut out, N as u32, 4) }
        .expect("fill_and_sum launch");
    let got = out.to_host_vec(&stream).unwrap();

    // Per thread: sum a[j] = (tid+0+100)+(tid+1)+(tid+2)+(tid+3) = 4*tid + 106;
    //             sum b[j] = tid*(1+2+3+4) = 10*tid;  total = 14*tid + 106.
    for tid in 0..N as u64 {
        let want = 14u64.wrapping_mul(tid).wrapping_add(106);
        assert_eq!(got[tid as usize], want, "thread {tid}");
    }

    println!("PASS: index_field_assign (Index->Field + ConstantIndex->Field writes)");
}
