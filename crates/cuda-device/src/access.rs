/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Choosing a tile shape from a target memory transaction width.
//!
//! Transaction width is set by the alignment of the element type, so it is a
//! consequence of the data layout rather than something a kernel can request.
//! That makes the useful question the inverted one: *given* the width I want,
//! what does the tile shape have to be?
//!
//! The `gemm_sol` example already reasons in that direction, by hand, in a
//! comment:
//!
//! ```text
//! Each K-group: 128 rows x 8 elements x 2 bytes = 2048 bytes
//! ```
//!
//! Read as a derivation: a 128-bit transaction of f16 is 8 elements, 128 rows
//! then move 1024 elements per group. Run the same arithmetic against a `K = 32`
//! tile and it forces `M >= 1024 / 32 = 32` - which is how CUTLASS states the
//! constraint for a row-major A operand at 128 threads and 8-element access.
//! Every step is arithmetic on the width, the element size, and the launch
//! shape - which is what this module computes, so it does not have to be
//! re-derived in a comment per kernel.
//!
//! Everything here is `const`, so a plan can drive a `const` generic or sit in a
//! `const {}` assertion and turn a tile shape that cannot reach the intended
//! width into a compile error.
//!
//! # Scope
//!
//! This answers "what shape does this width need". It does not decide whether
//! the width is *achievable*, which depends on the element type and, for loads,
//! on how the value is used.
//!
//! Measured on sm_86, with the element type 16-byte aligned throughout:
//!
//! ```text
//! how the element is used                     LDG          STG
//! never decomposed                            LDG.E.128    STG.E.128
//! lanes read, whole value assembled+stored    LDG.E x4     STG.E.128
//! lanes read, scalar stored                   LDG.E x4     STG.E
//! ```
//!
//! So `align` is a necessary condition, and for loads not a sufficient one:
//! touching individual lanes lets the optimiser split the local apart and load
//! the lanes separately whatever the alignment. Stores have no such condition.
//!
//! That limit is specific to a *direct blocked* load, where the wide value
//! itself feeds the per-element arithmetic. CUB's `BLOCK_LOAD_VECTORIZE`
//! reaches the width for such kernels anyway, by splitting the layouts: load
//! *striped* with wide loads (where nothing decomposes the value), then
//! exchange through shared memory into the blocked arrangement the arithmetic
//! wants. Decomposition still happens, but on registers that never came from
//! a wide load. cuda-oxide has no striped/blocked exchange primitive yet, so
//! a decomposing kernel currently cannot reach the width - the gap is the
//! missing layout change, not anything about alignment.
//!
//! A plan therefore describes the shape a width *needs*, not a width the kernel
//! is guaranteed to get - the data has to move as whole elements as well.
//! See the `vectorization` example for the alignment half.

/// One 32-bit transaction (`LDG.E` / `STG.E`).
pub const TXN_32: usize = 4;
/// One 64-bit transaction (`LDG.E.64` / `STG.E.64`).
pub const TXN_64: usize = 8;
/// One 128-bit transaction (`LDG.E.128` / `STG.E.128`), the widest available.
pub const TXN_128: usize = 16;

/// What a target transaction width implies for a tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AccessPlan {
    /// Bytes moved per thread per instruction.
    pub txn_bytes: usize,
    /// Element type alignment required to reach `txn_bytes`.
    ///
    /// Necessary, and for loads not sufficient - see the module's *Scope*
    /// section.
    pub align: usize,
    /// Elements each thread moves per instruction.
    pub elems_per_thread: usize,
    /// Threads participating.
    pub threads: usize,
    /// Elements the whole block moves per instruction.
    pub elems_per_block: usize,
}

impl AccessPlan {
    /// Smallest extent of one tile axis, given the contiguous extent of the
    /// other.
    ///
    /// This models a strided operand, the CUTLASS shape rule: `other` is the
    /// contiguous extent (the row length of a row-major tile), and rows are
    /// not assumed adjacent in memory, so no access may cross a row boundary.
    /// Two divisibilities follow: each thread's access must fit a whole
    /// number of times in a row (`elems_per_thread` divides `other`), and a
    /// block-wide access must cover whole rows (`other` divides
    /// `elems_per_block`). Returns `None` when either fails, because rounding
    /// here would silently change the tile.
    ///
    /// This is the CUTLASS derivation: with 1024 elements per block-wide access
    /// and `K = 32`, the minimum `M` is 32.
    #[must_use]
    pub const fn min_extent(&self, other: usize) -> Option<usize> {
        if other == 0
            || !other.is_multiple_of(self.elems_per_thread)
            || !self.elems_per_block.is_multiple_of(other)
        {
            return None;
        }
        Some(self.elems_per_block / other)
    }

    /// Whether a tile of `tile_elems` elements can be moved by whole block-wide
    /// accesses at this width.
    #[must_use]
    pub const fn fits_tile(&self, tile_elems: usize) -> bool {
        self.elems_per_block != 0
            && tile_elems != 0
            && tile_elems.is_multiple_of(self.elems_per_block)
    }

    /// Block-wide accesses needed to move a tile of `tile_elems`.
    #[must_use]
    pub const fn passes_for_tile(&self, tile_elems: usize) -> Option<usize> {
        if !self.fits_tile(tile_elems) {
            return None;
        }
        Some(tile_elems / self.elems_per_block)
    }
}

/// Plan a tile for a target transaction width.
///
/// Returns `None` when the width is not one the hardware offers, when it is not
/// a whole number of elements, or when `threads` is zero.
///
/// # Example
///
/// ```rust
/// #![feature(f16)]
/// use cuda_device::access::{self, AccessPlan};
///
/// // 128-bit transactions of f16, across 128 threads.
/// const PLAN: AccessPlan = match access::plan::<f16>(access::TXN_128, 128) {
///     Some(plan) => plan,
///     None => panic!("128 bits is a whole number of f16"),
/// };
/// // 8 elements per thread, 1024 per block, so K = 32 forces M >= 32.
/// const MIN_M: usize = match PLAN.min_extent(32) {
///     Some(extent) => extent,
///     None => panic!("32 divides 1024"),
/// };
/// assert_eq!((PLAN.elems_per_thread, PLAN.elems_per_block, MIN_M), (8, 1024, 32));
/// ```
#[must_use]
pub const fn plan<T>(txn_bytes: usize, threads: usize) -> Option<AccessPlan> {
    plan_for_elem_size(core::mem::size_of::<T>(), txn_bytes, threads)
}

/// [`plan`] with the element size supplied directly.
///
/// Useful when the element type is not nameable in the calling context, such as
/// a packed pair whose Rust type lives in another crate.
#[must_use]
pub const fn plan_for_elem_size(
    elem_size: usize,
    txn_bytes: usize,
    threads: usize,
) -> Option<AccessPlan> {
    if threads == 0 || elem_size == 0 {
        return None;
    }
    // Only these three widths exist; a "96-bit" request is a mistake, not
    // something to round down silently.
    if txn_bytes != TXN_32 && txn_bytes != TXN_64 && txn_bytes != TXN_128 {
        return None;
    }
    // A transaction has to be a whole number of elements. An f32 cannot do a
    // 6-byte access, and an f64 cannot do a 32-bit one.
    if !txn_bytes.is_multiple_of(elem_size) {
        return None;
    }
    let elems_per_thread = txn_bytes / elem_size;
    let Some(elems_per_block) = threads.checked_mul(elems_per_thread) else {
        return None;
    };
    Some(AccessPlan {
        txn_bytes,
        align: txn_bytes,
        elems_per_thread,
        threads,
        elems_per_block,
    })
}

/// The widest transaction a tile of `tile_elems` can be moved by, across
/// `threads` threads, without a partial access.
///
/// The auditing direction, kept for comparing an existing kernel against the
/// width it could have had.
#[must_use]
pub const fn widest<T>(tile_elems: usize, threads: usize) -> Option<AccessPlan> {
    let elem_size = core::mem::size_of::<T>();
    // Widest first, so the first whole fit wins.
    let candidates = [TXN_128, TXN_64, TXN_32];
    let mut i = 0;
    while i < candidates.len() {
        if let Some(p) = plan_for_elem_size(elem_size, candidates[i], threads)
            && p.fits_tile(tile_elems)
        {
            return Some(p);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduces the CUTLASS threadblock-shape derivation quoted in the module
    /// docs. If this drifts, the module's stated justification is wrong.
    #[test]
    fn reproduces_the_cutlass_threadblock_derivation() {
        // f16 is 2 bytes; a 128-bit transaction is therefore 8 elements.
        let p = plan_for_elem_size(2, TXN_128, 128).unwrap();
        assert_eq!(
            p.elems_per_thread, 8,
            "128-bit of f16 is an 8-element access"
        );
        assert_eq!(p.elems_per_block, 1024, "128 threads x 8 elements");
        assert_eq!(p.align, 16, "128-bit needs 16-byte alignment");
        // "ThreadblockShape M=32 is the minimum ... with 128 threads x
        // 8-element access", against K = 32.
        assert_eq!(p.min_extent(32), Some(32));
    }

    #[test]
    fn element_size_scales_elements_per_thread() {
        // Same width, three element sizes.
        assert_eq!(
            plan_for_elem_size(4, TXN_128, 32).unwrap().elems_per_thread,
            4
        );
        assert_eq!(
            plan_for_elem_size(2, TXN_128, 32).unwrap().elems_per_thread,
            8
        );
        assert_eq!(
            plan_for_elem_size(8, TXN_128, 32).unwrap().elems_per_thread,
            2
        );
        // f64 cannot do a 32-bit access: it is not a whole number of elements.
        assert_eq!(plan_for_elem_size(8, TXN_32, 32), None);
    }

    /// A width the hardware does not offer must be refused rather than rounded,
    /// otherwise the plan would describe an access that is never emitted.
    #[test]
    fn rejects_widths_that_do_not_exist() {
        for bad in [1usize, 2, 3, 6, 12, 24, 32, 0] {
            assert_eq!(
                plan_for_elem_size(4, bad, 32),
                None,
                "{bad}-byte transaction must be refused"
            );
        }
    }

    /// An extent that does not divide the block-wide access is refused, because
    /// rounding it would quietly change the tile the caller asked for.
    #[test]
    fn refuses_extents_that_do_not_divide() {
        let p = plan_for_elem_size(4, TXN_128, 128).unwrap(); // 512 elems/block
        assert_eq!(p.min_extent(32), Some(16));
        assert_eq!(p.min_extent(64), Some(8));
        assert_eq!(p.min_extent(48), None, "48 does not divide 512");
        assert_eq!(p.min_extent(0), None);
    }

    /// A row shorter than one thread's access is shape-impossible on a
    /// strided operand: the access would cross the row boundary. So `other`
    /// must be a multiple of `elems_per_thread`, not merely a divisor of
    /// `elems_per_block`.
    #[test]
    fn refuses_extents_shorter_than_one_thread_access() {
        // f16 at 128-bit across 128 threads: 8 elements per thread, 1024 per
        // block.
        let p = plan_for_elem_size(2, TXN_128, 128).unwrap();
        // 4 divides 1024, but a row of 4 f16 (8 bytes) cannot hold one
        // thread's 16-byte access.
        assert_eq!(p.min_extent(4), None, "4 f16 cannot hold a 128-bit access");
        // The smallest legal extent is exactly one thread access: 8 elements,
        // for 1024 / 8 = 128 rows.
        assert_eq!(p.min_extent(8), Some(128));
    }

    #[test]
    fn counts_passes_over_a_tile() {
        let p = plan_for_elem_size(4, TXN_128, 128).unwrap(); // 512 elems/block
        assert_eq!(p.passes_for_tile(512), Some(1));
        assert_eq!(p.passes_for_tile(2048), Some(4));
        assert_eq!(p.passes_for_tile(500), None, "partial access");
        assert!(p.fits_tile(1024) && !p.fits_tile(1023));
    }

    /// The auditing direction: the widest whole fit, not merely a legal one.
    #[test]
    fn widest_picks_the_largest_whole_fit() {
        // 32 threads of f32: 128-bit needs 128 elements per access.
        assert_eq!(widest::<f32>(128, 32).unwrap().txn_bytes, TXN_128);
        // 64 elements cannot take a 128-bit block access, but fits 64-bit.
        assert_eq!(widest::<f32>(64, 32).unwrap().txn_bytes, TXN_64);
        // 32 elements: one f32 each.
        assert_eq!(widest::<f32>(32, 32).unwrap().txn_bytes, TXN_32);
        // Nothing divides 33.
        assert_eq!(widest::<f32>(33, 32), None);
    }

    /// Usable in const position, which is the point: a tile shape that cannot
    /// reach the intended width should fail to compile, not to benchmark.
    #[test]
    fn plan_is_usable_at_compile_time() {
        const P: AccessPlan = match plan_for_elem_size(4, TXN_128, 256) {
            Some(p) => p,
            None => panic!("128-bit f32 across 256 threads is representable"),
        };
        const MIN_M: usize = match P.min_extent(64) {
            Some(m) => m,
            None => panic!("64 divides 1024"),
        };
        const _: () = assert!(MIN_M == 16);
        assert_eq!(P.elems_per_block, 1024);
        assert_eq!(MIN_M, 16);
    }
}
