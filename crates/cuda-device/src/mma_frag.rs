/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fragment index algebra for the `m16n8k16` tensor-core accumulator.
//!
//! [`crate::wmma::mma_m16n8k16_f32_f16`] takes and returns `[f32; 4]`, and the
//! four values are *not* four adjacent elements of the output tile. Each lane
//! holds a scattered quarter of a 16x8 tile, and the required mapping is stated
//! in the intrinsic's docs as index arithmetic the caller has to reproduce
//! exactly. Getting it wrong does not fail: it computes a different, wrong
//! matrix product. This module derives the mapping once so call sites do not
//! have to.
//!
//! # The layout is a mixed-radix numeral
//!
//! For the `m16n8k16` f32 accumulator, PTX assigns lane `l` the elements
//!
//! ```text
//!     group = l / 4          tig = l % 4
//!     c[0], c[1]  ->  row = group,      col = 2*tig + {0, 1}
//!     c[2], c[3]  ->  row = group + 8,  col = 2*tig + {0, 1}
//! ```
//!
//! Substituting into the row-major offset `row * 8 + col` and grouping terms
//! shows this is a positional numeral:
//!
//! ```text
//!     offset = (group + 8*(j>>1)) * 8 + 2*tig + (j&1)
//!            = (j&1)*1 + tig*2 + group*8 + (j>>1)*64
//!
//!     digit    range   place value   source
//!     j & 1      2         1         fragment index, low bit
//!     tig        4         2         l % 4
//!     group      8         8         l / 4
//!     j >> 1     2        64         fragment index, high bit
//! ```
//!
//! Each place value is the product of the ranges of all lower digits: `1`, then
//! `1*2 = 2`, then `2*4 = 8`, then `8*8 = 64`, with total span `64*2 = 128`.
//! That is the condition for a mixed-radix system to be uniquely decodable, so
//! the map `(l, j) -> offset` is injective; since the domain has `32 * 4 = 128`
//! points and the tile has `16 * 8 = 128` elements, it is a bijection. The
//! warp's fragments therefore tile the output exactly, with no overlap and no
//! gap.
//!
//! This is the same argument that underpins [`crate::thread::DisjointBlock`] and
//! [`crate::thread::DisjointTiling`]; the only difference is where the digits
//! come from. What makes this instance awkward to write by hand is that the
//! fragment index is *split across two non-adjacent digit positions*, at place
//! values 1 and 64, interleaved with the two lane digits.

/// Rows in the `m16n8k16` accumulator tile.
pub const ACC_ROWS: usize = 16;
/// Columns in the `m16n8k16` accumulator tile.
pub const ACC_COLS: usize = 8;
/// Accumulator registers each lane holds.
pub const ACC_PER_LANE: usize = 4;
/// Lanes participating in one `mma.sync`.
pub const WARP_LANES: usize = 32;

/// Row and column in the accumulator tile for one fragment register.
///
/// `lane` must be in `0..32` and `j` in `0..4`; both are reduced modulo their
/// range so an out-of-range argument stays inside the tile rather than producing
/// an offset past its end.
#[must_use]
#[inline(always)]
pub const fn acc_coords(lane: usize, j: usize) -> (usize, usize) {
    let lane = lane % WARP_LANES;
    let j = j % ACC_PER_LANE;
    let group = lane / 4;
    let tig = lane % 4;
    (group + 8 * (j >> 1), 2 * tig + (j & 1))
}

/// Row-major offset within the accumulator tile for one fragment register.
///
/// Equivalent to `row * ACC_COLS + col` from [`acc_coords`], written in the
/// numeral form derived in the module docs.
#[must_use]
#[inline(always)]
pub const fn acc_offset(lane: usize, j: usize) -> usize {
    let lane = lane % WARP_LANES;
    let j = j % ACC_PER_LANE;
    let group = lane / 4;
    let tig = lane % 4;
    // (j&1)*1 + tig*2 + group*8 + (j>>1)*64
    (j & 1) + tig * 2 + group * 8 + (j >> 1) * 64
}

/// Offset of a tile element within a larger row-major matrix.
///
/// `tile_row` and `tile_col` are the tile's origin in the matrix, and
/// `row_stride` the matrix's row length. This is the form GEMM epilogues need:
/// the accumulator tile is written into a strided destination rather than a
/// standalone 16x8 buffer.
///
/// Returns `None` on overflow so a bad stride cannot wrap into a plausible
/// offset.
#[must_use]
#[inline(always)]
pub fn acc_matrix_offset(
    lane: usize,
    j: usize,
    tile_row: usize,
    tile_col: usize,
    row_stride: usize,
) -> Option<usize> {
    let (row, col) = acc_coords(lane, j);
    row.checked_add(tile_row)?
        .checked_mul(row_stride)?
        .checked_add(col.checked_add(tile_col)?)
}

/// Recover `(lane, j)` from a tile offset.
///
/// The inverse of [`acc_offset`], and the reason the mapping is sound: it exists
/// precisely because the place values form the running product of the digit
/// ranges. Returns `None` for an offset outside the tile.
///
/// This is exported mainly so a caller can assert the round-trip in their own
/// tests, which is a cheap way to catch a hand-rolled layout that disagrees.
#[must_use]
#[inline(always)]
pub const fn acc_decode(offset: usize) -> Option<(usize, usize)> {
    if offset >= ACC_ROWS * ACC_COLS {
        return None;
    }
    // Read the digits back out, lowest place value first.
    let j_lo = offset % 2;
    let tig = (offset / 2) % 4;
    let group = (offset / 8) % 8;
    let j_hi = offset / 64;
    Some((group * 4 + tig, (j_hi << 1) | j_lo))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim the module rests on: `(lane, j) -> offset` is a bijection onto
    /// the tile. Overlap would mean two lanes writing one element; a gap would
    /// mean an element never written.
    #[test]
    fn lane_fragment_map_is_a_bijection_onto_the_tile() {
        let mut owner = [None::<(usize, usize)>; ACC_ROWS * ACC_COLS];
        for lane in 0..WARP_LANES {
            for j in 0..ACC_PER_LANE {
                let offset = acc_offset(lane, j);
                assert!(
                    offset < owner.len(),
                    "offset {offset} outside tile for lane {lane}, j {j}"
                );
                assert_eq!(
                    owner[offset], None,
                    "element {offset} claimed by both {:?} and (lane {lane}, j {j})",
                    owner[offset]
                );
                owner[offset] = Some((lane, j));
            }
        }
        assert!(
            owner.iter().all(Option::is_some),
            "the warp's fragments must cover every element of the tile"
        );
    }

    /// `acc_offset` must agree with the row/col form, which is how the PTX docs
    /// state the layout. A disagreement means the numeral regrouping is wrong.
    #[test]
    fn numeral_form_agrees_with_row_column_form() {
        for lane in 0..WARP_LANES {
            for j in 0..ACC_PER_LANE {
                let (row, col) = acc_coords(lane, j);
                assert!(row < ACC_ROWS && col < ACC_COLS, "coords escape the tile");
                assert_eq!(
                    acc_offset(lane, j),
                    row * ACC_COLS + col,
                    "numeral and row-major forms disagree at lane {lane}, j {j}"
                );
            }
        }
    }

    /// Decoding is the inverse of encoding. This is the property that makes the
    /// place values correct: a mixed-radix numeral is decodable exactly when
    /// each place value is the running product of the lower ranges.
    #[test]
    fn decode_inverts_offset() {
        for lane in 0..WARP_LANES {
            for j in 0..ACC_PER_LANE {
                assert_eq!(
                    acc_decode(acc_offset(lane, j)),
                    Some((lane, j)),
                    "round-trip failed for lane {lane}, j {j}"
                );
            }
        }
        assert_eq!(acc_decode(ACC_ROWS * ACC_COLS), None);
    }

    /// The two halves of the fragment index sit at place values 1 and 64, so
    /// `j` 0 and 1 differ by one column while `j` 0 and 2 differ by eight rows.
    /// Pinned because this is the part most easily written wrong by hand.
    #[test]
    fn fragment_index_splits_across_two_digit_positions() {
        for lane in 0..WARP_LANES {
            let (r0, c0) = acc_coords(lane, 0);
            let (r1, c1) = acc_coords(lane, 1);
            let (r2, c2) = acc_coords(lane, 2);
            // Low bit of j walks the column.
            assert_eq!((r1, c1), (r0, c0 + 1));
            // High bit of j jumps eight rows.
            assert_eq!((r2, c2), (r0 + 8, c0));
            assert_eq!(acc_offset(lane, 2) - acc_offset(lane, 0), 64);
        }
    }

    /// Placing the tile in a strided matrix must preserve disjointness, since
    /// that is how a GEMM epilogue actually writes it.
    #[test]
    fn matrix_placement_stays_disjoint() {
        let row_stride = 128;
        let (tile_row, tile_col) = (32, 16);
        let mut seen = [false; ACC_ROWS * 128];
        for lane in 0..WARP_LANES {
            for j in 0..ACC_PER_LANE {
                let off = acc_matrix_offset(lane, j, tile_row, tile_col, row_stride).unwrap();
                let local = off - tile_row * row_stride - tile_col;
                assert!(!seen[local], "collision in strided placement at {off}");
                seen[local] = true;
            }
        }
        assert_eq!(
            seen.iter().filter(|&&s| s).count(),
            WARP_LANES * ACC_PER_LANE
        );
    }

    #[test]
    fn matrix_offset_rejects_overflow() {
        assert_eq!(acc_matrix_offset(0, 0, usize::MAX, 0, 2), None);
        assert_eq!(acc_matrix_offset(0, 0, 1, usize::MAX, usize::MAX), None);
    }
}
