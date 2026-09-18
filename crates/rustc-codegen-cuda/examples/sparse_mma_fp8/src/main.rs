/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */

//! GPU oracle for the four plain (standard-metadata) SM89 sparse FP8 MMA forms:
//!
//! ```text
//! mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32
//! mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e5m2.f32
//! mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32
//! mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e5m2.f32
//! ```
//!
//! # The fragment contract
//!
//! A is 2:4 sparse along K, so it is held **compressed** as 16x32 in four `.b32`
//! registers; B is the full 64x8 in four `.b32` registers. For lane `l`, with
//! `g = l/4`, `t = l%4`, register `n` and byte position `i` (byte 0 is the
//! lowest byte of the register):
//!
//! ```text
//! A: row = g + 8*(n%2)                     compressed col = 4*t + 16*(n/2) + i
//! B: k   = 16*n + 4*t + i                  col            = g
//! D: regs {0,1} -> row g,   cols 2*t, 2*t+1
//!    regs {2,3} -> row g+8, cols 2*t, 2*t+1
//! metadata: nibble = i0 | i1<<2, the same code in all eight 4-bit groups;
//!           compressed column 2*q takes dense K 4*q + i0,
//!           compressed column 2*q+1 takes dense K 4*q + i1
//! selector = 0
//! ```
//!
//! So the four FP8 values in a register are little-endian: byte `i` is the
//! lowest-addressed element of the four it covers.
//!
//! # How that was determined
//!
//! Neither the byte order nor the nibble-to-K-group mapping is documented
//! anywhere this checkout can reach, so the hardware is the oracle. `--probe`
//! runs the e4m3/e4m3 form once per candidate wiring -- 13,824 candidates over
//! six independent axes: which A registers hold row `g+8`, which compressed
//! columns an A register holds, the A byte order, which K rows a B register
//! holds, the B byte order, and which of a nibble's two 2-bit fields names the
//! even compressed column. Every candidate is filled into the registers and
//! executed; the host compares against an exact integer GEMM that does not
//! depend on the candidate at all, so a wrong axis changes the sum.
//!
//! 32 candidates reproduce the reference bit-for-bit, and they are exactly two
//! families:
//!
//! * `A row = n%2, A band = 16*(n/2), A byte order = i, B band = 16*n,
//!    B byte order = i` for all twelve legal nibble codes -- 24 entries once
//!    the field-swap duplicate is removed;
//! * the same with **both** byte orders reversed and the nibble's fields
//!    swapped, for two codes only (`0x3`/`0xc` and `0x6`/`0x9`).
//!
//! The first family is the wiring: it is the only one that reproduces the
//! reference for the other ten nibble codes, and the byte-order axis is varied
//! on its own (reversing A alone or B alone matches nothing). The second
//! family is a self-compensating alias that exists only for the two nibbles
//! whose code has the larger index in the low field; the oracle therefore
//! exercises `0x4`, `0x8` and `0xe` as well, which no byte-order reversal can
//! reproduce.
//!
//! Re-run the sweep with:
//!
//! ```text
//! cargo run -p cargo-oxide -- run sparse_mma_fp8 -- --probe
//! ```

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, wmma};

const M: usize = 16;
const N: usize = 8;
const K: usize = 64;
/// Compressed K: two kept elements per 4-wide group.
const KC: usize = K / 2;

/// Exactly-checkable operands: every value is an integer in 1..=8, which both
/// e4m3 and e5m2 represent exactly, and A/B are positionally distinct so any
/// mis-route changes the sum instead of hiding behind a tolerance.
fn a_val(r: u32, c: u32) -> u32 {
    1 + (3 * r + 5 * c) % 7
}
fn b_val(k: u32, n: u32) -> u32 {
    1 + (2 * k + 3 * n) % 8
}
fn c_val(r: u32, n: u32) -> u32 {
    (5 * r + 7 * n) % 32
}

/// e4m3 / e5m2 encoding of the small non-negative integers used here.
///
/// The mantissa field is `(v - 2^e)` scaled by `2^(mantissa_bits - e)`: 5 is
/// `1.25 * 4`, so its e4m3 mantissa field is `2`, not `1`.
fn f8_bits(v: u32, e4m3: bool) -> u32 {
    if v == 0 {
        return 0;
    }
    let mut e = 0u32;
    if v >= 2 {
        e = 1;
    }
    if v >= 4 {
        e = 2;
    }
    if v >= 8 {
        e = 3;
    }
    let mant = v - (1u32 << e);
    if e4m3 {
        let sh = if e >= 3 { 0 } else { 3 - e };
        ((e + 7) << 3) | (mant << sh)
    } else {
        let sh = if e >= 2 { 0 } else { 2 - e };
        ((e + 15) << 2) | (mant << sh)
    }
}

/// The six legal "two distinct positions" metadata codes. `0x0`, `0x5`, `0xa`
/// and `0xf` are undefined behaviour because their two 2-bit fields are equal.
fn nibble_of(i: u32) -> u32 {
    if i == 0 {
        0x4
    } else if i == 1 {
        0x6
    } else if i == 2 {
        0x8
    } else if i == 3 {
        0x9
    } else if i == 4 {
        0xc
    } else {
        0xe
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// A fragment register `n`, byte `i`, lane `l`.
    fn a_slot(l: u32, n: u32, i: u32) -> (u32, u32) {
        (l / 4 + 8 * (n & 1), 4 * (l & 3) + 16 * (n / 2) + i)
    }

    /// B fragment register `n`, byte `i`, lane `l`.
    fn b_slot(l: u32, n: u32, i: u32) -> (u32, u32) {
        (16 * n + 4 * (l & 3) + i, l / 4)
    }

    /// This lane's four packed A registers: the compressed 16x32 matrix.
    fn a_regs(l: u32, e4m3: bool) -> [u32; 4] {
        let mut a = [0u32; 4];
        let mut n = 0;
        while n < 4 {
            let mut i = 0;
            while i < 4 {
                let (r, c) = a_slot(l, n, i);
                a[n as usize] |= f8_bits(a_val(r, c), e4m3) << (8 * i);
                i += 1;
            }
            n += 1;
        }
        a
    }

    /// This lane's four packed B registers: the full 64x8 matrix.
    fn b_regs(l: u32, e4m3: bool) -> [u32; 4] {
        let mut b = [0u32; 4];
        let mut n = 0;
        while n < 4 {
            let mut i = 0;
            while i < 4 {
                let (k, col) = b_slot(l, n, i);
                b[n as usize] |= f8_bits(b_val(k, col), e4m3) << (8 * i);
                i += 1;
            }
            n += 1;
        }
        b
    }

    /// This lane's four f32 accumulators, one per (row, column) pair.
    fn c_regs(l: u32) -> [f32; 4] {
        let r = l / 4;
        let c = (l & 3) * 2;
        [
            c_val(r, c) as f32,
            c_val(r, c + 1) as f32,
            c_val(r + 8, c) as f32,
            c_val(r + 8, c + 1) as f32,
        ]
    }

    fn store(out: &mut DisjointSlice<f32>, variant: usize, l: u32, d: [f32; 4]) {
        let p = variant * 128 + (l as usize) * 4;
        let mut j = 0usize;
        while j < 4 {
            unsafe { *out.get_unchecked_mut(p + j) = d[j] };
            j += 1;
        }
    }

    /// Variant `v` occupies `out[v*128 + lane*4 .. +4]`, with
    /// `v = nonzero_accumulator*24 + form*6 + metadata_code`.
    #[kernel]
    pub fn oracle(mut out: DisjointSlice<f32>) {
        let l = thread::threadIdx_x();
        let a4 = a_regs(l, true);
        let a5 = a_regs(l, false);
        let b4 = b_regs(l, true);
        let b5 = b_regs(l, false);
        let cz = [0.0f32; 4];
        let cn = c_regs(l);

        let mut base = 0usize;
        let mut acc = 0;
        while acc < 2 {
            let c = if acc == 0 { cz } else { cn };
            let mut ni = 0;
            while ni < 6 {
                let meta = nibble_of(ni) * 0x1111_1111;
                let d = unsafe { wmma::mma_sp_m16n8k64_f32_e4m3_e4m3_f32(c, a4, b4, meta, 0) };
                store(&mut out, base + ni as usize, l, d);
                let d = unsafe { wmma::mma_sp_m16n8k64_f32_e4m3_e5m2_f32(c, a4, b5, meta, 0) };
                store(&mut out, base + 6 + ni as usize, l, d);
                let d = unsafe { wmma::mma_sp_m16n8k64_f32_e5m2_e4m3_f32(c, a5, b4, meta, 0) };
                store(&mut out, base + 12 + ni as usize, l, d);
                let d = unsafe { wmma::mma_sp_m16n8k64_f32_e5m2_e5m2_f32(c, a5, b5, meta, 0) };
                store(&mut out, base + 18 + ni as usize, l, d);
                ni += 1;
            }
            base += 24;
            acc += 1;
        }
    }

    /// Layout sweep: runs the e4m3/e4m3 form once per candidate fragment
    /// layout so the host can check every candidate against an exact GEMM.
    #[kernel]
    pub fn probe(mut out: DisjointSlice<f32>) {
        let l = thread::threadIdx_x();
        let g = l / 4;
        let t = l % 4;
        let zero = [0.0f32; 4];

        let mut combo = 0usize;
        let mut ai = 0u32;
        while ai < A_LAYOUTS {
            let rp = ai / (A_BAND_OPTS * A_ORD_OPTS);
            let ab = (ai / A_ORD_OPTS) % A_BAND_OPTS;
            let ao = ai % A_ORD_OPTS;
            let mut a = [0u32; 4];
            let mut n = 0u32;
            while n < 4 {
                let r = g + 8 * a_row_off(rp, n);
                let mut i = 0u32;
                while i < 4 {
                    let c = 4 * t + a_band(ab, n) + byte_ord(ao, i);
                    a[n as usize] |= f8_bits(probe_a(r, c), true) << (8 * i);
                    i += 1;
                }
                n += 1;
            }

            let mut bi = 0u32;
            while bi < B_LAYOUTS {
                let bb = bi / B_ORD_OPTS;
                let bo = bi % B_ORD_OPTS;
                let mut b = [0u32; 4];
                let mut n = 0u32;
                while n < 4 {
                    let mut i = 0u32;
                    while i < 4 {
                        let k = b_band(bb, n) + 4 * t + byte_ord(bo, i);
                        b[n as usize] |= f8_bits(probe_b(k, g), true) << (8 * i);
                        i += 1;
                    }
                    n += 1;
                }

                let mut mi = 0u32;
                while mi < META_OPTS {
                    let nib = probe_nibble(mi % 12);
                    // `sw` exchanges the two 2-bit fields of the nibble.
                    let word = if mi / 12 == 0 {
                        nib
                    } else {
                        ((nib & 3) << 2) | (nib >> 2)
                    };
                    let meta = word * 0x1111_1111;
                    let d = unsafe { wmma::mma_sp_m16n8k64_f32_e4m3_e4m3_f32(zero, a, b, meta, 0) };
                    let p = combo * 128 + (l as usize) * 4;
                    let mut j = 0usize;
                    while j < 4 {
                        unsafe { *out.get_unchecked_mut(p + j) = d[j] };
                        j += 1;
                    }
                    combo += 1;
                    mi += 1;
                }
                bi += 1;
            }
            ai += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Host reference for the oracle.
// ---------------------------------------------------------------------------

/// D = A(compressed) @ B under the 2:4 pattern the metadata code selects, plus
/// the accumulator when `acc` is set. `nibble = i0 | i1 << 2` sends compressed
/// column `2g` to dense K `4g + i0` and column `2g + 1` to `4g + i1`.
fn expect(r: usize, col: usize, nibble: u32, acc: bool) -> f32 {
    let (r, col) = (r as u32, col as u32);
    let (i0, i1) = (nibble & 3, nibble >> 2);
    let mut s = if acc { c_val(r, col) } else { 0 };
    for g in 0..(K as u32) / 4 {
        s += a_val(r, 2 * g) * b_val(4 * g + i0, col);
        s += a_val(r, 2 * g + 1) * b_val(4 * g + i1, col);
    }
    s as f32
}

fn run_oracle(ctx: &std::sync::Arc<CudaContext>) {
    let s = ctx.default_stream();
    let module = kernels::load(ctx).expect("module");
    let mut out = DeviceBuffer::<f32>::zeroed(&s, 48 * 128).unwrap();
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.oracle(&s, cfg, &mut out) }.unwrap();
    let buf = out.to_host_vec(&s).unwrap();

    let names = ["e4m3 x e4m3", "e4m3 x e5m2", "e5m2 x e4m3", "e5m2 x e5m2"];
    let mut bad = 0usize;
    for (form, name) in names.iter().enumerate() {
        for acc in 0..2 {
            let mut form_bad = 0usize;
            for ni in 0..6u32 {
                let variant = acc * 24 + form * 6 + ni as usize;
                let nibble = nibble_of(ni);
                for l in 0..32usize {
                    for j in 0..4usize {
                        let r = l / 4 + 8 * (j / 2);
                        let col = (l % 4) * 2 + j % 2;
                        let want = expect(r, col, nibble, acc == 1);
                        let got = buf[variant * 128 + l * 4 + j];
                        if got.to_bits() != want.to_bits() {
                            eprintln!(
                                "{name} C{} nibble {nibble:#x} lane {l} reg {j} \
                                 (row {r}, col {col}): {got} != {want}",
                                if acc == 1 { "!=0" } else { "=0" }
                            );
                            form_bad += 1;
                        }
                    }
                }
            }
            println!(
                "  {name:12} {:>7} metadata codes 0x4 0x6 0x8 0x9 0xc 0xe: {} mismatches",
                if acc == 1 { "C!=0," } else { "C=0," },
                form_bad
            );
            bad += form_bad;
        }
    }
    assert_eq!(bad, 0, "sparse FP8 MMA accumulator mismatches");
    println!(
        "SUCCESS: all 4 sparse FP8 m16n8k64 forms x 6 metadata codes x C=0/C!=0; \
         all 32 lanes and 4 logical accumulators/lane match host GEMM exactly"
    );
}

// ---------------------------------------------------------------------------
// Layout sweep (`--probe`).
//
// The candidate space varies the axes independently:
//
//   A row   `a_row_off(rp, n)`  -- which A registers hold row group+8
//   A band  `a_band(ab, n)`     -- which compressed columns a register holds
//   A order `byte_ord(ao, i)`   -- byte position i -> column offset
//   B band  `b_band(bb, n)`     -- which dense K rows a register holds
//   B order `byte_ord(bo, i)`   -- byte position i -> K offset
//   code    the nibble and which of its two 2-bit fields names the even column
//
// Every candidate is filled into the registers and executed once; the host
// compares the result with an exact GEMM whose answer does not depend on the
// candidate at all. A wrong axis changes the sum, so only the true wiring
// reproduces it.
// ---------------------------------------------------------------------------

const A_ROW_OPTS: u32 = 4;
const A_BAND_OPTS: u32 = 6;
const A_ORD_OPTS: u32 = 2;
const B_BAND_OPTS: u32 = 6;
const B_ORD_OPTS: u32 = 2;
const META_SWAP_OPTS: u32 = 2;

const A_LAYOUTS: u32 = A_ROW_OPTS * A_BAND_OPTS * A_ORD_OPTS;
const B_LAYOUTS: u32 = B_BAND_OPTS * B_ORD_OPTS;
const META_OPTS: u32 = 12 * META_SWAP_OPTS;
const COMBOS: u32 = A_LAYOUTS * B_LAYOUTS * META_OPTS;

/// Distinct from the oracle's operands so no single axis can hide behind a
/// period of the value function: `c/16` makes the A band visible and `k/16`
/// makes the B band visible.
fn probe_a(r: u32, c: u32) -> u32 {
    1 + (3 * r + 5 * c) % 7 + c / 16
}
fn probe_b(k: u32, n: u32) -> u32 {
    1 + (2 * k + 3 * n + 4 * (k / 16)) % 8
}

fn a_row_off(rp: u32, n: u32) -> u32 {
    if rp == 0 {
        n & 1
    } else if rp == 1 {
        n / 2
    } else if rp == 2 {
        (n & 1) ^ 1
    } else {
        (n / 2) ^ 1
    }
}

fn a_band(ab: u32, n: u32) -> u32 {
    if ab == 0 {
        16 * (n / 2)
    } else if ab == 1 {
        4 * (n / 2)
    } else if ab == 2 {
        8 * n
    } else if ab == 3 {
        4 * n
    } else if ab == 4 {
        8 * (n / 2)
    } else {
        16 * n
    }
}

fn b_band(bb: u32, n: u32) -> u32 {
    if bb == 0 {
        16 * n
    } else if bb == 1 {
        8 * n
    } else if bb == 2 {
        4 * n
    } else if bb == 3 {
        16 * (2 * (n & 1) + n / 2)
    } else if bb == 4 {
        16 * (n / 2)
    } else {
        16 * (n & 1)
    }
}

fn byte_ord(o: u32, i: u32) -> u32 {
    if o == 0 { i } else { 3 - i }
}

/// The twelve legal nibbles: two 2-bit fields that differ.
fn probe_nibble(idx: u32) -> u32 {
    if idx == 0 {
        0x1
    } else if idx == 1 {
        0x2
    } else if idx == 2 {
        0x3
    } else if idx == 3 {
        0x4
    } else if idx == 4 {
        0x6
    } else if idx == 5 {
        0x7
    } else if idx == 6 {
        0x8
    } else if idx == 7 {
        0x9
    } else if idx == 8 {
        0xb
    } else if idx == 9 {
        0xc
    } else if idx == 10 {
        0xd
    } else {
        0xe
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    rp: u32,
    ab: u32,
    ao: u32,
    bb: u32,
    bo: u32,
    nib: u32,
    sw: u32,
}

impl Candidate {
    fn a_slot(&self, l: u32, n: u32, i: u32) -> (i64, i64) {
        (
            (l / 4 + 8 * a_row_off(self.rp, n)) as i64,
            (4 * (l % 4) + a_band(self.ab, n) + byte_ord(self.ao, i)) as i64,
        )
    }
    fn b_slot(&self, l: u32, n: u32, i: u32) -> (i64, i64) {
        (
            (b_band(self.bb, n) + 4 * (l % 4) + byte_ord(self.bo, i)) as i64,
            (l / 4) as i64,
        )
    }

    /// (A entries placed, B entries placed) out of 512 each.
    fn coverage(&self) -> (u32, u32) {
        let mut ap = 0;
        let mut bp = 0;
        for l in 0..32u32 {
            for n in 0..4u32 {
                for i in 0..4u32 {
                    let (r, c) = self.a_slot(l, n, i);
                    if (0..M as i64).contains(&r) && (0..KC as i64).contains(&c) {
                        ap += 1;
                    }
                    let (k, col) = self.b_slot(l, n, i);
                    if (0..K as i64).contains(&k) && (0..N as i64).contains(&col) {
                        bp += 1;
                    }
                }
            }
        }
        (ap, bp)
    }

    /// Dense K position that compressed column `c` feeds under this code.
    fn kmap(&self, c: u32) -> u32 {
        let low = self.nib & 3;
        let high = self.nib >> 2;
        let field = if (c % 2 == 0) == (self.sw == 0) {
            low
        } else {
            high
        };
        4 * (c / 2) + field
    }

    fn predict(&self) -> Vec<f32> {
        let mut a_seen = vec![vec![0.0f32; KC]; M];
        let mut b_seen = vec![vec![0.0f32; N]; K];
        for l in 0..32u32 {
            for n in 0..4u32 {
                for i in 0..4u32 {
                    let (r, c) = self.a_slot(l, n, i);
                    if (0..M as i64).contains(&r) && (0..KC as i64).contains(&c) {
                        a_seen[r as usize][c as usize] = probe_a(r as u32, c as u32) as f32;
                    }
                    let (k, col) = self.b_slot(l, n, i);
                    if (0..K as i64).contains(&k) && (0..N as i64).contains(&col) {
                        b_seen[k as usize][col as usize] = probe_b(k as u32, col as u32) as f32;
                    }
                }
            }
        }
        let mut d = vec![0.0f32; M * N];
        for r in 0..M {
            for n in 0..N {
                let mut s = 0.0f32;
                for c in 0..KC {
                    s += a_seen[r][c] * b_seen[self.kmap(c as u32) as usize][n];
                }
                d[r * N + n] = s;
            }
        }
        d
    }
}

/// Decode one lane's four accumulators into D[row][col].
fn observed(combo: usize, buf: &[f32]) -> Vec<f32> {
    let mut d = vec![0.0f32; M * N];
    for l in 0..32usize {
        for j in 0..4usize {
            let r = l / 4 + 8 * (j / 2);
            let c = (l % 4) * 2 + j % 2;
            d[r * N + c] = buf[combo * 128 + l * 4 + j];
        }
    }
    d
}

fn run_probe(ctx: &std::sync::Arc<CudaContext>) {
    let s = ctx.default_stream();
    let module = kernels::load(ctx).expect("module");
    let mut out = DeviceBuffer::<f32>::zeroed(&s, COMBOS as usize * 128).unwrap();
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.probe(&s, cfg, &mut out) }.unwrap();
    let buf = out.to_host_vec(&s).unwrap();

    let mut exact = Vec::new();
    let mut total = 0usize;
    for ai in 0..A_LAYOUTS {
        for bi in 0..B_LAYOUTS {
            for mi in 0..META_OPTS {
                let cand = Candidate {
                    rp: ai / (A_BAND_OPTS * A_ORD_OPTS),
                    ab: (ai / A_ORD_OPTS) % A_BAND_OPTS,
                    ao: ai % A_ORD_OPTS,
                    bb: bi / B_ORD_OPTS,
                    bo: bi % B_ORD_OPTS,
                    nib: probe_nibble(mi % 12),
                    sw: mi / 12,
                };
                let idx = total;
                total += 1;
                let want = cand.predict();
                let got = observed(idx, &buf);
                if want
                    .iter()
                    .zip(got.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                {
                    exact.push(cand);
                }
            }
        }
    }
    println!("probe: {total} candidates, {} exact", exact.len());
    for c in &exact {
        let (ap, bp) = c.coverage();
        println!(
            "  exact rp={} ab={} ao={} bb={} bo={} nib={:#x} sw={} (A {ap}/512, B {bp}/512 placed)",
            c.rp, c.ab, c.ao, c.bb, c.bo, c.nib, c.sw
        );
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let (major, minor) = ctx.compute_capability().unwrap();
    if major < 8 || (major == 8 && minor < 9) {
        println!("skipping: sparse FP8 MMA requires sm_89+, found sm_{major}{minor}");
        return;
    }
    if std::env::args().any(|a| a == "--probe") {
        run_probe(&ctx);
    } else {
        run_oracle(&ctx);
    }
}
