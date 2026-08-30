/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
//! Nonzero all-lane oracle for all eight ordered sparse floating MMA forms.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, wmma};
const META: u32 = 0x4444_4444;
const fn val(x: u32, y: u32) -> u32 {
    1 + ((x + y) & 1)
}
const fn h(x: u32) -> u32 {
    if x == 1 { 0x3c00 } else { 0x4000 }
}
const fn b(x: u32) -> u32 {
    if x == 1 { 0x3f80 } else { 0x4000 }
}
const fn t(x: u32) -> u32 {
    if x == 1 { 0x3f80_0000 } else { 0x4000_0000 }
}
const fn pk(x: u32, y: u32) -> u32 {
    x | (y << 16)
}

#[cuda_module]
mod kernels {
    use super::*;
    fn aa(l: u32, n: u32, bf: u32) -> u32 {
        let r = l / 4 + 8 * (n & 1);
        let q = (l & 3) * 2 + 8 * (n / 2);
        let x = val(r, q);
        let y = val(r, q + 1);
        if bf == 0 {
            pk(h(x), h(y))
        } else {
            pk(b(x), b(y))
        }
    }
    fn bb(l: u32, n: u32, bf: u32) -> u32 {
        let c = l / 4;
        let k = (l & 3) * 2 + 8 * n;
        let x = val(k, c);
        let y = val(k + 1, c);
        if bf == 0 {
            pk(h(x), h(y))
        } else {
            pk(b(x), b(y))
        }
    }
    fn at(l: u32, n: u32) -> u32 {
        let r = l / 4 + 8 * (n & 1);
        let q = (l & 3) + 4 * (n / 2);
        t(val(r, q))
    }
    fn bt(l: u32, n: u32) -> u32 {
        t(val((l & 3) + 4 * n, l / 4))
    }
    fn cp(l: u32, n: u32) -> u32 {
        let r = l / 4 + 8 * n;
        let c = (l & 3) * 2;
        pk(h(val(r, c)), h(val(r, c + 1)))
    }
    fn cf(l: u32) -> [f32; 4] {
        let r = l / 4;
        let c = (l & 3) * 2;
        [
            val(r, c) as f32,
            val(r, c + 1) as f32,
            val(r + 8, c) as f32,
            val(r + 8, c + 1) as f32,
        ]
    }
    #[kernel]
    pub fn oracle(mut po: DisjointSlice<u32>, mut fo: DisjointSlice<f32>) {
        let l = thread::threadIdx_x();
        let c = [cp(l, 0), cp(l, 1)];
        let f = cf(l);
        let (h16, h32, x16, x32, y16, y32, z8, z16) = unsafe {
            (
                wmma::mma_sp_ordered_metadata_m16n8k16_f16_f16(
                    c,
                    [aa(l, 0, 0), aa(l, 1, 0)],
                    [bb(l, 0, 0), bb(l, 1, 0)],
                    META,
                    0,
                ),
                wmma::mma_sp_ordered_metadata_m16n8k32_f16_f16(
                    c,
                    [aa(l, 0, 0), aa(l, 1, 0), aa(l, 2, 0), aa(l, 3, 0)],
                    [bb(l, 0, 0), bb(l, 1, 0), bb(l, 2, 0), bb(l, 3, 0)],
                    META,
                    0,
                ),
                wmma::mma_sp_ordered_metadata_m16n8k16_f32_f16(
                    f,
                    [aa(l, 0, 0), aa(l, 1, 0)],
                    [bb(l, 0, 0), bb(l, 1, 0)],
                    META,
                    0,
                ),
                wmma::mma_sp_ordered_metadata_m16n8k32_f32_f16(
                    f,
                    [aa(l, 0, 0), aa(l, 1, 0), aa(l, 2, 0), aa(l, 3, 0)],
                    [bb(l, 0, 0), bb(l, 1, 0), bb(l, 2, 0), bb(l, 3, 0)],
                    META,
                    0,
                ),
                wmma::mma_sp_ordered_metadata_m16n8k16_f32_bf16(
                    f,
                    [aa(l, 0, 1), aa(l, 1, 1)],
                    [bb(l, 0, 1), bb(l, 1, 1)],
                    META,
                    0,
                ),
                wmma::mma_sp_ordered_metadata_m16n8k32_f32_bf16(
                    f,
                    [aa(l, 0, 1), aa(l, 1, 1), aa(l, 2, 1), aa(l, 3, 1)],
                    [bb(l, 0, 1), bb(l, 1, 1), bb(l, 2, 1), bb(l, 3, 1)],
                    META,
                    0,
                ),
                wmma::mma_sp_ordered_metadata_m16n8k8_f32_tf32(
                    f,
                    [at(l, 0), at(l, 1)],
                    [bt(l, 0), bt(l, 1)],
                    META,
                    0,
                ),
                wmma::mma_sp_ordered_metadata_m16n8k16_f32_tf32(
                    f,
                    [at(l, 0), at(l, 1), at(l, 2), at(l, 3)],
                    [bt(l, 0), bt(l, 1), bt(l, 2), bt(l, 3)],
                    META,
                    0,
                ),
            )
        };
        let p = l as usize * 4;
        for (i, v) in h16.into_iter().chain(h32).enumerate() {
            unsafe { *po.get_unchecked_mut(p + i) = v }
        }
        let p = l as usize * 24;
        for (i, v) in x16
            .into_iter()
            .chain(x32)
            .chain(y16)
            .chain(y32)
            .chain(z8)
            .chain(z16)
            .enumerate()
        {
            unsafe { *fo.get_unchecked_mut(p + i) = v }
        }
    }
}
fn rc(l: usize, r: usize) -> (usize, usize) {
    (l / 4 + 8 * (r / 2), (l % 4) * 2 + r % 2)
}
fn reference(r: usize, c: usize, kdim: usize) -> u32 {
    let mut s = val(r as u32, c as u32);
    for k in 0..kdim {
        if k % 4 < 2 {
            let q = (k / 4) * 2 + k % 4;
            s += val(r as u32, q as u32) * val(k as u32, c as u32)
        }
    }
    s
}
fn reference_tf32(r: usize, c: usize, kdim: usize) -> u32 {
    let mut s = val(r as u32, c as u32);
    for k in 0..kdim {
        if k % 2 == 0 {
            s += val(r as u32, (k / 2) as u32) * val(k as u32, c as u32)
        }
    }
    s
}
fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let (major, minor) = ctx.compute_capability().unwrap();
    if major < 8 {
        println!("skip sm_{major}{minor}");
        return;
    }
    let s = ctx.default_stream();
    let m = kernels::load(&ctx).expect("module");
    let mut pd = DeviceBuffer::<u32>::zeroed(&s, 128).unwrap();
    let mut fd = DeviceBuffer::<f32>::zeroed(&s, 768).unwrap();
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { m.oracle(&s, cfg, &mut pd, &mut fd) }.unwrap();
    let p = pd.to_host_vec(&s).unwrap();
    let f = fd.to_host_vec(&s).unwrap();
    let mut bad = 0;
    for l in 0..32 {
        for v in 0..2 {
            for r in 0..4 {
                let w = p[l * 4 + v * 2 + r / 2];
                let got = if r % 2 == 0 { w & 0xffff } else { w >> 16 };
                let (x, y) = rc(l, r);
                let want =
                    half::f16::from_f32(reference(x, y, [16, 32][v]) as f32).to_bits() as u32;
                if got != want {
                    eprintln!("f16 variant {v} lane {l} reg {r}: {got:#x} != {want:#x}");
                    bad += 1
                }
            }
        }
        for v in 0..6 {
            for r in 0..4 {
                let got = f[l * 24 + v * 4 + r];
                let (x, y) = rc(l, r);
                let want = if v < 4 {
                    reference(x, y, [16, 32, 16, 32][v])
                } else {
                    reference_tf32(x, y, [8, 16][v - 4])
                } as f32;
                if got != want {
                    eprintln!("f32 variant {v} lane {l} reg {r}: {got} != {want}");
                    bad += 1
                }
            }
        }
    }
    assert_eq!(bad, 0, "accumulator mismatches");
    println!(
        "SUCCESS: all 8 variants; all 32 lanes and 4 logical accumulators/lane match host GEMM"
    )
}
