// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::ptx_asm;

fn main() {
    let a: u32;
    let b: u32;
    let c: u32;
    let d: u32;
    let e: u32;
    let f: u32;
    let g: u32;
    let h: u32;
    let i: u32;
    let j: u32;
    let k: u32;
    let l: u32;
    let m: u32;
    let n: u32;
    let o: u32;
    let p: u32;
    let q: u32;

    unsafe {
        ptx_asm!(
            "nop;", out("=r") a, out("=r") b, out("=r") c, out("=r") d, out("=r") e, out("=r") f, out("=r") g, out("=r") h, out("=r") i, out("=r") j, out("=r") k, out("=r") l, out("=r") m, out("=r") n, out("=r") o, out("=r") p, out("=r") q,
        );
    }
}
