// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, unused_variables)]

use cuda_macros::ptx_asm;

mod cuda_device {
    pub mod ptx {
        pub unsafe fn __ptx_asm_out_0<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
        ) -> T {
            panic!("test marker")
        }

        pub unsafe fn __ptx_asm_out_1<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            A0,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            _a0: A0,
        ) -> T {
            panic!("test marker")
        }

        pub unsafe fn __ptx_asm_out_2<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            A0,
            A1,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            _a0: A0,
            _a1: A1,
        ) -> T {
            panic!("test marker")
        }

        pub unsafe fn __ptx_asm_void_0<
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
        ) {
        }
    }
}

fn accepts_cuda_doc_shape() {
    let x = 1u32;
    let z = 2u32;
    let y: u32;
    let reg_only: u32;
    let lane: u32;

    unsafe {
        ptx_asm!(
            "add.u32 %0, %1, %2;",
            out("=r") y,
            in("r") x,
            in("r") z,
        );
        ptx_asm!(
            "mul.lo.u32 %0, %1, %1;",
            out("=r") reg_only,
            in("r") x,
            options(register_only),
        );
        ptx_asm!("mov.u32 %0, %%laneid;", out("=r") lane);
        ptx_asm!("membar.gl;", clobber("memory"));
    }

    let _ = (y, reg_only, lane);
}

fn main() {}
