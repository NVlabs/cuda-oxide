/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline PTX support.
//!
//! User code should use [`ptx_asm!`](crate::ptx_asm), not the hidden functions
//! in this module. The hidden functions are compiler markers: the MIR importer
//! recognizes calls to them and replaces those calls with inline PTX.

macro_rules! define_ptx_asm_out {
    ($name:ident; $($arg:ident : $ty:ident),*) => {
        #[doc(hidden)]
        #[inline(never)]
        #[allow(unused_variables)]
        #[allow(clippy::too_many_arguments)]
        /// # Safety
        ///
        /// Compiler marker for `ptx_asm!`; user code must not call this directly.
        pub unsafe fn $name<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            $($ty,)*
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            $($arg: $ty,)*
        ) -> T {
            unreachable!("ptx_asm marker called outside CUDA kernel context")
        }
    };
}

macro_rules! define_ptx_asm_void {
    ($name:ident; $($arg:ident : $ty:ident),*) => {
        #[doc(hidden)]
        #[inline(never)]
        #[allow(unused_variables)]
        #[allow(clippy::too_many_arguments)]
        /// # Safety
        ///
        /// Compiler marker for `ptx_asm!`; user code must not call this directly.
        pub unsafe fn $name<
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            $($ty,)*
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            $($arg: $ty,)*
        ) {
            unreachable!("ptx_asm marker called outside CUDA kernel context")
        }
    };
}

// Rust has no variadic generics, so expose marker stubs for fixed arities.
define_ptx_asm_out!(__ptx_asm_out_0;);
define_ptx_asm_out!(__ptx_asm_out_1; a0: A0);
define_ptx_asm_out!(__ptx_asm_out_2; a0: A0, a1: A1);
define_ptx_asm_out!(__ptx_asm_out_3; a0: A0, a1: A1, a2: A2);
define_ptx_asm_out!(__ptx_asm_out_4; a0: A0, a1: A1, a2: A2, a3: A3);
define_ptx_asm_out!(__ptx_asm_out_5; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4);
define_ptx_asm_out!(__ptx_asm_out_6; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5);
define_ptx_asm_out!(__ptx_asm_out_7; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6);
define_ptx_asm_out!(__ptx_asm_out_8; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7);
define_ptx_asm_out!(__ptx_asm_out_9; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8);
define_ptx_asm_out!(__ptx_asm_out_10; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9);
define_ptx_asm_out!(__ptx_asm_out_11; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10);
define_ptx_asm_out!(__ptx_asm_out_12; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11);
define_ptx_asm_out!(__ptx_asm_out_13; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12);
define_ptx_asm_out!(__ptx_asm_out_14; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13);
define_ptx_asm_out!(__ptx_asm_out_15; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14);
define_ptx_asm_out!(__ptx_asm_out_16; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15);

define_ptx_asm_void!(__ptx_asm_void_0;);
define_ptx_asm_void!(__ptx_asm_void_1; a0: A0);
define_ptx_asm_void!(__ptx_asm_void_2; a0: A0, a1: A1);
define_ptx_asm_void!(__ptx_asm_void_3; a0: A0, a1: A1, a2: A2);
define_ptx_asm_void!(__ptx_asm_void_4; a0: A0, a1: A1, a2: A2, a3: A3);
define_ptx_asm_void!(__ptx_asm_void_5; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4);
define_ptx_asm_void!(__ptx_asm_void_6; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5);
define_ptx_asm_void!(__ptx_asm_void_7; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6);
define_ptx_asm_void!(__ptx_asm_void_8; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7);
define_ptx_asm_void!(__ptx_asm_void_9; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8);
define_ptx_asm_void!(__ptx_asm_void_10; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9);
define_ptx_asm_void!(__ptx_asm_void_11; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10);
define_ptx_asm_void!(__ptx_asm_void_12; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11);
define_ptx_asm_void!(__ptx_asm_void_13; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12);
define_ptx_asm_void!(__ptx_asm_void_14; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13);
define_ptx_asm_void!(__ptx_asm_void_15; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14);
define_ptx_asm_void!(__ptx_asm_void_16; a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11, a12: A12, a13: A13, a14: A14, a15: A15);
