/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end oracle for the generated `ldmatrix.m8n8.x4.b16` intrinsic.
//!
//! One warp fills four distinct 8x8 b16 matrices in shared memory. Every lane
//! then supplies one 16-byte row address to `ldmatrix`; the host checks all 128
//! returned register fragments against their exact source matrix, row, and
//! column pair.
//!
//! Build and run with:
//!   cargo oxide run generated_ldmatrix

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use cuda_intrinsics::matrix::ldmatrix_m8n8_x4_b16;
use cuda_intrinsics::sreg::thread_idx_x;

const LANES: usize = 32;
const MATRICES: usize = 4;
const ROWS: usize = 8;
const COLUMNS: usize = 8;
const WORDS_PER_ROW: usize = COLUMNS / 2;
const SHARED_WORDS: usize = MATRICES * ROWS * WORDS_PER_ROW;
const OUTPUT_WORDS: usize = LANES * MATRICES;

/// A distinct nonzero value for every matrix element.
///
/// ```text
/// bit 15 = marker; bits 11:10 = matrix; bits 7:5 = row; bits 2:0 = column
/// ```
#[inline(always)]
const fn matrix_element(matrix: usize, row: usize, column: usize) -> u16 {
    0x8000 | ((matrix as u16) << 10) | ((row as u16) << 5) | column as u16
}

/// Pack adjacent b16 columns exactly as `ldmatrix` returns them: the lower
/// column in bits 0..15 and the upper column in bits 16..31.
#[inline(always)]
const fn matrix_word(matrix: usize, row: usize, pair: usize) -> u32 {
    let lower = matrix_element(matrix, row, pair * 2) as u32;
    let upper = matrix_element(matrix, row, pair * 2 + 1) as u32;
    lower | (upper << 16)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn ldmatrix_x4_oracle(mut output: DisjointSlice<u32>) {
        // Four 8x8 b16 matrices occupy 512 bytes. Each row is four u32 words
        // (16 bytes), and the explicit alignment makes every row a legal
        // `ldmatrix.sync.aligned.m8n8.x4` source.
        static mut INPUT: SharedArray<u32, SHARED_WORDS, 16> = SharedArray::UNINIT;

        let lane = thread_idx_x() as usize;
        if lane >= LANES {
            return;
        }

        // Lanes 0..7 initialize matrix 0, lanes 8..15 matrix 1, and so on.
        // Each lane owns one whole row and therefore performs no conflicting
        // writes before the block-wide barrier.
        let source_matrix = lane / ROWS;
        let source_row = lane % ROWS;
        let row_word = lane * WORDS_PER_ROW;
        let shared = core::ptr::addr_of_mut!(INPUT) as *mut u32;
        unsafe {
            shared
                .add(row_word)
                .write(matrix_word(source_matrix, source_row, 0));
            shared
                .add(row_word + 1)
                .write(matrix_word(source_matrix, source_row, 1));
            shared
                .add(row_word + 2)
                .write(matrix_word(source_matrix, source_row, 2));
            shared
                .add(row_word + 3)
                .write(matrix_word(source_matrix, source_row, 3));
        }

        // `ldmatrix` is a weak shared-memory read; this barrier makes every
        // lane's row initialization visible before the warp reads the tile.
        thread::sync_threads();

        // SAFETY:
        // - the launch has exactly one full 32-lane warp and no lane exits;
        // - all lanes execute this same generated intrinsic unconditionally;
        // - lane `n` supplies row `n % 8` of matrix `n / 8`, as x4 requires;
        // - each address points to a live, initialized, 16-byte-aligned row;
        // - the barrier above orders the shared-memory writes before the load.
        let registers = unsafe { ldmatrix_m8n8_x4_b16(shared.add(row_word).cast_const()) };

        // Result register m contains matrix m, row lane/4, column pair lane%4.
        // Each lane owns four unique output slots.
        let output_word = lane * MATRICES;
        unsafe {
            *output.get_unchecked_mut(output_word) = registers[0];
            *output.get_unchecked_mut(output_word + 1) = registers[1];
            *output.get_unchecked_mut(output_word + 2) = registers[2];
            *output.get_unchecked_mut(output_word + 3) = registers[3];
        }
    }
}

fn main() {
    let context = CudaContext::new(0).expect("failed to create CUDA context");
    let (major, minor) = context
        .compute_capability()
        .expect("failed to query compute capability");
    let compute_capability = major * 10 + minor;

    // The example still builds on every host, but loading or executing an
    // ldmatrix PTX module below sm_75 would be invalid.
    if compute_capability < 75 {
        println!(
            "PASS (skipped): ldmatrix.m8n8.x4.b16 requires sm_75+; device is sm_{major}{minor}"
        );
        return;
    }

    let stream = context.default_stream();
    let module = kernels::load(&context).expect("failed to load generated ldmatrix PTX");
    let mut output =
        DeviceBuffer::<u32>::zeroed(&stream, OUTPUT_WORDS).expect("failed to allocate output");

    // SAFETY: one complete 32-lane warp executes the convergent ldmatrix
    // operation, and every lane writes four words within the live
    // LANES*MATRICES output allocation.
    unsafe {
        module
            .ldmatrix_x4_oracle(
                &stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (LANES as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut output,
            )
            .expect("failed to launch ldmatrix oracle");
    }

    let actual = output
        .to_host_vec(&stream)
        .expect("failed to copy ldmatrix results to the host");

    for lane in 0..LANES {
        let row = lane / WORDS_PER_ROW;
        let pair = lane % WORDS_PER_ROW;
        for register in 0..MATRICES {
            let expected = matrix_word(register, row, pair);
            let observed = actual[lane * MATRICES + register];
            assert_eq!(
                observed,
                expected,
                "lane {lane}, register {register}: expected matrix {register}, row {row}, \
                 columns {}..{} = {expected:#010x}, got {observed:#010x}",
                pair * 2,
                pair * 2 + 1,
            );
        }
    }

    println!("PASS: all {OUTPUT_WORDS} ldmatrix x4 fragments matched on sm_{major}{minor}");
}
