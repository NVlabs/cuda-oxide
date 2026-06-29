/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for MIR union aggregate construction.
//!
//! Every field of a Rust union is a different typed view of the same bytes.
//! This example checks construction through either field, cross-field reads,
//! unequal field sizes, arrays, nested structs, generic fields, pointers, and
//! passing a union through an ordinary device function.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C)]
union Bits {
    word: u32,
    bytes: [u8; 4],
}

// Both views accept every four-byte bit pattern, and zero is valid.
unsafe impl cuda_core::DeviceCopy for Bits {}

#[allow(dead_code)]
#[repr(C)]
union Wide {
    word: u64,
    half: u16,
    bytes: [u8; 8],
}

#[allow(dead_code)]
union Generic<T: Copy> {
    value: T,
    bytes: [u8; 8],
}

#[allow(dead_code)]
union PointerBits {
    ptr: *const u32,
    bits: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Pair {
    lo: u16,
    hi: u16,
}

#[allow(dead_code)]
union StructView {
    pair: Pair,
    word: u32,
}

struct Wrapper {
    tag: u8,
    bits: Bits,
    tail: u16,
}

#[inline(never)]
fn make_bits(word: u32) -> Bits {
    Bits { word }
}

#[inline(never)]
fn read_word(bits: Bits) -> u32 {
    unsafe { bits.word }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn union_aggregate(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let test = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = match test {
                // Construct field 0, read field 1.
                0 => unsafe { Bits { word: 0x1122_3300 }.bytes[0] as u32 },
                // Construct field 1, read field 0.
                1 => unsafe {
                    Bits {
                        bytes: [0x44, 0x33, 0x22, 0x11],
                    }
                    .word
                },
                // Union return and argument values retain their bytes.
                2 => read_word(make_bits(0x5566_7788)),
                // Assignment through a projected union field writes at offset zero.
                3 => {
                    let mut bits = Bits { word: 0 };
                    bits.bytes = [0x78, 0x56, 0x34, 0x12];
                    unsafe { bits.word }
                }
                // Consecutive union array elements use the Rust stride.
                4 => {
                    let values = [Bits { word: 0x0000_00aa }, Bits { word: 0x0000_00bb }];
                    unsafe { values[0].bytes[0] as u32 | ((values[1].bytes[0] as u32) << 8) }
                }
                // Smaller fields still begin at byte zero of wider storage.
                5 => unsafe {
                    Wide {
                        word: 0x0102_0304_0506_0708,
                    }
                    .half as u32
                },
                // Arrays of a wider union step by the full eight-byte size.
                6 => {
                    let values = [Wide { word: 0x11 }, Wide { word: 0x22 }];
                    unsafe { values[1].word as u32 }
                }
                // Generic union fields share storage too.
                7 => unsafe {
                    Generic::<u32> {
                        bytes: [0xef, 0xbe, 0xad, 0xde, 1, 2, 3, 4],
                    }
                    .value
                },
                // A pointer survives union construction and extraction.
                8 => {
                    let pointee = 0xcafe_babe;
                    let bits = PointerBits {
                        ptr: &raw const pointee,
                    };
                    unsafe { *bits.ptr }
                }
                // A union nested inside a struct keeps its offset and alignment.
                9 => {
                    let wrapper = Wrapper {
                        tag: 1,
                        bits: Bits {
                            bytes: [0x11, 0x22, 0x33, 0x44],
                        },
                        tail: 2,
                    };
                    let _ = wrapper.tag + wrapper.tail as u8;
                    unsafe { wrapper.bits.word }
                }
                // Constructing and reading the non-first field is also direct.
                10 => unsafe {
                    Bits {
                        bytes: [1, 2, 3, 4],
                    }
                    .bytes[2] as u32
                },
                // Nested projected writes still start from the union's byte zero.
                11 => {
                    let mut view = StructView { word: 0 };
                    unsafe {
                        view.pair.lo = 0x7788;
                        view.pair.hi = 0x5566;
                        view.word
                    }
                }
                _ => 0,
            };
        }
    }

    #[kernel]
    pub fn union_argument(bits: Bits, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            unsafe {
                *slot = bits.word;
            }
        }
    }
}

fn main() {
    const EXPECTED: [u32; 12] = [
        0,
        0x1122_3344,
        0x5566_7788,
        0x1234_5678,
        0x0000_bbaa,
        0x0000_0708,
        0x22,
        0xdead_beef,
        0xcafe_babe,
        0x4433_2211,
        3,
        0x5566_7788,
    ];
    const N: usize = EXPECTED.len();

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .union_aggregate(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
        .expect("kernel launch");

    let out = out_dev.to_host_vec(&stream).unwrap();
    for (i, (&got, &expected)) in out.iter().zip(EXPECTED.iter()).enumerate() {
        if got != expected {
            eprintln!("FAIL lane {i}: got {got:#x}, expected {expected:#x}");
            std::process::exit(1);
        }
    }

    let mut argument_out = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    module
        .union_argument(
            &stream,
            LaunchConfig::for_num_elems(1),
            Bits {
                bytes: [0x04, 0x03, 0x02, 0x01],
            },
            &mut argument_out,
        )
        .expect("union argument kernel launch");
    let argument_out = argument_out.to_host_vec(&stream).unwrap();
    if argument_out != [0x0102_0304] {
        eprintln!("FAIL union argument: got {argument_out:?}");
        std::process::exit(1);
    }

    println!("union_aggregate: PASS");
}
