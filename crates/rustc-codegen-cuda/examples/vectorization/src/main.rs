/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Alignment-driven vectorization across the CUDA built-in vector types.
//!
//! Each type is the Rust equivalent of a CUDA vector type (e.g. `f32x4` =
//! `float4`), with its exact CUDA size and alignment. A per-type copy kernel
//! (`output[i] = input[i]`) shows how alignment governs whether the whole-
//! element load/store fuses into a vectorized `ld/st.global.v*` or stays
//! scalar. For every type, `main` checks the Rust layout matches CUDA, the
//! round-trip copy is bit-correct on the GPU, and reports/asserts the codegen.
//!
//! Run: `cargo oxide run vectorization`

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::cuda_module;

/// One row of the per-type report.
struct Row {
    name: &'static str,
    cuda: &'static str,
    size: usize,
    align: usize,
    kernel: &'static str,
    copy_ok: bool,
}

/// Generate, from a single table, the CUDA-style vector types, the
/// `#[cuda_module]` of per-type copy kernels, and `run_all` (launch + verify
/// each). The macro emits the *whole* `#[cuda_module]` so the kernel repetition
/// expands before the attribute macro runs (a `macro_rules!` *inside* the
/// module would be invisible to it).
macro_rules! vector_suite {
    ($($ty:ident: $base:ty; $n:literal; align $align:literal; size $size:literal; $cuda:literal,)*) => {
        $(
            #[repr(C, align($align))]
            #[derive(Clone, Copy, PartialEq, Debug)]
            pub struct $ty([$base; $n]);
            // Plain POD aggregate, no pointers: safe to memcpy to/from the device.
            unsafe impl cuda_core::DeviceCopy for $ty {}
        )*

        #[cuda_module]
        mod kernels {
            use cuda_device::{kernel, thread, DisjointSlice};
            $(
                #[kernel]
                pub fn $ty(input: &[super::$ty], mut output: DisjointSlice<super::$ty>) {
                    let idx = thread::index_1d();
                    let i = idx.get();
                    if let Some(o) = output.get_mut(idx) {
                        *o = input[i];
                    }
                }
            )*
        }

        /// Launch every kernel, asserting each type's Rust layout matches CUDA
        /// and its copy round-trips.
        fn run_all(
            module: &kernels::LoadedModule,
            stream: &cuda_core::CudaStream,
            cfg: LaunchConfig,
            n: usize,
        ) -> Vec<Row> {
            let mut rows = Vec::new();
            $(
                {
                    assert_eq!(core::mem::size_of::<$ty>(), $size, concat!(stringify!($ty), " size"));
                    assert_eq!(core::mem::align_of::<$ty>(), $align, concat!(stringify!($ty), " align"));
                    let input: Vec<$ty> = (0..n)
                        .map(|i| $ty(core::array::from_fn(|j| (i * $n + j + 1) as $base)))
                        .collect();
                    let in_dev = DeviceBuffer::from_host(stream, &input).unwrap();
                    let mut out_dev = DeviceBuffer::<$ty>::zeroed(stream, n).unwrap();
                    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
                    unsafe { module
                        .$ty(stream, cfg, &in_dev, &mut out_dev) }
                        .expect(concat!(stringify!($ty), " launch"));
                    let out = out_dev.to_host_vec(stream).unwrap();
                    rows.push(Row {
                        name: stringify!($ty),
                        cuda: $cuda,
                        size: $size,
                        align: $align,
                        kernel: stringify!($ty),
                        copy_ok: out == input,
                    });
                }
            )*
            rows
        }
    };
}

vector_suite! {
    i8x1: i8; 1; align 1; size 1; "char1",
    i8x2: i8; 2; align 2; size 2; "char2",
    i8x3: i8; 3; align 1; size 3; "char3",
    i8x4: i8; 4; align 4; size 4; "char4",
    i16x1: i16; 1; align 2; size 2; "short1",
    i16x2: i16; 2; align 4; size 4; "short2",
    i16x3: i16; 3; align 2; size 6; "short3",
    i16x4: i16; 4; align 8; size 8; "short4",
    i32x1: i32; 1; align 4; size 4; "int1",
    i32x2: i32; 2; align 8; size 8; "int2",
    i32x3: i32; 3; align 4; size 12; "int3",
    i32x4: i32; 4; align 16; size 16; "int4",
    i64x1: i64; 1; align 8; size 8; "longlong1",
    i64x2: i64; 2; align 16; size 16; "longlong2",
    i64x3: i64; 3; align 8; size 24; "longlong3",
    i64x4: i64; 4; align 16; size 32; "longlong4",
    i64x4_a32: i64; 4; align 32; size 32; "longlong4_32a",
    f32x1: f32; 1; align 4; size 4; "float1",
    f32x2: f32; 2; align 8; size 8; "float2",
    f32x3: f32; 3; align 4; size 12; "float3",
    f32x4: f32; 4; align 16; size 16; "float4",
    f64x1: f64; 1; align 8; size 8; "double1",
    f64x2: f64; 2; align 16; size 16; "double2",
    f64x3: f64; 3; align 8; size 24; "double3",
    f64x4: f64; 4; align 16; size 32; "double4",
    f64x4_a32: f64; 4; align 32; size 32; "double4_32a",
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    const N: usize = 256;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let module = ctx
        .load_module_from_file("vectorization.ptx")
        .expect("Failed to load vectorization.ptx");
    let module = kernels::from_module(module).expect("Failed to initialize typed module");

    let rows = run_all(&module, &stream, cfg, N);

    // Inspect the PTX we just launched to report/assert the codegen shape.
    let ptx = std::fs::read_to_string("vectorization.ptx")
        .expect("vectorization.ptx not found (run with `cargo oxide run vectorization`)");

    println!(
        "{:<11} {:<14} {:>4} {:>5}  {:<22} vectorized",
        "rust", "cuda", "size", "align", "ptx load"
    );
    let mut errors = 0;
    for r in &rows {
        let body = kernel_body(&ptx, r.kernel);
        let load = first_mem_op(body);
        let vectorized = load.contains(".v2.") || load.contains(".v4.") || load.contains(".v8.");
        if !r.copy_ok {
            errors += 1;
            println!("  !! {} round-trip copy mismatch", r.name);
        }
        // The robust, alignment-gated invariant this example exists to show: a
        // type aligned to the 128-bit vector width (or wider) always fuses into
        // a vector `ld/st.global.v*`. (Some smaller types also vectorize, and
        // the 8-byte ones coalesce into a single `b64` -- reported, not asserted.)
        let expect_vec = r.align >= 16;
        if expect_vec && !vectorized {
            errors += 1;
            println!("  !! {} expected to vectorize but did not", r.name);
        }
        println!(
            "{:<11} {:<14} {:>4} {:>5}  {:<22} {}",
            r.name,
            r.cuda,
            r.size,
            r.align,
            load,
            if vectorized { "yes" } else { "no" }
        );
    }

    if errors == 0 {
        println!(
            "\n\u{2713} SUCCESS: {} CUDA vector types -- layout, copy, and codegen all correct",
            rows.len()
        );
    } else {
        println!("\n\u{2717} FAILED: {} problem(s)", errors);
        std::process::exit(1);
    }
}

/// PTX text of a `.visible .entry <name>` kernel, header to closing `}`.
fn kernel_body<'a>(ptx: &'a str, name: &str) -> &'a str {
    let start = ptx
        .find(&format!(".visible .entry {name}("))
        .unwrap_or_else(|| panic!("kernel `{name}` not found in PTX"));
    let body = &ptx[start..];
    let end = body.find("\n}").map_or(body.len(), |e| e + 2);
    &body[..end]
}

/// First global memory op mnemonic in a kernel body (e.g. `ld.global.v2.b64`).
fn first_mem_op(body: &str) -> String {
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t
            .strip_prefix("ld.global.")
            .or_else(|| t.strip_prefix("st.global."))
        {
            let mnem: String = rest.split_whitespace().next().unwrap_or("").to_string();
            let kind = if t.starts_with("ld") {
                "ld.global."
            } else {
                "st.global."
            };
            return format!("{kind}{mnem}");
        }
    }
    "(none)".to_string()
}
