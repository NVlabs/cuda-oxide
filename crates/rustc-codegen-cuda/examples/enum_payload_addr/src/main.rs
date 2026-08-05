/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Mutating an enum payload in place, through `&mut` and through assignment.
//!
//! Both forms need the address of the payload. Without enum payload addressing
//! the importer had no way to compute it: a mutable borrow was refused outright
//! rather than silently copied, and `(x as Variant).field = v` was rejected as
//! an unimplemented projection pair.
//!
//! The kernels below cover the paths that differ in lowering:
//!
//! - `assign_payload` writes through `(x as Variant).field = v`.
//! - `borrow_payload` takes `&mut` and hands it to a `#[device]` helper, so the
//!   borrow survives into a call and cannot fold into a direct store.
//! - `shared_bytes` uses an enum whose two payload variants hold different
//!   types at the same offset, so at most one of them has an LLVM slot of its
//!   own and the other is addressed by byte offset.
//! - `shared_bytes_no_slot` mutates the variant WITHOUT a slot of its own
//!   (`Bits` shares `Real`'s bytes), so the byte-offset addressing path runs
//!   against the original storage.
//! - `rebuild_payload` is the workaround this replaces, kept as a baseline.
//!
//! Each kernel reads its value back after mutating, so a write that landed in
//! a copy shows up as an unchanged element rather than passing quietly.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};
use std::time::Instant;

const LEN: u32 = 1 << 20;
const BLOCK: u32 = 256;
const RUNS: u32 = 20;

#[cuda_module]
mod kernels {
    use super::*;

    pub enum Slot {
        Occupied(f32),
        Empty,
    }

    /// Two payloads of different types sharing the same bytes, so at most one
    /// of them gets an LLVM slot and the other is addressed by offset.
    pub enum Either {
        Real(f32),
        Bits(u32),
    }

    /// Scale a borrowed payload. Taking `&mut f32` across a call boundary
    /// keeps the borrow from folding into a plain store.
    #[device]
    pub fn scale_in_place(value: &mut f32, factor: f32) {
        *value *= factor;
    }

    /// Write through `(slot as Occupied).0 = v`.
    #[kernel]
    pub fn assign_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut slot = Slot::Occupied(0.0);
        if let Slot::Occupied(value) = &mut slot {
            *value = input[i] * 2.0;
        }
        let result = match slot {
            Slot::Occupied(value) => value,
            Slot::Empty => f32::NAN,
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Pass `&mut` to the payload into a helper.
    #[kernel]
    pub fn borrow_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut slot = Slot::Occupied(input[i]);
        if let Slot::Occupied(value) = &mut slot {
            scale_in_place(value, 2.0);
        }
        let result = match slot {
            Slot::Occupied(value) => value,
            Slot::Empty => f32::NAN,
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Mutate the payload of an enum whose variants share bytes.
    #[kernel]
    pub fn shared_bytes(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut either = Either::Real(input[i]);
        if let Either::Real(value) = &mut either {
            *value *= 2.0;
        }
        let result = match either {
            Either::Real(value) => value,
            Either::Bits(bits) => f32::from_bits(bits),
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Mutate the payload that has NO slot of its own. `Real`'s f32 claims
    /// the shared bytes first, so `Bits` is addressed by byte offset off the
    /// original enum storage; a write landing in a copy (or at the wrong
    /// offset) shows up as an unchanged or corrupted element.
    #[kernel]
    pub fn shared_bytes_no_slot(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut either = Either::Bits(input[i].to_bits());
        if let Either::Bits(bits) = &mut either {
            *bits = (f32::from_bits(*bits) * 2.0).to_bits();
        }
        let result = match either {
            Either::Real(value) => value,
            Either::Bits(bits) => f32::from_bits(bits),
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// The workaround this replaces: rebuild the enum from a matched copy.
    #[kernel]
    pub fn rebuild_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let slot = Slot::Occupied(input[i]);
        let slot = match slot {
            Slot::Occupied(value) => Slot::Occupied(value * 2.0),
            Slot::Empty => Slot::Empty,
        };
        let result = match slot {
            Slot::Occupied(value) => value,
            Slot::Empty => f32::NAN,
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module_from_file("enum_payload_addr.ptx")?;
    let module = kernels::from_module(module)?;

    let host: Vec<f32> = (0..LEN).map(|i| (i % 1000) as f32 * 0.5).collect();
    let input = DeviceBuffer::from_host(&stream, &host)?;
    let config = LaunchConfig {
        grid_dim: (LEN.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let check = |name: &str,
                 run: &dyn Fn(&mut DeviceBuffer<f32>) -> Result<(), cuda_core::DriverError>|
     -> Result<(), Box<dyn std::error::Error>> {
        // Fill with a sentinel, so a kernel that never wrote is not mistaken
        // for one that wrote the right answer.
        let mut out = DeviceBuffer::from_host(&stream, &vec![f32::MIN; LEN as usize])?;
        run(&mut out)?;
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        for (i, value) in got.iter().enumerate() {
            let expected = host[i] * 2.0;
            if (value - expected).abs() > 1e-6 {
                return Err(format!("{name}: element {i} is {value}, expected {expected}").into());
            }
        }
        println!("{name}: {LEN} payloads mutated in place, exact match");
        Ok(())
    };

    // SAFETY for each launch below: the grid covers exactly `LEN` elements and
    // both buffers hold that many.
    check("assign_payload", &|out| unsafe {
        module.assign_payload(&stream, config, &input, out)
    })?;
    check("borrow_payload", &|out| unsafe {
        module.borrow_payload(&stream, config, &input, out)
    })?;
    check("shared_bytes", &|out| unsafe {
        module.shared_bytes(&stream, config, &input, out)
    })?;
    check("shared_bytes_no_slot", &|out| unsafe {
        module.shared_bytes_no_slot(&stream, config, &input, out)
    })?;
    check("rebuild_payload", &|out| unsafe {
        module.rebuild_payload(&stream, config, &input, out)
    })?;

    // In-place mutation against the rebuild-from-a-copy workaround.
    let mut out = DeviceBuffer::from_host(&stream, &vec![0.0f32; LEN as usize])?;
    let mut time = |label: &str,
                    run: &dyn Fn(&mut DeviceBuffer<f32>) -> Result<(), cuda_core::DriverError>|
     -> Result<f64, Box<dyn std::error::Error>> {
        run(&mut out)?;
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..RUNS {
            run(&mut out)?;
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
        println!("  {label:<18} {ms:7.4} ms");
        Ok(ms)
    };

    println!("\n{LEN} elements, {RUNS} timed runs:");
    let in_place = time("borrow in place", &|out| unsafe {
        module.borrow_payload(&stream, config, &input, out)
    })?;
    let rebuilt = time("rebuild from copy", &|out| unsafe {
        module.rebuild_payload(&stream, config, &input, out)
    })?;
    println!(
        "  ratio in-place / rebuild: {:.3}",
        in_place / rebuilt.max(f64::MIN_POSITIVE)
    );

    println!("\nSUCCESS");
    Ok(())
}
