// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{
    DisjointSlice,
    atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64},
    kernel, thread,
};
use cuda_host::cuda_module;

const N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn integer_rmw(
        counter_u32: &[DeviceAtomicU32],
        counter_u64: &[DeviceAtomicU64],
        mut old_values: DisjointSlice<(u32, u64)>,
    ) {
        let index = thread::index_1d();
        if index.get() >= N {
            return;
        }

        // AcqRel deliberately exercises the fence-splitting path: the LLVM
        // atomicrmw that reaches legacy NVVM is monotonic, while the fences
        // preserve the source ordering.
        let old_u32 = counter_u32[0].fetch_add(1, AtomicOrdering::AcqRel);
        let old_u64 = counter_u64[0].fetch_add(1, AtomicOrdering::AcqRel);
        if let Some(slot) = old_values.get_mut(index) {
            *slot = (old_u32, old_u64);
        }
    }

    #[allow(clippy::manual_unwrap_or)]
    #[kernel]
    pub fn integer_cas(
        counter_u32: &[DeviceAtomicU32],
        counter_u64: &[DeviceAtomicU64],
        mut observed_u32: DisjointSlice<(u32, u32)>,
        mut observed_u64: DisjointSlice<(u64, u64)>,
    ) {
        let index = thread::index_1d();
        if index.get() != 0 {
            return;
        }

        // Relaxed on both sides: the legacy legalizer rejects ordered CAS
        // because libNVVM lowers ordered cmpxchg to a bare, unordered
        // atom.cas. Issue #922 tracks ordered CAS on the legacy path via
        // inline PTX.
        let success_u32 = match counter_u32[0].compare_exchange(
            7,
            11,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(old) => old,
            Err(_) => u32::MAX,
        };
        let failure_u32 = match counter_u32[0].compare_exchange(
            7,
            13,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => u32::MAX,
            Err(old) => old,
        };

        let success_u64 = match counter_u64[0].compare_exchange(
            7,
            11,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(old) => old,
            Err(_) => u64::MAX,
        };
        let failure_u64 = match counter_u64[0].compare_exchange(
            7,
            13,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => u64::MAX,
            Err(old) => old,
        };

        if let Some(slot) = observed_u32.get_mut(thread::index_1d()) {
            *slot = (success_u32, failure_u32);
        }
        if let Some(slot) = observed_u64.get_mut(thread::index_1d()) {
            *slot = (success_u64, failure_u64);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;

    let rmw_u32 = DeviceBuffer::<u32>::zeroed(&stream, 1)?.cast_elem::<DeviceAtomicU32>();
    let rmw_u64 = DeviceBuffer::<u64>::zeroed(&stream, 1)?.cast_elem::<DeviceAtomicU64>();
    let mut old_values = DeviceBuffer::<(u32, u64)>::zeroed(&stream, N)?;

    // SAFETY: the launch covers exactly N unique one-dimensional indices.
    // `old_values` has N elements, and both one-element counters use atomic
    // wrapper pointees so all shared updates are atomic.
    unsafe {
        module.integer_rmw(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &rmw_u32,
            &rmw_u64,
            &mut old_values,
        )?;
    }
    stream.synchronize()?;

    let got_u32 = rmw_u32.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let got_u64 = rmw_u64.cast_elem::<u64>().to_host_vec(&stream)?[0];
    let old_values = old_values.to_host_vec(&stream)?;
    let mut old_u32 = old_values
        .iter()
        .map(|&(value, _)| value)
        .collect::<Vec<_>>();
    let mut old_u64 = old_values
        .iter()
        .map(|&(_, value)| value)
        .collect::<Vec<_>>();
    old_u32.sort_unstable();
    old_u64.sort_unstable();

    let old_u32_is_permutation = old_u32
        .iter()
        .enumerate()
        .all(|(index, &value)| value == index as u32);
    let old_u64_is_permutation = old_u64
        .iter()
        .enumerate()
        .all(|(index, &value)| value == index as u64);
    if got_u32 != N as u32
        || got_u64 != N as u64
        || !old_u32_is_permutation
        || !old_u64_is_permutation
    {
        return Err(format!(
            "legacy integer RMW mismatch: u32={got_u32}, u64={got_u64}, old_u32_permutation={old_u32_is_permutation}, old_u64_permutation={old_u64_is_permutation}"
        )
        .into());
    }

    let cas_u32 = DeviceBuffer::from_host(&stream, &[7_u32])?.cast_elem::<DeviceAtomicU32>();
    let cas_u64 = DeviceBuffer::from_host(&stream, &[7_u64])?.cast_elem::<DeviceAtomicU64>();
    let mut observed_u32 = DeviceBuffer::<(u32, u32)>::zeroed(&stream, 1)?;
    let mut observed_u64 = DeviceBuffer::<(u64, u64)>::zeroed(&stream, 1)?;

    // SAFETY: exactly one thread executes the two CAS probes for each
    // one-element atomic counter and writes one element in each output buffer.
    unsafe {
        module.integer_cas(
            &stream,
            LaunchConfig::for_num_elems(1),
            &cas_u32,
            &cas_u64,
            &mut observed_u32,
            &mut observed_u64,
        )?;
    }
    stream.synchronize()?;

    let observed_u32 = observed_u32.to_host_vec(&stream)?[0];
    let observed_u64 = observed_u64.to_host_vec(&stream)?[0];
    let final_u32 = cas_u32.cast_elem::<u32>().to_host_vec(&stream)?[0];
    let final_u64 = cas_u64.cast_elem::<u64>().to_host_vec(&stream)?[0];
    if observed_u32 != (7, 11) || observed_u64 != (7, 11) || final_u32 != 11 || final_u64 != 11 {
        return Err(format!(
            "legacy integer CAS mismatch: observed_u32={observed_u32:?}, observed_u64={observed_u64:?}, final_u32={final_u32}, final_u64={final_u64}"
        )
        .into());
    }

    println!(
        "legacy_atomic_rmw_cas: PASS (rmw_u32={got_u32}, rmw_u64={got_u64}, cas_u32={final_u32}, cas_u64={final_u64})"
    );
    Ok(())
}
