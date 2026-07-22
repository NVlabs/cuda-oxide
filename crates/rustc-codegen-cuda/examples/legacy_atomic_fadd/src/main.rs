// SPDX-License-Identifier: Apache-2.0

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{
    DisjointSlice,
    atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64},
    kernel, thread,
};
use cuda_host::cuda_module;

const N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn atomic_add(
        counter_f32: &mut [f32],
        counter_f64: &mut [f64],
        mut completed: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d();
        if index.get() >= N {
            return;
        }

        // SAFETY: both buffers contain one correctly aligned scalar. The
        // atomic wrappers provide interior mutability for concurrent updates.
        let counter_f32 = unsafe { DeviceAtomicF32::from_ptr(counter_f32.as_mut_ptr()) };
        let counter_f64 = unsafe { DeviceAtomicF64::from_ptr(counter_f64.as_mut_ptr()) };
        counter_f32.fetch_add(1.0, AtomicOrdering::Relaxed);
        counter_f64.fetch_add(1.0, AtomicOrdering::Relaxed);

        if let Some(slot) = completed.get_mut(index) {
            *slot = 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let mut counter_f32 = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut counter_f64 = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
    let mut completed = DeviceBuffer::<u32>::zeroed(&stream, N)?;

    // SAFETY: the launch covers N threads; both counters contain one scalar,
    // and the completion buffer contains N elements.
    unsafe {
        module.atomic_add(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut counter_f32,
            &mut counter_f64,
            &mut completed,
        )?;
    }
    stream.synchronize()?;

    let got_f32 = counter_f32.to_host_vec(&stream)?[0];
    let got_f64 = counter_f64.to_host_vec(&stream)?[0];
    let completed = completed.to_host_vec(&stream)?;
    if got_f32 != N as f32 || got_f64 != N as f64 || completed.iter().any(|&value| value != 1) {
        return Err(format!(
            "legacy atomic add mismatch: f32={got_f32}, f64={got_f64}, completed={}",
            completed.iter().filter(|&&value| value == 1).count()
        )
        .into());
    }

    println!("legacy_atomic_fadd: PASS (f32={got_f32}, f64={got_f64})");
    Ok(())
}
