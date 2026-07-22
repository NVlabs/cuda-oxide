// SPDX-License-Identifier: Apache-2.0

//! Regression for generic branches controlled by an associated constant.

use core::marker::PhantomData;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const N: usize = 4;

#[repr(transparent)]
struct Tagged<T, M> {
    value: T,
    marker: PhantomData<M>,
}

impl<T: Copy, M> Copy for Tagged<T, M> {}

impl<T: Copy, M> Clone for Tagged<T, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: core::fmt::Debug, M> core::fmt::Debug for Tagged<T, M> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.value.fmt(formatter)
    }
}

impl<T: PartialEq, M> PartialEq for Tagged<T, M> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T, M> Tagged<T, M> {
    const fn new(value: T) -> Self {
        Self {
            value,
            marker: PhantomData,
        }
    }
}

// SAFETY: `Tagged<T, M>` is transparent over `T`; `PhantomData<M>` has no
// runtime representation.
unsafe impl<T: DeviceCopy, M> DeviceCopy for Tagged<T, M> {}

trait ExplicitMode: Sized + 'static {
    const ENABLED: bool;
    fn hook(value: u32) -> u32;
}

struct ExplicitOn;
struct ExplicitOff;

impl ExplicitMode for ExplicitOn {
    const ENABLED: bool = true;

    #[inline(never)]
    fn hook(value: u32) -> u32 {
        value * 3
    }
}

impl ExplicitMode for ExplicitOff {
    const ENABLED: bool = false;

    #[inline(never)]
    fn hook(_value: u32) -> u32 {
        unreachable!()
    }
}

trait DefaultMode: Sized + 'static {
    const ENABLED: bool;

    #[inline(never)]
    fn hook(_value: u32) -> u32 {
        unreachable!()
    }
}

struct DefaultOn;
struct DefaultOff;

impl DefaultMode for DefaultOn {
    const ENABLED: bool = true;

    #[inline(never)]
    fn hook(value: u32) -> u32 {
        value + 11
    }
}

impl DefaultMode for DefaultOff {
    const ENABLED: bool = false;
}

trait DynamicMode: Sized + 'static {
    fn hook(value: u32) -> u32;
}

struct Dynamic;

impl DynamicMode for Dynamic {
    #[inline(never)]
    fn hook(value: u32) -> u32 {
        value + 100
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn select_explicit<M: ExplicitMode>(value: u32) -> u32 {
        if M::ENABLED { M::hook(value) } else { value }
    }

    #[inline(never)]
    fn select_default<M: DefaultMode>(value: u32) -> u32 {
        if M::ENABLED { M::hook(value) } else { value }
    }

    #[inline(never)]
    fn select_dynamic<M: DynamicMode>(enabled: bool, value: u32) -> u32 {
        if enabled { M::hook(value) } else { value }
    }

    #[kernel]
    pub fn explicit<M: ExplicitMode>(
        input: &[Tagged<u32, M>],
        mut output: DisjointSlice<Tagged<u32, M>>,
    ) {
        let index = thread::index_1d();
        if let (Some(value), Some(slot)) = (input.get(index.get()), output.get_mut(index)) {
            *slot = Tagged::new(select_explicit::<M>(value.value));
        }
    }

    #[kernel]
    pub fn defaulted<M: DefaultMode>(
        input: &[Tagged<u32, M>],
        mut output: DisjointSlice<Tagged<u32, M>>,
    ) {
        let index = thread::index_1d();
        if let (Some(value), Some(slot)) = (input.get(index.get()), output.get_mut(index)) {
            *slot = Tagged::new(select_default::<M>(value.value));
        }
    }

    #[kernel]
    pub fn dynamic<M: DynamicMode>(
        input: &[Tagged<u32, M>],
        mut output: DisjointSlice<Tagged<u32, M>>,
    ) {
        let index = thread::index_1d();
        if let (Some(value), Some(slot)) = (input.get(index.get()), output.get_mut(index)) {
            let enabled = value.value & 1 == 1;
            *slot = Tagged::new(select_dynamic::<M>(enabled, value.value));
        }
    }
}

fn tagged<M>(values: &[u32]) -> Vec<Tagged<u32, M>> {
    values.iter().copied().map(Tagged::new).collect()
}

fn check<M>(
    name: &str,
    stream: &Arc<CudaStream>,
    input: &[Tagged<u32, M>],
    expected: &[Tagged<u32, M>],
    launch: impl FnOnce(LaunchConfig, &DeviceBuffer<Tagged<u32, M>>, &mut DeviceBuffer<Tagged<u32, M>>),
) -> bool {
    let device_input = DeviceBuffer::from_host(stream, input).expect("input allocation");
    let mut device_output =
        DeviceBuffer::zeroed(stream, expected.len()).expect("output allocation");
    launch(
        LaunchConfig::for_num_elems(expected.len() as u32),
        &device_input,
        &mut device_output,
    );
    let output = device_output.to_host_vec(stream).expect("copy output");
    let passed = output == expected;
    println!("{name}: {}", if passed { "PASS" } else { "FAIL" });
    passed
}

fn main() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load module");
    let input = [1, 2, 3, 4];
    let tripled = [3, 6, 9, 12];
    let incremented = [12, 13, 14, 15];
    let dynamically_selected = [101, 2, 103, 4];

    let mut passed = true;
    passed &= check(
        "explicit live arm",
        &stream,
        &tagged::<ExplicitOn>(&input),
        &tagged::<ExplicitOn>(&tripled),
        |config, device_input, device_output| {
            // SAFETY: the launch covers the output and the kernel bounds-checks it.
            unsafe {
                module
                    .explicit::<ExplicitOn>(&stream, config, device_input, device_output)
                    .expect("launch")
            }
        },
    );
    passed &= check(
        "explicit dead arm",
        &stream,
        &tagged::<ExplicitOff>(&input),
        &tagged::<ExplicitOff>(&input),
        |config, device_input, device_output| {
            // SAFETY: the launch covers the output and the kernel bounds-checks it.
            unsafe {
                module
                    .explicit::<ExplicitOff>(&stream, config, device_input, device_output)
                    .expect("launch")
            }
        },
    );
    passed &= check(
        "default live arm",
        &stream,
        &tagged::<DefaultOn>(&input),
        &tagged::<DefaultOn>(&incremented),
        |config, device_input, device_output| {
            // SAFETY: the launch covers the output and the kernel bounds-checks it.
            unsafe {
                module
                    .defaulted::<DefaultOn>(&stream, config, device_input, device_output)
                    .expect("launch")
            }
        },
    );
    passed &= check(
        "default dead arm",
        &stream,
        &tagged::<DefaultOff>(&input),
        &tagged::<DefaultOff>(&input),
        |config, device_input, device_output| {
            // SAFETY: the launch covers the output and the kernel bounds-checks it.
            unsafe {
                module
                    .defaulted::<DefaultOff>(&stream, config, device_input, device_output)
                    .expect("launch")
            }
        },
    );
    passed &= check(
        "dynamic conservative arms",
        &stream,
        &tagged::<Dynamic>(&input),
        &tagged::<Dynamic>(&dynamically_selected),
        |config, device_input, device_output| {
            // SAFETY: the launch covers the output and the kernel bounds-checks it.
            unsafe {
                module
                    .dynamic::<Dynamic>(&stream, config, device_input, device_output)
                    .expect("launch")
            }
        },
    );

    if passed {
        println!("const_bool_dead_branch: PASS ({N} values, five generic instantiations)");
    } else {
        std::process::exit(1);
    }
}
