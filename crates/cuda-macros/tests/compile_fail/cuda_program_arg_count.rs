#![allow(unused_imports)]

use cuda_core::DeviceBuffer;
use cuda_macros::{cuda_program, kernel};

pub struct DisjointSlice<T>(*mut T);

#[cuda_program]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fill(mut output: DisjointSlice<f32>, value: f32) {
        let _ = &mut output;
        let _ = value;
    }

    #[program]
    pub fn forward(output: &mut DeviceBuffer<f32>, n: u32) {
        fill(output).grid_len(n);
    }
}

fn main() {}
