// SPDX-License-Identifier: Apache-2.0
//
// A DELIBERATELY BROKEN device-kernel crate used to probe how the in-process
// `compile_to_ptx` reacts to a kernel that fails rustc's own front-end (a type
// error here). It models a real consumer feeding a malformed kernel. Do not
// "fix" the error below: the type mismatch is the point.

use cuda_core::CudaContext;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            // TYPE ERROR (intentional): assign a `&str` into an `f32` slot.
            *c_elem = "definitely not an f32";
            let _ = (a, b, idx_raw);
        }
    }
}

fn main() {
    let _ = CudaContext::new(0);
}
