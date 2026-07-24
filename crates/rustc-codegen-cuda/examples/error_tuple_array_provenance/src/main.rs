// SPDX-License-Identifier: Apache-2.0

//! Negative regression for pointer provenance in tuple array constants.

use cuda_device::{kernel, thread};

static FIRST: u32 = 11;
static SECOND: u32 = 17;
const POINTERS: [(&u32, bool); 2] = [(&FIRST, false), (&SECOND, true)];

/// # Safety
///
/// `output` must point to writable device-accessible storage for one `u32` per
/// launched thread.
#[kernel]
pub unsafe fn tuple_array_pointer(output: *mut u32) {
    let index = thread::index_1d().get();
    let (pointer, flag) = POINTERS[index & 1];
    unsafe {
        output.add(index).write(*pointer + flag as u32);
    }
}

fn main() {
    println!("This example must fail during device compilation.");
}
