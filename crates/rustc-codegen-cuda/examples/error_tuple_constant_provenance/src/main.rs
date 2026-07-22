// SPDX-License-Identifier: Apache-2.0

//! Negative regression for pointer provenance in a direct tuple constant.

use cuda_device::{kernel, thread};

static FIRST: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const DIRECT: (&[u8; 16], bool) = (&FIRST, true);

/// # Safety
///
/// `output` must point to writable device-accessible storage for one `u8` per
/// launched thread.
#[kernel]
pub unsafe fn direct_tuple_pointer(output: *mut u8) {
    let index = thread::index_1d().get();
    let (pointer, flag) = DIRECT;
    unsafe {
        output.add(index).write(pointer[index & 15] + flag as u8);
    }
}

fn main() {
    println!("This example must fail during device compilation.");
}
