// SPDX-License-Identifier: Apache-2.0

//! Negative regression for pointer provenance in a struct constant.
//!
//! The pointer field's stored bytes are a relocation record (the offset into
//! the target allocation plus a side-table entry), not an address. Decoding
//! them as field bytes would fabricate a null or garbage pointer, so the
//! importer must reject the constant with a diagnostic instead.

use cuda_device::{kernel, thread};

static FIRST: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

pub struct Holder {
    pub pointer: &'static [u8; 16],
    pub flag: bool,
}

const DIRECT: Holder = Holder {
    pointer: &FIRST,
    flag: true,
};

/// # Safety
///
/// `output` must point to writable device-accessible storage for one `u8` per
/// launched thread.
#[kernel]
pub unsafe fn direct_struct_pointer(output: *mut u8) {
    let index = thread::index_1d().get();
    let holder = DIRECT;
    unsafe {
        output
            .add(index)
            .write(holder.pointer[index & 15] + holder.flag as u8);
    }
}

fn main() {
    println!("This example must fail during device compilation.");
}
