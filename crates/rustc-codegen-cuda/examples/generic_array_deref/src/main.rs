//! Bug-96 probe: `GenericArray<T, N>::Deref` raw-pointer-cast +
//! `slice::from_raw_parts` lowering.
//!
//! ## Background
//!
//! vanity-miner-rs slot 96 fails on
//! `EncodedPoint::from_affine_coordinates(&GX, &GY, true)` with
//! hardcoded generator bytes — so the bug is not in
//! `is_identity`, `conditional_select`, or `to_bytes`. It's
//! purely in the construction. The function ([sec1
//! 0.7.3/src/point.rs:121][sec1-source]):
//!
//! ```ignore
//! pub fn from_affine_coordinates(x: &GA<u8,N>, y: &GA<u8,N>, compress: bool) -> Self {
//!     let tag = if compress { Tag::compress_y(y.as_slice()) } else { Tag::Uncompressed };
//!     let mut bytes = GenericArray::default();      // GA<u8, EncodedSize>
//!     bytes[0] = tag.into();                         // IndexMut via Deref
//!     bytes[1..(Size::to_usize() + 1)].copy_from_slice(x);
//!     if !compress {
//!         bytes[(Size::to_usize() + 1)..].copy_from_slice(y);
//!     }
//!     Self { bytes }
//! }
//! ```
//!
//! Every `bytes[i]` and `bytes[a..b]` goes through `GenericArray`'s
//! manual `Deref` to `[T]`:
//!
//! ```ignore
//! impl<T, N: ArrayLength<T>> Deref for GenericArray<T, N> {
//!     type Target = [T];
//!     fn deref(&self) -> &[T] {
//!         unsafe { slice::from_raw_parts(self as *const Self as *const T, N::USIZE) }
//!     }
//! }
//! ```
//!
//! Two operations cuda-oxide could miscompile:
//! 1. The `*const Self as *const T` pointer cast (PtrToPtr).
//! 2. `slice::from_raw_parts(ptr, len)` constructing the
//!    `&[T]` fat pointer.
//!
//! [sec1-source]: https://docs.rs/sec1/0.7.3/src/sec1/point.rs.html#121
//!
//! ## What this test locks down
//!
//! This crate manually reproduces `GenericArray<u8, U33>` without
//! the generic-array crate. Same Deref shape (raw-ptr cast +
//! from_raw_parts), same write/read patterns as
//! `from_affine_coordinates`:
//!
//! 1. `Probe::default()` — zero-init.
//! 2. `p[0] = tag` — indexed write through Deref.
//! 3. `p[1..33].copy_from_slice(&src)` — slice copy via Deref.
//! 4. Read back via `p[i]` and via slice indexing.
//!
//! If any of these miscompile, we'll see it locally in PTX.
//!
//! ## Build
//!
//!     cargo oxide build generic_array_deref

use core::ops::{Deref, DerefMut};

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors `GenericArray<u8, U33>` from the generic-array crate.
/// Inner field is `[u8; 33]` (the concrete instantiation for
/// secp256k1 compressed-point encoding: tag byte + 32 coord bytes).
pub struct Probe {
    data: [u8; 33],
}

impl Default for Probe {
    fn default() -> Self {
        Probe { data: [0u8; 33] }
    }
}

impl Deref for Probe {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        // EXACT shape from generic-array's Deref. The raw-ptr cast
        // `self as *const Self as *const u8` + `from_raw_parts` is
        // the candidate miscompile.
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, 33) }
    }
}

impl DerefMut for Probe {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self as *mut Self as *mut u8, 33) }
    }
}

/// Faithful copy of `EncodedPoint::from_affine_coordinates`'s
/// shape: zero-init via `default`, write tag at index 0, copy
/// the 32-byte coordinate into `[1..33]`. `#[inline(never)]`
/// preserves the function boundary so the codegen pattern
/// matches the cross-crate dalek/k256 cases.
#[inline(never)]
fn build_encoded(tag: u8, coord: &[u8; 32]) -> Probe {
    let mut p = Probe::default();
    p[0] = tag;
    p[1..33].copy_from_slice(coord);
    p
}

#[inline(never)]
fn check_generic_array_deref() -> u32 {
    // Shape 1: write tag at index 0 via DerefMut; read back via
    // Deref. If the raw-ptr cast or from_raw_parts miscompiles
    // here, the byte we wrote ends up at the wrong address.
    {
        let mut p = Probe::default();
        let tag = core::hint::black_box(0x02u8);
        p[0] = tag;
        if p[0] != tag {
            return 0;
        }
    }

    // Shape 2: bytes[1..33] copy_from_slice with hardcoded
    // generator-coord-like bytes. Reads back each byte and
    // confirms position.
    {
        let coord_host: [u8; 32] = [
            0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
            0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
            0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
            0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
        ];
        // black_box the input so the compiler can't fold the copy.
        let coord = core::hint::black_box(coord_host);
        let p = build_encoded(0x02, &coord);

        if p[0] != 0x02 {
            return 0;
        }
        // Verify every coordinate byte made it through.
        for i in 0..32 {
            if p[1 + i] != coord[i] {
                return 0;
            }
        }
    }

    // Shape 3: full slice extraction. `p.as_slice()` (== `p.deref()`)
    // should produce a 33-byte slice with the right content.
    {
        let coord_host: [u8; 32] = [
            0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65,
            0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
            0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19,
            0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8,
        ];
        let coord = core::hint::black_box(coord_host);
        let p = build_encoded(0x03, &coord);

        // Borrow as full slice (exercises GA::Deref directly).
        let s: &[u8] = &p[..];
        if s.len() != 33 {
            return 0;
        }
        if s[0] != 0x03 {
            return 0;
        }
        for i in 0..32 {
            if s[1 + i] != coord[i] {
                return 0;
            }
        }
    }

    1
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            *slot = check_generic_array_deref();
        }
    }
}

fn main() {
    println!("=== generic_array_deref ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let n_out = 4usize;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, n_out).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(n_out as u32), &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for (i, r) in result.iter().enumerate() {
        assert_eq!(*r, 1, "thread {} got {} (expected 1)", i, r);
    }
    println!("SUCCESS: GenericArray<u8, U33>-shape Deref + index + copy_from_slice");
}
