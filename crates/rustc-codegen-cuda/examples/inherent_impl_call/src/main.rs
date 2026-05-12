//! Regression test for call-site / definition naming agreement
//! on inherent-impl method calls.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Symbol inherent_impl_call__field___impl_inherent_impl_call__
//!   backend__serial__u64__field__InvertableLimbs___pseudo_invert
//!   not found
//!   Failed operation:
//!     llvm.call @<that_symbol> (...)
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via curve25519-dalek's
//! `FieldElement51::invert`. The struct is in
//! `backend::serial::u64::field` and the inherent `impl` block is in
//! a sibling `field` module — that produces a FQDN of the form
//! `<crate>::field::<impl <crate>::backend::serial::u64::field::Foo>::method`,
//! containing `<` and `>` chars.
//!
//! `compute_export_name` (collector / definition side) switched to
//! the v0-mangled symbol whenever the FQDN had invalid PTX chars or
//! generic args. `extract_func_info` (translator / call side) only
//! switched on generic args. Non-generic inherent-impl calls hit the
//! mismatch.
//!
//! Type-and-impl in the same module (the k256 `FieldElement::invert`
//! shape) doesn't trigger it — no `<impl ...>` brackets in the FQDN.
//!
//! ## What landed
//!
//! `extract_func_info` now checks for the same invalid PTX chars
//! `compute_export_name` uses (`<`, `>`, `'`, ` `, `{`, `}`, `#`).
//! When present, the call name is taken from
//! `Instance::resolve(...).mangled_name()` instead of the FQDN —
//! matching the def side. Both sides now produce the same symbol.
//!
//! Long-term the check probably belongs in a shared helper between
//! collector and translator, but the duplicated predicate is a
//! self-contained patch that doesn't need to plumb a new map.
//!
//! ## Build with
//!
//!     cargo oxide build inherent_impl_call

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirror curve25519-dalek's layout: the type lives in
/// `backend::serial::u64::field` and the inherent `impl` block is in
/// a different sibling module (`field`). This produces a FQDN of the
/// form `crate::field::<impl crate::backend::serial::u64::field::Foo>::method`,
/// with `<impl ...>` brackets — which is what triggers
/// `compute_export_name`'s mangled-name fallback on the def side.
pub mod backend {
    pub mod serial {
        pub mod u64 {
            pub mod field {
                pub struct InvertableLimbs(pub [u64; 4]);
            }
        }
    }
}

/// Sibling module that holds the inherent impl. Re-export the type
/// name so both `field::InvertableLimbs` and `backend::serial::u64::
/// field::InvertableLimbs` resolve, matching curve25519-dalek.
pub mod field {
    pub use super::backend::serial::u64::field::InvertableLimbs;

    impl InvertableLimbs {
        /// Method on inherent impl declared in a different module
        /// than the type — FQDN becomes
        /// `<crate>::field::<impl <crate>::backend::serial::u64::field::InvertableLimbs>::pseudo_invert`,
        /// which has `<` and `>` chars.
        #[inline(never)]
        pub fn pseudo_invert(&self) -> u64 {
            self.0[0]
                .wrapping_mul(self.0[1].wrapping_add(1))
                .wrapping_add(self.0[2])
                .wrapping_add(self.0[3])
        }
    }
}

pub use backend::serial::u64::field::InvertableLimbs;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 4 <= input.len()
        {
            let base = i * 4;
            let limbs = InvertableLimbs([
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ]);
            *slot = limbs.pseudo_invert();
        }
    }
}

fn main() {
    println!("=== inherent_impl_call ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u64> = (0..(N * 4) as u64).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let base = i * 4;
        let expected = host[base]
            .wrapping_mul(host[base + 1].wrapping_add(1))
            .wrapping_add(host[base + 2])
            .wrapping_add(host[base + 3]);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: inherent-impl call resolved");
}
