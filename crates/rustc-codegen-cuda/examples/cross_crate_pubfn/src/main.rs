//! Regression test for cross-crate `pub fn` codegen: a kernel calls
//! into a GPU-agnostic dependency crate (no `cuda-device` dep, no
//! `#[device]`/`#[cuda_module]`/`#[kernel]` annotations, no `#[inline]`
//! on the public surface) and is expected to build and produce real
//! PTX for every reachable callee.
//!
//! This is the shape `~/vanity-miner-rs` actually uses:
//! `shallenge-logic/` is the analogue of `vanity-miner-rs/logic/` —
//! pure no_std Rust shared between a CPU mode (called directly) and a
//! GPU mode (called from a `#[kernel]`). The logic crate must remain
//! cuda-oxide-blind, so the kernel binary cannot rely on `#[device]`
//! or even `#[inline]` reaching into it.
//!
//! ## The bug this guards against
//!
//! Before the fix, this example failed at `llc` verification with:
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Verification failed for 'llvm module':
//!        Symbol shallenge_logic__generate_and_check_shallenge not found
//! ```
//!
//! Tracing through `collector.rs::process_call_operand`:
//!
//! 1. The kernel's `Terminator::Call` to
//!    `shallenge_logic::generate_and_check_shallenge` resolved to a
//!    fully-monomorphized `Instance`.
//! 2. `should_collect_from_crate` returned `Collect` — external
//!    crates reachable from a kernel are allowed.
//! 3. `is_mir_available(resolved.def_id())` returned `false`. The
//!    collector silently returned. No body was queued for emission.
//! 4. The MIR translator on the caller side had already emitted a
//!    `call.uni` to `shallenge_logic__generate_and_check_shallenge`.
//!    llc verified, found no definition, errored out.
//!
//! Step 3 is the load-bearing one. Rustc only encodes MIR for
//! cross-crate consumers when an item is generic, `#[inline]`, or
//! the build sets `-Z always-encode-mir`. Non-generic non-inline
//! `pub fn`s in a dep crate have only their signature in the rmeta;
//! the codegen backend never sees the body.
//!
//! The diagnostic signature: marking a single fn `#[inline]` did
//! not fix the build, it just moved the failure one level deeper
//! to the next non-inline callee
//! (`shallenge_logic__generate_base64_nonce`). `#[inline]` cascading
//! through every dep fn is exactly the "fork every dep" non-fix the
//! GPU-agnostic-crate pattern is meant to avoid.
//!
//! ## The fix
//!
//! One line in `crates/cargo-oxide/src/commands.rs::build_rustflags`:
//! append `-Z always-encode-mir=yes`. This forces every crate
//! compiled through cargo-oxide to encode its full MIR into the
//! rmeta. The existing `is_mir_available` check in `collector.rs`
//! then just starts returning `true` for cross-crate `pub fn`s, the
//! body gets walked and emitted, and the build succeeds.
//!
//! Cost: ~10-30% larger intermediate rmetas in `target/` (rust-cuda
//! anecdote). No impact on emitted PTX or host-side binary size —
//! the collector still walks reachable-from-kernel only, so
//! unreachable dep code is never codegened.
//!
//! ## What this example is NOT
//!
//! - Not a rerun of `cross_crate_kernel`. That example puts
//!   `#[cuda_module]` *inside* the library crate, so the library is
//!   already part of the cuda-aware compilation surface. This
//!   example gives the library no cuda annotations whatsoever — the
//!   "consume a no_std crate unchanged" workload.
//! - Not about `thread::*` intrinsics or the `#[device]` macro.
//!   Those are covered by `helper_outside_module` and
//!   `helper_no_inline` and are orthogonal: they fire on intrinsic-
//!   wrapping helpers in *any* unannotated location (same crate,
//!   sibling mod, dep crate). This bug is purely about MIR
//!   availability cross-crate for pure-arithmetic callees.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicU32},
    cuda_module, kernel, thread,
};

use shallenge_logic::{
    ShallengeRequest, ShallengeResult, generate_and_check_shallenge, shallenge,
};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[inline]
    pub unsafe fn atomic_add_u32(address: &mut u32, val: u32) -> u32 {
        unsafe {
            DeviceAtomicU32::from_ptr(address as *mut u32)
                .fetch_add(val, AtomicOrdering::Relaxed)
        }
    }

    #[inline]
    unsafe fn handle_shallenge_match_found(
        result: ShallengeResult,
        thread_idx: usize,
        found_matches: &mut [u32],
        found_hash: &mut [u8],
        found_nonce: &mut [u8],
        found_nonce_len: &mut [usize],
        found_thread_idx: &mut [u32],
    ) {
        found_hash.copy_from_slice(&result.hash);
        found_nonce.copy_from_slice(&result.nonce);
        found_nonce_len[0] = result.nonce_len;
        found_thread_idx[0] = thread_idx as u32;
        unsafe { atomic_add_u32(&mut found_matches[0], 1) };
    }

    /// Shallenge search kernel. Every helper call goes into the
    /// `shallenge_logic` sibling crate, which has no `#[device]`
    /// and no `#[inline]`. The `thread::index_1d()` call is
    /// inlined here so the intrinsic rewrite hook fires; the only
    /// failure surface is the cross-crate `pub fn` calls.
    #[kernel]
    #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
    pub unsafe fn kernel_find_better_shallenge_nonce(
        username: &[u8],
        target_hash: &[u8],
        rng_seed: u64,
        found_matches: &mut [u32],
        found_hash: &mut [u8],
        found_nonce: &mut [u8],
        found_nonce_len: &mut [usize],
        found_thread_idx: &mut [u32],
    ) {
        let thread_idx = thread::index_1d().get();
        let username_len = username.len();
        let target_hash_array: &[u8; 32] =
            unsafe { &*(target_hash.as_ptr() as *const [u8; 32]) };

        let request = ShallengeRequest {
            username,
            username_len,
            target_hash: target_hash_array,
            thread_idx,
            rng_seed,
        };

        // ← The failing call. Resolves to a fully-monomorphized
        // Instance, `should_collect_from_crate` returns Collect,
        // but `is_mir_available` returns false because rustc did
        // not encode MIR for this non-generic non-inline pub fn
        // into shallenge-logic's rmeta. Body never reaches
        // codegen, llc rejects the dangling reference.
        let result = generate_and_check_shallenge(&request);

        if result.is_better {
            unsafe {
                handle_shallenge_match_found(
                    result,
                    thread_idx,
                    found_matches,
                    found_hash,
                    found_nonce,
                    found_nonce_len,
                    found_thread_idx,
                );
            }
        }
    }
}

// ============================================================================
// Host harness — identical shape to shallenge_repro/. Won't run
// while the cross-crate bug is open.
// ============================================================================

fn main() {
    println!("=== cross_crate_pubfn ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let username = b"testuser";
    let target_hash: [u8; 32] = [0xffu8; 32];

    let username_dev = DeviceBuffer::from_host(&stream, username.as_slice()).unwrap();
    let target_hash_dev = DeviceBuffer::from_host(&stream, &target_hash).unwrap();
    let mut found_matches_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    let mut found_hash_dev = DeviceBuffer::<u8>::zeroed(&stream, 32).unwrap();
    let mut found_nonce_dev = DeviceBuffer::<u8>::zeroed(&stream, 64).unwrap();
    let mut found_nonce_len_dev = DeviceBuffer::<usize>::zeroed(&stream, 1).unwrap();
    let mut found_thread_idx_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");

    let rng_seed: u64 = 0xdeadbeefcafebabe;

    unsafe {
        module
            .kernel_find_better_shallenge_nonce(
                &stream,
                LaunchConfig::for_num_elems(256),
                &username_dev,
                &target_hash_dev,
                rng_seed,
                &mut found_matches_dev,
                &mut found_hash_dev,
                &mut found_nonce_dev,
                &mut found_nonce_len_dev,
                &mut found_thread_idx_dev,
            )
            .expect("kernel launch");
    }

    let mut found_matches = [0u32; 1];
    let mut found_hash = [0u8; 32];
    let mut found_nonce = [0u8; 64];
    let mut found_nonce_len = [0usize; 1];

    found_matches_dev
        .copy_to_host(&stream, &mut found_matches)
        .unwrap();
    found_hash_dev
        .copy_to_host(&stream, &mut found_hash)
        .unwrap();
    found_nonce_dev
        .copy_to_host(&stream, &mut found_nonce)
        .unwrap();
    found_nonce_len_dev
        .copy_to_host(&stream, &mut found_nonce_len)
        .unwrap();
    stream.synchronize().unwrap();

    let nl = found_nonce_len[0];
    println!("found_matches = {}", found_matches[0]);
    println!("found_nonce_len = {}", nl);
    println!("found_hash[..8] = {:02x?}", &found_hash[..8]);

    if found_matches[0] == 0 {
        eprintln!();
        eprintln!("FAIL: kernel produced 0 matches.");
        std::process::exit(1);
    }

    if nl != 21 {
        eprintln!("FAIL: found_nonce_len = {} (expected 21).", nl);
        std::process::exit(1);
    }

    let nonce_slice = &found_nonce[..nl];
    let expected = shallenge(username, username.len(), nonce_slice, nl);
    if expected != found_hash {
        eprintln!("FAIL: hash mismatch between device and host recompute.");
        std::process::exit(1);
    }

    println!();
    println!(
        "SUCCESS: kernel found {} matches; hash on reported nonce verifies host-side.",
        found_matches[0]
    );
    println!("PASS");
}
