/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor Core Gen 5 (tcgen05) for Blackwell architectures (sm_100+).
//!
//! tcgen05 is Blackwell's tensor core instruction set, replacing WGMMA from Hopper.
//! The key architectural change is **single-thread MMA semantics** - one thread can
//! issue an entire matrix multiply operation.
//!
//! # Key Differences from WGMMA
//!
//! | Aspect | WGMMA (Hopper) | tcgen05 (Blackwell) |
//! |--------|---------------|---------------------|
//! | MMA issue | 128 threads collectively | **1 thread** |
//! | Matrix A/D storage | Registers/SMEM | **Tensor Memory (TMEM)** |
//! | Allocation | Implicit | **Dynamic (tcgen05.alloc)** |
//! | Wait mechanism | `wgmma.wait_group` | **mbarrier.try_wait** |
//!
//! # Tensor Memory (TMEM)
//!
//! TMEM is a new per-SM memory type for tensor operands:
//! - Dynamically allocated at runtime
//! - Allocation unit: 32 columns (range: 32-512)
//! - One WARP must perform allocation (warp-synchronous)
//! - Must be explicitly deallocated before kernel exits
//!
//! # Thread Semantics
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │              Thread Requirements by Operation                        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  tcgen05_mma_ws:    █         (1 thread)                            │
//! │  tcgen05_commit:    █         (1 thread)                            │
//! │  tcgen05_fence:     █         (1 thread)                            │
//! │  tcgen05_alloc:     ████████  (32 threads / 1 warp)                 │
//! │  tcgen05_dealloc:   ████████  (32 threads / 1 warp)                 │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::tcgen05::*;
//! use cuda_device::barrier::*;
//!
//! // 1. Allocate TMEM (warp-synchronous - all 32 threads in warp)
//! if warp_id() == 0 {
//!     tcgen05_alloc(tmem_addr_ptr, 64);  // 64 columns
//! }
//! sync_threads();
//!
//! // 2. Single thread issues MMA
//! if thread_id() == 0 {
//!     tcgen05_fence_before_thread_sync();
//!     tcgen05_mma_ws_f16(d_tmem, a_tmem, a_desc, b_desc, idesc, false);
//!     tcgen05_commit(&mbar);
//! }
//!
//! // 3. All threads wait via mbarrier
//! mbarrier_try_wait(&mbar, 0);
//!
//! // 4. Deallocate TMEM (warp-synchronous)
//! if warp_id() == 0 {
//!     tcgen05_dealloc(tmem_addr, 64);
//! }
//! ```
//!
//! # Hardware Support
//!
//! - **sm_100/sm_100a**: B100, B200 (Data Center)
//! - **sm_120/sm_120a**: RTX 5090 (Consumer)

use core::marker::PhantomData;

// =============================================================================
// Tensor Memory (TMEM) Types
// =============================================================================

/// Handle to allocated Tensor Memory.
///
/// TMEM addresses are 32-bit values returned by `tcgen05_alloc`. The address
/// is written to shared memory by the allocation instruction.
///
/// # Note
///
/// This is NOT a Rust-managed allocation. You must explicitly call
/// `tcgen05_dealloc` before the kernel exits.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct TensorMemoryHandle {
    /// The 32-bit TMEM address
    pub addr: u32,
}

impl TensorMemoryHandle {
    /// Create a handle from a raw TMEM address.
    ///
    /// # Safety
    ///
    /// The address must have been returned by `tcgen05_alloc`.
    #[inline(always)]
    pub const unsafe fn from_raw(addr: u32) -> Self {
        Self { addr }
    }

    /// Get the raw TMEM address.
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.addr
    }
}

// =============================================================================
// TMEM Guard with Typestate (Managed TMEM Lifecycle)
// =============================================================================

/// State marker: TMEM slot not yet allocated.
pub struct TmemUninit;

/// State marker: TMEM is allocated and ready to use.
pub struct TmemReady;

/// State marker: TMEM has been deallocated.
pub struct TmemDeallocated;

/// Newtype for TMEM addresses with type safety.
///
/// Prevents accidentally using raw u32 values as TMEM addresses.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct TmemAddress(u32);

impl TmemAddress {
    /// Get the raw TMEM address value.
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Create from raw value (for internal/advanced use).
    ///
    /// # Safety
    ///
    /// The value must be a valid TMEM address from `tcgen05_alloc`.
    #[inline(always)]
    pub const unsafe fn from_raw(addr: u32) -> Self {
        Self(addr)
    }
}

/// Managed TMEM allocation with typestate lifecycle.
///
/// This type provides compile-time safety for TMEM allocation/deallocation:
/// - Cannot use TMEM before allocation
/// - Cannot allocate twice
/// - Cannot use after deallocation
///
/// # Type Parameters
///
/// - `State`: Lifecycle state (`TmemUninit`, `TmemReady`, `TmemDeallocated`)
/// - `N_COLS`: Number of columns (must be power of 2: 32, 64, 128, 256, 512)
///
/// # Thread Requirements
///
/// TMEM allocation/deallocation is **warp-synchronous**:
/// - All 32 threads in the allocating warp must participate
/// - Other warps do nothing during alloc/dealloc
///
/// The `alloc()` and `dealloc()` methods handle this automatically using the
/// "all threads call, designated warp executes" pattern.
///
/// # Example
///
/// ```rust,ignore
/// // Declare shared memory slot for TMEM address storage
/// static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
///
/// // ALL threads create Uninit handle
/// let tmem = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
///
/// // ALL threads call alloc - only warp 0 actually allocates
/// let tmem = unsafe { tmem.alloc() };  // Returns TmemReady
///
/// // Get the address for use in MMA operations
/// let addr = tmem.address();
///
/// // ... use TMEM in MMA operations ...
///
/// // ALL threads call dealloc - only warp 0 actually deallocates
/// let _dead = unsafe { tmem.dealloc() };  // Returns TmemDeallocated
/// ```
pub struct TmemGuard<State, const N_COLS: u32> {
    /// Pointer to shared memory where the TMEM address is stored
    smem_ptr: *mut u32,
    _state: PhantomData<State>,
}

// Safety: Pointer is only accessed through synchronized operations
unsafe impl<S, const N: u32> Send for TmemGuard<S, N> {}

impl<const N_COLS: u32> TmemGuard<TmemUninit, N_COLS> {
    /// Create an uninitialized TMEM guard from an explicit static declaration.
    ///
    /// The shared memory slot will be used by `tcgen05_alloc` to store the
    /// allocated TMEM address.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
    ///
    /// let tmem = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
    /// ```
    #[inline(always)]
    pub fn from_static(smem_ptr: *mut u32) -> Self {
        TmemGuard {
            smem_ptr,
            _state: PhantomData,
        }
    }

    /// Allocate TMEM columns.
    ///
    /// **All threads in the block should call this.** Only warp 0 performs
    /// the actual allocation; all threads synchronize and receive a `TmemReady` handle.
    ///
    /// This is a convenience wrapper for `alloc_by(0)`.
    ///
    /// # Safety
    ///
    /// - Must be called before any TMEM operations
    /// - All participating threads in the block must call this together
    /// - N_COLS must be a power of 2 in range [32, 512]
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // ALL threads call alloc - only warp 0 actually allocates
    /// let tmem = unsafe { tmem.alloc() };
    /// ```
    #[inline(always)]
    pub unsafe fn alloc(self) -> TmemGuard<TmemReady, N_COLS> {
        unsafe { self.alloc_by(0) }
    }

    /// Allocate TMEM columns with a specific warp performing allocation.
    ///
    /// **All threads in the block should call this.** Only threads in the
    /// designated warp perform the actual allocation; all threads synchronize
    /// and receive a `TmemReady` handle.
    ///
    /// # Parameters
    ///
    /// - `alloc_warp`: Warp ID that performs allocation (all 32 threads in this warp)
    ///
    /// # Safety
    ///
    /// - Must be called before any TMEM operations
    /// - All participating threads in the block must call this together
    /// - `alloc_warp` must be a valid warp ID within the block
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use warp 1 for allocation
    /// let tmem = unsafe { tmem.alloc_by(1) };
    /// ```
    #[inline(always)]
    pub unsafe fn alloc_by(self, alloc_warp: u32) -> TmemGuard<TmemReady, N_COLS> {
        let warp_id = crate::warp::warp_id();

        if warp_id == alloc_warp {
            unsafe { tcgen05_alloc(self.smem_ptr, N_COLS) };
        }

        // All threads synchronize - ensures allocation is complete
        crate::thread::sync_threads();

        TmemGuard {
            smem_ptr: self.smem_ptr,
            _state: PhantomData,
        }
    }
}

impl<const N_COLS: u32> TmemGuard<TmemReady, N_COLS> {
    /// Get the allocated TMEM address.
    ///
    /// This reads the address that `tcgen05_alloc` wrote to shared memory.
    ///
    /// # Note
    ///
    /// All threads in the block can call this - they all read the same value.
    #[inline(always)]
    pub fn address(&self) -> TmemAddress {
        unsafe { TmemAddress(*self.smem_ptr) }
    }

    /// Get the raw TMEM address value.
    ///
    /// Convenience method for passing to intrinsics that take raw u32.
    #[inline(always)]
    pub fn raw_address(&self) -> u32 {
        unsafe { *self.smem_ptr }
    }

    /// Get the number of allocated columns.
    #[inline(always)]
    pub const fn n_cols(&self) -> u32 {
        N_COLS
    }

    /// Deallocate the TMEM.
    ///
    /// **All threads in the block should call this.** Only warp 0 performs
    /// the actual deallocation; all threads synchronize before returning.
    ///
    /// This is a convenience wrapper for `dealloc_by(0)`.
    ///
    /// Consumes the `TmemReady` guard and returns a `TmemDeallocated` guard.
    ///
    /// # CRITICAL
    ///
    /// ALL allocated TMEM MUST be deallocated before the kernel exits.
    /// Failure to do so results in `CUDA_ERROR_TENSOR_MEMORY_LEAK`.
    ///
    /// # Safety
    ///
    /// - All threads must have completed their TMEM operations before calling
    /// - All participating threads in the block must call this together
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // ALL threads call dealloc - only warp 0 actually deallocates
    /// let _dead = unsafe { tmem.dealloc() };
    /// ```
    #[inline(always)]
    pub unsafe fn dealloc(self) -> TmemGuard<TmemDeallocated, N_COLS> {
        unsafe { self.dealloc_by(0) }
    }

    /// Deallocate TMEM with a specific warp performing deallocation.
    ///
    /// **All threads in the block should call this.** Only threads in the
    /// designated warp perform the actual deallocation; all threads synchronize
    /// before returning.
    ///
    /// # Parameters
    ///
    /// - `dealloc_warp`: Warp ID that performs deallocation (must match alloc warp)
    ///
    /// # Safety
    ///
    /// - All threads must have completed their TMEM operations before calling
    /// - All participating threads in the block must call this together
    /// - `dealloc_warp` should typically match the warp that allocated
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use warp 1 for deallocation (should match allocation warp)
    /// let _dead = unsafe { tmem.dealloc_by(1) };
    /// ```
    #[inline(always)]
    pub unsafe fn dealloc_by(self, dealloc_warp: u32) -> TmemGuard<TmemDeallocated, N_COLS> {
        // Ensure all threads are done with TMEM before deallocating
        crate::thread::sync_threads();

        let warp_id = crate::warp::warp_id();

        if warp_id == dealloc_warp {
            unsafe {
                let tmem_addr = *self.smem_ptr;
                tcgen05_dealloc(tmem_addr, N_COLS);
            }
        }

        // All threads synchronize - ensures deallocation is complete
        crate::thread::sync_threads();

        TmemGuard {
            smem_ptr: self.smem_ptr,
            _state: PhantomData,
        }
    }
}

// TmemGuard<TmemDeallocated, _> has NO methods - can't use after dealloc!

// =============================================================================
// TMEM Allocation / Deallocation
// =============================================================================

/// Allocate Tensor Memory columns.
///
/// This is a **warp-synchronous** operation - ALL 32 threads in the warp must
/// execute this instruction together.
///
/// # How It Works
///
/// The hardware allocates TMEM and **writes the allocated address back** to the
/// shared memory location you provide. You then read this address from shared
/// memory to use in subsequent tcgen05 operations:
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────┐
/// │  tcgen05_alloc(&mut SMEM_ADDR, 64)                                  │
/// │                                                                     │
/// │  1. You provide: shared memory pointer (SMEM_ADDR)                  │
/// │  2. Hardware allocates: 64 columns of TMEM                          │
/// │  3. Hardware writes: allocated TMEM address → [SMEM_ADDR]           │
/// │  4. You read: tmem_addr = *SMEM_ADDR  (e.g., 0, 64, 128...)         │
/// └─────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Parameters
///
/// - `dst_smem`: Pointer to shared memory where the **hardware will write**
///   the allocated TMEM address (output parameter)
/// - `n_cols`: Number of columns to allocate (must be power of 2: 32, 64, 128, 256, 512)
///
/// # Blocking Behavior
///
/// This instruction blocks if the requested TMEM is not available, and unblocks
/// when sufficient TMEM becomes available.
///
/// # Example
///
/// ```rust,ignore
/// static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
///
/// // Warp 0 allocates (all 32 threads must participate)
/// if warp_id == 0 {
///     tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 64);
/// }
/// sync_threads();
///
/// // Read the allocated address from shared memory
/// let tmem_addr = *(&raw const TMEM_ADDR as *const u32);
/// // tmem_addr is now the TMEM address (e.g., 0, 64, 128...)
/// ```
///
/// # Safety
///
/// - Must be called by ALL threads in a warp together
/// - `dst_smem` must point to valid shared memory (4-byte aligned)
/// - `n_cols` must be a power of 2 in range [32, 512]
/// - Allocations should not increase in size between calls (hardware constraint)
///
/// # PTX
///
/// ```ptx
/// tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], n_cols;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_alloc(dst_smem: *mut u32, n_cols: u32) {
    let _ = (dst_smem, n_cols);
    unreachable!("tcgen05_alloc called outside CUDA kernel context")
}

/// Deallocate Tensor Memory.
///
/// This is a **warp-synchronous** operation - ALL 32 threads in the warp must
/// execute this instruction together.
///
/// # CRITICAL
///
/// ALL Tensor Memory allocated with `tcgen05_alloc` MUST be deallocated before
/// the kernel exits. Failure to do so results in `CUDA_ERROR_TENSOR_MEMORY_LEAK`.
///
/// # Parameters
///
/// - `tmem_addr`: The TMEM address that was **written to shared memory** by
///   `tcgen05_alloc`. Read it from shared memory before calling this function.
/// - `n_cols`: Number of columns (must match the value used in allocation)
///
/// # Example
///
/// ```rust,ignore
/// // Read the TMEM address that tcgen05_alloc wrote to shared memory
/// let tmem_addr = *(&raw const TMEM_ADDR as *const u32);
///
/// // Deallocate (warp 0, all 32 threads)
/// if warp_id == 0 {
///     tcgen05_dealloc(tmem_addr, 64);
/// }
/// ```
///
/// # Safety
///
/// - Must be called by ALL threads in a warp together
/// - `tmem_addr` must be a valid address from `tcgen05_alloc`
/// - `n_cols` must match the value used in allocation
///
/// # PTX
///
/// ```ptx
/// tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, n_cols;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_dealloc(tmem_addr: u32, n_cols: u32) {
    let _ = (tmem_addr, n_cols);
    unreachable!("tcgen05_dealloc called outside CUDA kernel context")
}

/// Relinquish the right to allocate Tensor Memory.
///
/// After any thread in a CTA executes this, no further `tcgen05_alloc` calls
/// are allowed from that CTA. This is an optional optimization hint.
///
/// This is a **warp-synchronous** operation.
///
/// # PTX
///
/// ```ptx
/// tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
/// ```
#[inline(never)]
pub fn tcgen05_relinquish_alloc_permit() {
    unreachable!("tcgen05_relinquish_alloc_permit called outside CUDA kernel context")
}

// =============================================================================
// Synchronization Primitives
// =============================================================================

/// Fence for ordering BEFORE thread synchronization.
///
/// Ensures all prior tcgen05 async operations complete before any subsequent
/// thread synchronization. Use this when signaling to other threads via relaxed
/// memory operations.
///
/// # When to Use
///
/// ```rust,ignore
/// // Producer thread:
/// tcgen05_mma_ws_f16(...);
/// tcgen05_commit(&mbar);
/// tcgen05_fence_before_thread_sync();  // Ensure MMA done before signal
/// flag = 1;  // Relaxed store to signal consumer
/// ```
///
/// # Note
///
/// If you use `mbarrier_try_wait()` for synchronization, you likely DON'T need
/// this fence - mbarrier provides the ordering guarantee.
///
/// # PTX
///
/// ```ptx
/// tcgen05.fence::before_thread_sync;
/// ```
#[inline(never)]
pub fn tcgen05_fence_before_thread_sync() {
    unreachable!("tcgen05_fence_before_thread_sync called outside CUDA kernel context")
}

/// Fence for ordering AFTER thread synchronization.
///
/// Ensures all subsequent tcgen05 operations wait until after thread
/// synchronization completes. Use this when receiving signals from other threads.
///
/// # When to Use
///
/// ```rust,ignore
/// // Consumer thread:
/// while flag == 0 {}  // Relaxed load spin
/// tcgen05_fence_after_thread_sync();  // Ensure signal received before TMEM access
/// // Now safe to access TMEM results
/// ```
///
/// # Note
///
/// If you use `mbarrier_try_wait()` for synchronization, you likely DON'T need
/// this fence - mbarrier provides the ordering guarantee.
///
/// # PTX
///
/// ```ptx
/// tcgen05.fence::after_thread_sync;
/// ```
#[inline(never)]
pub fn tcgen05_fence_after_thread_sync() {
    unreachable!("tcgen05_fence_after_thread_sync called outside CUDA kernel context")
}

/// Commit pending tcgen05 operations to an mbarrier.
///
/// This signals the mbarrier when all prior tcgen05 async operations complete.
/// Use `mbarrier_try_wait()` to wait for completion.
///
/// This is a **single-thread** operation - only one thread needs to call this.
///
/// # Parameters
///
/// - `mbar`: Pointer to mbarrier object in shared memory
///
/// # Safety
///
/// - `mbar` must point to a valid, initialized mbarrier
///
/// # PTX
///
/// ```ptx
/// tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar];
/// ```
#[inline(never)]
pub unsafe fn tcgen05_commit(mbar: *mut u64) {
    let _ = mbar;
    unreachable!("tcgen05_commit called outside CUDA kernel context")
}

/// Commit tcgen05 work and arrive at the barrier using the `.shared::cluster` variant.
///
/// # Safety
///
/// - `mbar` must be a valid pointer to a barrier in shared memory
/// - Must be called from within a CUDA kernel context on sm_100a+
///
/// PTX:
/// ```ptx
/// tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];
/// ```
#[inline(never)]
pub unsafe fn tcgen05_commit_shared_cluster(mbar: *mut u64) {
    let _ = mbar;
    unreachable!("tcgen05_commit_shared_cluster called outside CUDA kernel context")
}

// NOTE: tcgen05_make_smem_desc and tcgen05_make_smem_desc_strided were removed.
// Use Tcgen05SmemDescriptor::builder() instead (see below).
// =============================================================================
// Instruction Descriptor
// =============================================================================

/// Matrix element types for tcgen05 MMA operations.
///
/// These correspond to the `atype` and `btype` fields in the instruction descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tcgen05ElementType {
    /// FP16 (IEEE half precision)
    F16 = 0,
    /// BF16 (Brain floating point)
    BF16 = 1,
    /// TF32 (TensorFloat-32) - only valid for .kind::tf32
    TF32 = 2,
    /// FP8 E4M3 (4 exponent + 3 mantissa bits). Blackwell-native.
    E4M3 = 3,
    /// FP8 E5M2 (5 exponent + 2 mantissa bits). Blackwell-native.
    E5M2 = 4,
    /// FP6 E2M3 (2 exponent + 3 mantissa bits). Blackwell-native.
    E2M3 = 5,
    /// FP6 E3M2 (3 exponent + 2 mantissa bits). Blackwell-native.
    E3M2 = 6,
    /// FP4 E2M1 (2 exponent + 1 mantissa bit). Blackwell-native.
    E2M1 = 7,
}

impl Tcgen05ElementType {
    /// Returns the idesc atype/btype encoding value for this element type.
    ///
    /// The idesc encoding is kind-dependent per the PTX ISA:
    /// - `kind::f16`: F16=0, BF16=1
    /// - `kind::tf32`: TF32=2
    /// - `kind::f8f6f4`: E4M3=0, E5M2=1, E2M3=3, E3M2=4, E2M1=5
    ///
    /// The enum discriminants are only used for identification; this method
    /// performs the correct mapping so that `build()` always emits the right
    /// bit pattern regardless of the kind family.
    pub const fn idesc_value(self) -> u32 {
        match self {
            Self::F16 => 0,
            Self::BF16 => 1,
            Self::TF32 => 2,
            Self::E4M3 => 0,
            Self::E5M2 => 1,
            Self::E2M3 => 3,
            Self::E3M2 => 4,
            Self::E2M1 => 5,
        }
    }
}

/// Accumulator (output D) data type for tcgen05 MMA operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tcgen05AccumulatorType {
    /// FP16 accumulator (only for .kind::f16)
    F16 = 0,
    /// FP32 accumulator (default for most operations)
    F32 = 1,
}

/// MMA shape for tcgen05.mma.ws operations.
///
/// For `.ws` (warp-sync) with `cta_group::1`:
/// - **f16/bf16**: M ∈ {32, 64, 128}, N ∈ {64, 128, 256}, K = 16
/// - **tf32**: M ∈ {32, 64, 128}, N ∈ {64, 128, 256}, K = 8
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tcgen05MmaShape {
    /// M dimension (rows of A and D). Valid: 32, 64, 128
    pub m: u32,
    /// N dimension (columns of B and D). Valid: 64, 128, 256
    pub n: u32,
}

impl Tcgen05MmaShape {
    /// 32×64 MMA shape (smallest)
    pub const M32_N64: Self = Self { m: 32, n: 64 };
    /// 32×128 MMA shape
    pub const M32_N128: Self = Self { m: 32, n: 128 };
    /// 32×256 MMA shape
    pub const M32_N256: Self = Self { m: 32, n: 256 };
    /// 64×64 MMA shape
    pub const M64_N64: Self = Self { m: 64, n: 64 };
    /// 64×128 MMA shape
    pub const M64_N128: Self = Self { m: 64, n: 128 };
    /// 64×256 MMA shape
    pub const M64_N256: Self = Self { m: 64, n: 256 };
    /// 128×64 MMA shape
    pub const M128_N64: Self = Self { m: 128, n: 64 };
    /// 128×128 MMA shape
    pub const M128_N128: Self = Self { m: 128, n: 128 };
    /// 128×256 MMA shape (largest, best throughput)
    pub const M128_N256: Self = Self { m: 128, n: 256 };
    /// 256×64 UMMA shape (cta_group::2, 128 M-rows per SM)
    pub const M256_N64: Self = Self { m: 256, n: 64 };
    /// 256×128 UMMA shape (cta_group::2, 128 M-rows per SM)
    pub const M256_N128: Self = Self { m: 256, n: 128 };
    /// 256×256 UMMA shape (cta_group::2, 128 M-rows per SM)
    pub const M256_N256: Self = Self { m: 256, n: 256 };
}

/// Instruction descriptor for tcgen05 MMA operations.
///
/// The 32-bit descriptor encodes the matrix shapes, data types, sparsity mode,
/// and other operation details according to the PTX ISA specification.
///
/// # Bit Layout for .kind::tf32, .kind::f16 (from PTX ISA 8.6+)
///
/// ```text
/// Bits  | Field              | Description
/// ------+--------------------+------------------------------------------
/// 0-1   | Sparsity selector  | 0-3 (if sparsity enabled)
/// 2     | Sparsity           | 0 = Dense, 1 = Sparse
/// 3     | Reserved           | 0
/// 4-5   | dtype              | D type: F16=0, F32=1
/// 6     | Reserved           | 0
/// 7-9   | atype              | A type (kind-dependent, see [`Tcgen05ElementType::idesc_value`])
/// 10-12 | btype              | B type (kind-dependent, see [`Tcgen05ElementType::idesc_value`])
/// 13    | Negate A           | 0 = No, 1 = Yes
/// 14    | Negate B           | 0 = No, 1 = Yes
/// 15    | Transpose A        | 0 = No, 1 = Yes
/// 16    | Transpose B        | 0 = No, 1 = Yes
/// 17-22 | N dimension        | N >> 3 (N/8)
/// 23    | Reserved           | 0
/// 24-28 | M dimension        | M >> 4 (M/16)
/// 29    | Reserved           | 0
/// 30-31 | B reuse shift      | 0=none, 1=8, 2=16, 3=32
/// ```
///
/// # Example
///
/// ```rust,ignore
/// // Create descriptor for 128×256 f16 MMA with f32 accumulator
/// let idesc = Tcgen05InstructionDescriptor::builder()
///     .shape(Tcgen05MmaShape::M128_N256)
///     .element_type(Tcgen05ElementType::F16)
///     .accumulator_type(Tcgen05AccumulatorType::F32)
///     .build();
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05InstructionDescriptor {
    raw: u32,
}

impl Tcgen05InstructionDescriptor {
    /// Create a new instruction descriptor builder.
    #[inline(always)]
    pub const fn builder() -> Tcgen05InstructionDescriptorBuilder {
        Tcgen05InstructionDescriptorBuilder::new()
    }

    /// Create descriptor for f16 MMA with default settings.
    ///
    /// Default: 128×256 shape, F16 inputs, F32 accumulator, dense, no transpose.
    #[inline(always)]
    pub const fn new_f16() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::F16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for bf16 MMA with default settings.
    ///
    /// Default: 128×256 shape, BF16 inputs, F32 accumulator, dense, no transpose.
    #[inline(always)]
    pub const fn new_bf16() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::BF16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for tf32 MMA with default settings.
    ///
    /// Default: 128×256 shape, TF32 inputs, F32 accumulator, dense, no transpose.
    #[inline(always)]
    pub const fn new_tf32() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::TF32)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for FP8 E4M3 MMA with default settings.
    ///
    /// Default: 128×256 shape, E4M3 inputs, F32 accumulator.
    #[inline(always)]
    pub const fn new_e4m3() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::E4M3)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for FP8 E5M2 MMA with default settings.
    ///
    /// Default: 128×256 shape, E5M2 inputs, F32 accumulator.
    #[inline(always)]
    pub const fn new_e5m2() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::E5M2)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for FP6 E2M3 MMA with default settings.
    ///
    /// Default: 128×256 shape, E2M3 inputs, F32 accumulator.
    #[inline(always)]
    pub const fn new_e2m3() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::E2M3)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for FP6 E3M2 MMA with default settings.
    ///
    /// Default: 128×256 shape, E3M2 inputs, F32 accumulator.
    #[inline(always)]
    pub const fn new_e3m2() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::E3M2)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor for FP4 E2M1 MMA with default settings.
    ///
    /// Default: 128×256 shape, E2M1 inputs, F32 accumulator.
    #[inline(always)]
    pub const fn new_e2m1() -> Self {
        Self::builder()
            .shape(Tcgen05MmaShape::M128_N256)
            .element_type(Tcgen05ElementType::E2M1)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build()
    }

    /// Create descriptor from raw 32-bit value.
    #[inline(always)]
    pub const fn from_raw(raw: u32) -> Self {
        Self { raw }
    }

    /// Get the raw 32-bit descriptor value.
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.raw
    }
}

/// Builder for `Tcgen05InstructionDescriptor`.
///
/// Allows fluent construction of instruction descriptors with compile-time
/// validation of common configurations.
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05InstructionDescriptorBuilder {
    // Bit 2: Sparsity (0 = dense)
    sparse: bool,
    // Bits 4-5: dtype
    dtype: Tcgen05AccumulatorType,
    // Bits 7-9: atype (and btype, typically same)
    atype: Tcgen05ElementType,
    // Bits 10-12: btype
    btype: Tcgen05ElementType,
    // Bit 13: negate A
    negate_a: bool,
    // Bit 14: negate B
    negate_b: bool,
    // Bit 15: transpose A
    transpose_a: bool,
    // Bit 16: transpose B
    transpose_b: bool,
    // Bits 17-22: N >> 3
    n_dim: u32,
    // Bits 24-28: M >> 4
    m_dim: u32,
    // Bits 30-31: B reuse shift
    b_reuse_shift: u8,
}

impl Tcgen05InstructionDescriptorBuilder {
    /// Create a new builder with default settings.
    ///
    /// Defaults:
    /// - Shape: 128×256
    /// - Element type: F16
    /// - Accumulator: F32
    /// - Dense (not sparse)
    /// - No negation or transposition
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            sparse: false,
            dtype: Tcgen05AccumulatorType::F32,
            atype: Tcgen05ElementType::F16,
            btype: Tcgen05ElementType::F16,
            negate_a: false,
            negate_b: false,
            transpose_a: false,
            transpose_b: false,
            n_dim: 256,
            m_dim: 128,
            b_reuse_shift: 0,
        }
    }

    /// Set the MMA shape (M and N dimensions).
    ///
    /// For `.ws` operations:
    /// - M ∈ {32, 64, 128}
    /// - N ∈ {64, 128, 256}
    #[inline(always)]
    pub const fn shape(mut self, shape: Tcgen05MmaShape) -> Self {
        self.m_dim = shape.m;
        self.n_dim = shape.n;
        self
    }

    /// Set M dimension directly.
    ///
    /// Valid values: 32, 64, 128
    #[inline(always)]
    pub const fn m(mut self, m: u32) -> Self {
        self.m_dim = m;
        self
    }

    /// Set N dimension directly.
    ///
    /// Valid values: 64, 128, 256
    #[inline(always)]
    pub const fn n(mut self, n: u32) -> Self {
        self.n_dim = n;
        self
    }

    /// Set the element type for both A and B matrices.
    #[inline(always)]
    pub const fn element_type(mut self, ty: Tcgen05ElementType) -> Self {
        self.atype = ty;
        self.btype = ty;
        self
    }

    /// Set element type for matrix A only.
    #[inline(always)]
    pub const fn a_type(mut self, ty: Tcgen05ElementType) -> Self {
        self.atype = ty;
        self
    }

    /// Set element type for matrix B only.
    #[inline(always)]
    pub const fn b_type(mut self, ty: Tcgen05ElementType) -> Self {
        self.btype = ty;
        self
    }

    /// Set the accumulator (output D) type.
    #[inline(always)]
    pub const fn accumulator_type(mut self, ty: Tcgen05AccumulatorType) -> Self {
        self.dtype = ty;
        self
    }

    /// Enable sparse matrix A.
    ///
    /// When enabled, K dimension is doubled (K=32 for f16, K=16 for tf32).
    #[inline(always)]
    pub const fn sparse(mut self, enable: bool) -> Self {
        self.sparse = enable;
        self
    }

    /// Enable negation of matrix A values.
    #[inline(always)]
    pub const fn negate_a(mut self, enable: bool) -> Self {
        self.negate_a = enable;
        self
    }

    /// Enable negation of matrix B values.
    #[inline(always)]
    pub const fn negate_b(mut self, enable: bool) -> Self {
        self.negate_b = enable;
        self
    }

    /// Enable transposition of matrix A.
    #[inline(always)]
    pub const fn transpose_a(mut self, enable: bool) -> Self {
        self.transpose_a = enable;
        self
    }

    /// Enable transposition of matrix B.
    #[inline(always)]
    pub const fn transpose_b(mut self, enable: bool) -> Self {
        self.transpose_b = enable;
        self
    }

    /// Set maximum B matrix reuse shift.
    ///
    /// - 0: No shift (default)
    /// - 1: Maximum shift of 8
    /// - 2: Maximum shift of 16
    /// - 3: Maximum shift of 32
    #[inline(always)]
    pub const fn b_reuse_shift(mut self, shift: u8) -> Self {
        self.b_reuse_shift = shift;
        self
    }

    /// Build the instruction descriptor.
    ///
    /// Encodes all settings into the 32-bit descriptor format.
    #[inline(always)]
    pub const fn build(self) -> Tcgen05InstructionDescriptor {
        let mut raw: u32 = 0;

        // Bits 0-1: Sparsity selector (always 0 for now)
        // raw |= 0;

        // Bit 2: Sparsity
        if self.sparse {
            raw |= 1 << 2;
        }

        // Bit 3: Reserved (0)

        // Bits 4-5: dtype (accumulator type)
        raw |= (self.dtype as u32) << 4;

        // Bit 6: Reserved (0)

        // Bits 7-9: atype (3 bits) — encoding is kind-dependent
        raw |= self.atype.idesc_value() << 7;

        // Bits 10-12: btype (3 bits) — encoding is kind-dependent
        raw |= self.btype.idesc_value() << 10;

        // Bit 13: Negate A
        if self.negate_a {
            raw |= 1 << 13;
        }

        // Bit 14: Negate B
        if self.negate_b {
            raw |= 1 << 14;
        }

        // Bit 15: Transpose A
        if self.transpose_a {
            raw |= 1 << 15;
        }

        // Bit 16: Transpose B
        if self.transpose_b {
            raw |= 1 << 16;
        }

        // Bits 17-22: N >> 3 (6 bits)
        raw |= ((self.n_dim >> 3) & 0x3F) << 17;

        // Bit 23: Reserved (0)

        // Bits 24-28: M >> 4 (5 bits)
        raw |= ((self.m_dim >> 4) & 0x1F) << 24;

        // Bit 29: Reserved (0)

        // Bits 30-31: B reuse shift (2 bits)
        raw |= ((self.b_reuse_shift as u32) & 0x3) << 30;

        Tcgen05InstructionDescriptor { raw }
    }
}

impl Default for Tcgen05InstructionDescriptorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SMEM Descriptor
// =============================================================================

/// Swizzle modes for tcgen05 SMEM descriptors.
///
/// Swizzling reorders data in shared memory to avoid bank conflicts during
/// tensor core operations. The mode determines the XOR pattern applied to
/// addresses.
///
/// # Encoding (bits 61-63 of SMEM descriptor)
///
/// | Mode | Value | Pattern Start |
/// |------|-------|---------------|
/// | None | 0 | - |
/// | 128B + 32B atomicity | 1 | 1024B boundary |
/// | 128B | 2 | 1024B boundary |
/// | 64B | 4 | 512B boundary |
/// | 32B | 6 | 256B boundary |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Tcgen05SwizzleMode {
    /// No swizzling
    #[default]
    None = 0,
    /// 128-byte swizzle with 32-byte atomicity (best for tensor cores)
    Swizzle128B32BAtom = 1,
    /// 128-byte swizzle (most common for tensor cores)
    Swizzle128B = 2,
    /// 64-byte swizzle
    Swizzle64B = 4,
    /// 32-byte swizzle
    Swizzle32B = 6,
}

/// SMEM descriptor for tcgen05 MMA operations (64-bit).
///
/// Describes a matrix tile in shared memory for tcgen05.mma instructions.
/// The descriptor encodes the start address, stride dimensions, and swizzle mode.
///
/// # Bit Layout (from PTX ISA)
///
/// ```text
/// Bits  | Field
/// ------+----------------------------------------------------------
/// 0-13  | (start_address >> 4) & 0x3FFF
/// 14-15 | Reserved
/// 16-29 | (leading_dim_bytes >> 4) & 0x3FFF
/// 30-31 | Reserved
/// 32-45 | (stride_bytes >> 4) & 0x3FFF
/// 46-48 | Fixed: 0b001
/// 49-51 | Base offset (for non-aligned swizzle patterns)
/// 52    | Leading dim mode: 0=relative, 1=absolute
/// 53-60 | Fixed: 0x00
/// 61-63 | Swizzle mode
/// ```
///
/// # Example
///
/// ```rust,ignore
/// // For 128×16 f16 matrix A in K-major tiled layout:
/// let desc = Tcgen05SmemDescriptor::for_k_major(
///     smem_a_ptr as u64,
///     128,  // M dimension
///     16,   // K dimension
///     2,    // f16 = 2 bytes per element
///     Tcgen05SwizzleMode::None,
/// );
///
/// // Use in MMA
/// tcgen05_mma_f16(d_tmem, desc.raw(), b_desc, idesc, true);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05SmemDescriptor {
    raw: u64,
}

impl Tcgen05SmemDescriptor {
    /// Create a new SMEM descriptor builder.
    #[inline(always)]
    pub const fn builder() -> Tcgen05SmemDescriptorBuilder {
        Tcgen05SmemDescriptorBuilder::new()
    }

    /// Build descriptor for K-major tiled layout (matrix A).
    ///
    /// K-major layout is required for the A matrix in tcgen05.mma:
    /// - Matrix divided into 8×8 tiles
    /// - Tiles arranged column-major
    /// - Within each tile: row-major (K dimension contiguous)
    ///
    /// # Parameters
    ///
    /// - `smem_addr`: Shared memory address (must be 16-byte aligned)
    /// - `m`: M dimension (rows of the matrix)
    /// - `k`: K dimension (columns of the matrix)
    /// - `elem_bytes`: Element size in bytes (2 for f16/bf16, 4 for tf32/f32)
    /// - `swizzle`: Swizzle mode for bank conflict avoidance
    ///
    /// # Layout Computation
    ///
    /// For an M×K matrix with 8×8 tiles:
    /// - **SBO** (stride byte offset) = 8 × 8 × elem_bytes = 64 × elem_bytes
    /// - **LBO** (leading dim byte offset) = (M/8) × 8 × 8 × elem_bytes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // 128×16 f16 matrix
    /// let desc = Tcgen05SmemDescriptor::for_k_major(
    ///     smem_ptr as u64, 128, 16, 2, Tcgen05SwizzleMode::None
    /// );
    /// // SBO = 8*8*2 = 128 bytes
    /// // LBO = (128/8)*8*8*2 = 16*64*2 = 2048 bytes
    /// ```
    #[inline(always)]
    pub const fn for_k_major(
        smem_addr: u64,
        m: usize,
        k: usize,
        elem_bytes: usize,
        swizzle: Tcgen05SwizzleMode,
    ) -> Self {
        let _ = k; // K determines number of tiles horizontally, but not SBO/LBO directly

        // SBO = one 8×8 tile size in bytes
        let sbo = (8 * 8 * elem_bytes) as u32;

        // LBO = column of tiles = (M/8) tiles × 64 elements × elem_bytes
        let lbo = ((m / 8) * 8 * 8 * elem_bytes) as u32;

        Self::builder()
            .address(smem_addr)
            .leading_dim_bytes(lbo)
            .stride_bytes(sbo)
            .swizzle(swizzle)
            .build()
    }

    /// Build descriptor for MN-major tiled layout (matrix B).
    ///
    /// MN-major layout is required for the B matrix in tcgen05.mma:
    /// - Matrix divided into 8×8 tiles
    /// - Tiles arranged column-major
    /// - Within each tile: column-major (MN dimension contiguous)
    ///
    /// # Parameters
    ///
    /// - `smem_addr`: Shared memory address (must be 16-byte aligned)
    /// - `n`: N dimension (rows of B, which is K×N transposed view)
    /// - `k`: K dimension
    /// - `elem_bytes`: Element size in bytes
    /// - `swizzle`: Swizzle mode
    #[inline(always)]
    pub const fn for_mn_major(
        smem_addr: u64,
        n: usize,
        k: usize,
        elem_bytes: usize,
        swizzle: Tcgen05SwizzleMode,
    ) -> Self {
        let _ = k;

        let sbo = (8 * 8 * elem_bytes) as u32;
        let lbo = ((n / 8) * 8 * 8 * elem_bytes) as u32;

        Self::builder()
            .address(smem_addr)
            .leading_dim_bytes(lbo)
            .stride_bytes(sbo)
            .swizzle(swizzle)
            .build()
    }

    /// Build descriptor from explicit byte offsets.
    ///
    /// Use this when you need full control over the descriptor parameters.
    ///
    /// # Parameters
    ///
    /// - `smem_addr`: Shared memory address (16-byte aligned)
    /// - `leading_dim_bytes`: Leading dimension stride in bytes (LBO)
    /// - `stride_bytes`: Stride dimension offset in bytes (SBO)
    /// - `swizzle`: Swizzle mode
    #[inline(always)]
    pub const fn from_bytes(
        smem_addr: u64,
        leading_dim_bytes: u32,
        stride_bytes: u32,
        swizzle: Tcgen05SwizzleMode,
    ) -> Self {
        Self::builder()
            .address(smem_addr)
            .leading_dim_bytes(leading_dim_bytes)
            .stride_bytes(stride_bytes)
            .swizzle(swizzle)
            .build()
    }

    /// Create descriptor from raw 64-bit value.
    #[inline(always)]
    pub const fn from_raw(raw: u64) -> Self {
        Self { raw }
    }

    /// Get the raw 64-bit descriptor value.
    #[inline(always)]
    pub const fn raw(self) -> u64 {
        self.raw
    }
}

/// Builder for `Tcgen05SmemDescriptor`.
///
/// Provides fluent API for constructing SMEM descriptors with compile-time
/// validation of common configurations.
#[derive(Clone, Copy, Debug)]
pub struct Tcgen05SmemDescriptorBuilder {
    address: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
    base_offset: u8,
    leading_dim_absolute: bool,
    swizzle: Tcgen05SwizzleMode,
}

impl Tcgen05SmemDescriptorBuilder {
    /// Create a new builder with default settings.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            address: 0,
            leading_dim_bytes: 0,
            stride_bytes: 0,
            base_offset: 0,
            leading_dim_absolute: false,
            swizzle: Tcgen05SwizzleMode::None,
        }
    }

    /// Set the shared memory start address.
    ///
    /// Must be 16-byte aligned.
    #[inline(always)]
    pub const fn address(mut self, addr: u64) -> Self {
        self.address = addr;
        self
    }

    /// Set the leading dimension byte offset (LBO).
    ///
    /// This is the stride to move to the next "column" of tiles.
    /// Must be 16-byte aligned.
    #[inline(always)]
    pub const fn leading_dim_bytes(mut self, lbo: u32) -> Self {
        self.leading_dim_bytes = lbo;
        self
    }

    /// Set the stride dimension byte offset (SBO).
    ///
    /// This is typically the size of one tile (8×8 × element_size).
    /// Must be 16-byte aligned.
    #[inline(always)]
    pub const fn stride_bytes(mut self, sbo: u32) -> Self {
        self.stride_bytes = sbo;
        self
    }

    /// Set the swizzle mode.
    #[inline(always)]
    pub const fn swizzle(mut self, mode: Tcgen05SwizzleMode) -> Self {
        self.swizzle = mode;
        self
    }

    /// Set the base offset for non-aligned swizzle patterns.
    ///
    /// Only needed when the swizzle pattern doesn't start at a natural boundary.
    /// Formula: `base_offset = (pattern_start_addr >> 7) & 0x7`
    #[inline(always)]
    pub const fn base_offset(mut self, offset: u8) -> Self {
        self.base_offset = offset;
        self
    }

    /// Use absolute addressing for leading dimension.
    ///
    /// When true, leading_dim_bytes is an absolute address rather than
    /// a relative offset. Only supported on sm_103a+.
    #[inline(always)]
    pub const fn leading_dim_absolute(mut self, absolute: bool) -> Self {
        self.leading_dim_absolute = absolute;
        self
    }

    /// Build the SMEM descriptor.
    ///
    /// Encodes all settings into the 64-bit descriptor format.
    #[inline(always)]
    pub const fn build(self) -> Tcgen05SmemDescriptor {
        let mut raw: u64 = 0;

        // Bits 0-13: (address >> 4) & 0x3FFF
        raw |= (self.address >> 4) & 0x3FFF;

        // Bits 16-29: (leading_dim_bytes >> 4) & 0x3FFF
        raw |= (((self.leading_dim_bytes >> 4) & 0x3FFF) as u64) << 16;

        // Bits 32-45: (stride_bytes >> 4) & 0x3FFF
        raw |= (((self.stride_bytes >> 4) & 0x3FFF) as u64) << 32;

        // Bits 46-48: Fixed = 0b001
        raw |= 1u64 << 46;

        // Bits 49-51: Base offset (3 bits)
        raw |= ((self.base_offset & 0x7) as u64) << 49;

        // Bit 52: Leading dim mode (0 = relative, 1 = absolute)
        if self.leading_dim_absolute {
            raw |= 1u64 << 52;
        }

        // Bits 53-60: Fixed = 0 (already zero)

        // Bits 61-63: Swizzle mode (3 bits)
        raw |= (self.swizzle as u64) << 61;

        Tcgen05SmemDescriptor { raw }
    }
}

impl Default for Tcgen05SmemDescriptorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Collector Buffer Usage
// =============================================================================

/// Collector buffer usage hints for matrix B caching.
///
/// Collector buffers are internal to the Tensor Core (not addressable).
/// They cache matrix B data to avoid repeated shared memory reads when
/// using the same B matrix with different A matrices.
///
/// # Example
///
/// ```rust,ignore
/// // First MMA: Load B into collector b0
/// tcgen05_mma_ws_f16_with_collector(..., CollectorUsage::Fill(0));
///
/// // Subsequent MMAs: Reuse B from collector
/// tcgen05_mma_ws_f16_with_collector(..., CollectorUsage::Use(0));
/// tcgen05_mma_ws_f16_with_collector(..., CollectorUsage::Use(0));
///
/// // Last use: Discard after using
/// tcgen05_mma_ws_f16_with_collector(..., CollectorUsage::LastUse(0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectorUsage {
    /// Read B from SMEM and cache in buffer N
    Fill(u8),
    /// Use B from buffer N (must have been filled)
    Use(u8),
    /// Use B from buffer N, then discard
    LastUse(u8),
    /// Discard buffer N without using
    Discard(u8),
}

// =============================================================================
// tcgen05 MMA Instructions
// =============================================================================

/// tcgen05 Matrix Multiply-Accumulate with f16 inputs.
///
/// Performs D = A × B + D (or D = A × B if `enable_d` is false).
///
/// **SINGLE-THREAD SEMANTICS**: Unlike WGMMA which requires 128 threads,
/// ONE thread issues the entire MMA operation!
///
/// # Matrix Convention
///
/// - **A**: Left multiplicand (M×K) in Tensor Memory
/// - **B**: Right multiplicand (K×N) in Shared Memory
/// - **D**: Destination AND accumulator (M×N) in Tensor Memory
///
/// # Parameters
///
/// - `d_tmem`: TMEM address for D matrix (destination/accumulator)
/// - `a_tmem`: TMEM address for A matrix
/// - `a_desc`: SMEM descriptor for A (from `Tcgen05SmemDescriptor::builder()`)
/// - `b_desc`: SMEM descriptor for B (from `Tcgen05SmemDescriptor::builder()`)
/// - `idesc`: Instruction descriptor (from `Tcgen05InstructionDescriptor`)
/// - `enable_d`: true = D += A×B, false = D = A×B
///
/// # Safety
///
/// - TMEM addresses must have been allocated with `tcgen05_alloc`
/// - Descriptors must point to valid shared memory
///
/// # PTX
///
/// ```ptx
/// tcgen05.mma.ws.cta_group::1.kind::f16 [d-tmem], [a-tmem], a-desc, b-desc, idesc, enable-d;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_f16(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_ws_f16 called outside CUDA kernel context")
}

/// tcgen05 Matrix Multiply-Accumulate with f16 inputs (non-warp-specialized form).
/// tcgen05 MMA with FP6 E2M3 inputs.
///
/// FP6 E2M3 (2-bit exponent, 3-bit mantissa) offers higher precision than
/// E3M2 at the cost of reduced dynamic range.
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_e2m3(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_ws_e2m3 called outside CUDA kernel context")
}

/// tcgen05 MMA with FP6 E3M2 inputs.
///
/// FP6 E3M2 (3-bit exponent, 2-bit mantissa) offers wider dynamic range
/// than E2M3 at the cost of reduced precision.
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_e3m2(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_ws_e3m2 called outside CUDA kernel context")
}

/// tcgen05 MMA with FP4 E2M1 inputs.
///
/// FP4 E2M1 (2-bit exponent, 1-bit mantissa) is the lowest precision
/// Blackwell-native floating-point format, offering the highest throughput.
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_e2m1(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_ws_e2m1 called outside CUDA kernel context")
}

/// tcgen05 MMA with f16 inputs (non-warp-specialized form).
///
/// Performs D = A × B + D (or D = A × B if `enable_d` is false).
///
/// **SINGLE-THREAD SEMANTICS**: ONE thread issues the entire MMA operation.
///
/// This form consumes **both A and B from shared memory descriptors**, unlike
/// `tcgen05_mma_ws_f16` which commonly consumes A from TMEM.
///
/// # Parameters
///
/// - `d_tmem`: TMEM address for D matrix (destination/accumulator)
/// - `a_desc`: SMEM descriptor for A (from `Tcgen05SmemDescriptor::builder()`)
/// - `b_desc`: SMEM descriptor for B (from `Tcgen05SmemDescriptor::builder()`)
/// - `idesc`: Instruction descriptor
/// - `enable_d`: true = D += A×B, false = D = A×B
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
///
/// ```ptx
/// tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, {0,0,0,0}, enable-d;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_mma_f16(d_tmem: u32, a_desc: u64, b_desc: u64, idesc: u32, enable_d: bool) {
    let _ = (d_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_f16 called outside CUDA kernel context")
}

/// tcgen05 MMA with bf16 inputs.
///
/// Same as `tcgen05_mma_ws_f16` but uses bfloat16 input format.
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_bf16(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_ws_bf16 called outside CUDA kernel context")
}

/// tcgen05 MMA with tf32 inputs.
///
/// TensorFloat-32 provides better precision than f16/bf16 while maintaining
/// high tensor core throughput.
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_tf32(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_ws_tf32 called outside CUDA kernel context")
}

/// tcgen05 MMA with FP8 E4M3 inputs.
///
/// FP8 E4M3 (4-bit exponent, 3-bit mantissa) is a Blackwell-native format
/// offering higher throughput than FP16 at reduced precision.
///
/// # PTX
///
/// ```ptx
/// tcgen05.mma.ws.cta_group::1.kind::f16 [d-tmem], [a-tmem], a-desc, b-desc, idesc, enable-d;
/// ```
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_e4m3(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_ws_e4m3 called outside CUDA kernel context")
}

/// tcgen05 MMA with FP8 E5M2 inputs.
///
/// FP8 E5M2 (5-bit exponent, 2-bit mantissa) offers a wider dynamic range
/// than E4M3 at the cost of reduced precision.
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_e5m2(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_ws_e5m2 called outside CUDA kernel context")
}

/// tcgen05 MMA with f16 inputs and collector buffer hint.
///
/// Use collector buffers to cache matrix B when reusing the same B matrix
/// across multiple MMA operations.
///
/// # Safety
///
/// - All descriptors must be valid and properly initialized
/// - Must be called from within a CUDA kernel context on sm_100a+
#[inline(never)]
pub unsafe fn tcgen05_mma_ws_f16_with_collector(
    d_tmem: u32,
    a_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
    collector: CollectorUsage,
) {
    let _ = (d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d, collector);
    unreachable!("tcgen05_mma_ws_f16_with_collector called outside CUDA kernel context")
}

// =============================================================================
// TMEM Load/Store (for results)
// =============================================================================

/// Copy data from shared memory to Tensor Memory.
///
/// # Parameters
///
/// - `tmem_addr`: Destination TMEM address
/// - `smem_desc`: Source shared memory descriptor
///
/// # Safety
///
/// - `smem_desc` must be a valid shared memory descriptor
/// - `tmem_addr` must be a valid TMEM address
/// - Must be called from within a CUDA kernel context on sm_100a+
///
/// # PTX
///
/// ```ptx
/// tcgen05.cp.cta_group::1.128x256b [tmem_addr], smem_desc;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_cp_smem_to_tmem(tmem_addr: u32, smem_desc: u64) {
    let _ = (tmem_addr, smem_desc);
    unreachable!("tcgen05_cp_smem_to_tmem called outside CUDA kernel context")
}

// NOTE: The following deprecated intrinsics were removed:
// - tcgen05_st_tmem_to_smem (wrong approach - no direct TMEM→SMEM instruction)
// - tcgen05_st_tmem_to_smem_offset (wrong approach)
// - tcgen05_ld_16x256b_x4/x8/x16/x32 (wrong - stored to SMEM instead of returning registers)
// - tcgen05_ld_32x32b_x64 (wrong - stored to SMEM instead of returning registers)
//
// Use tcgen05_ld_16x256b_pure or tcgen05_ld_16x256b_x8_pure instead.
// These return values in registers, allowing proper epilogue processing:
//   1. Load from TMEM → registers
//   2. Convert f32 → bf16/f16 in registers
//   3. Store to SMEM via stmatrix

// =============================================================================
// Pure TMEM Load (Returns Registers - No SMEM Store)
// =============================================================================

use crate::cusimd::CuSimd;

/// Result of `tcgen05_ld_16x256b_x8_pure` - 32 f32 values per thread.
///
/// This is a type alias for `CuSimd<f32, 32>` which provides:
/// - Array-style indexing: `regs[i]` for runtime access
/// - Const generic access: `regs.get::<0>()` for compile-time access
/// - Shorthand accessors: `regs.x()`, `regs.y()`, etc.
///
/// # Usage
///
/// ```rust,ignore
/// let regs: TmemF32x32 = tcgen05_ld_16x256b_x8_pure(tmem_addr);
/// tcgen05_load_wait();
///
/// // Access via index
/// let val0 = regs[0];
/// let val1 = regs[1];
///
/// // Convert and pack for stmatrix
/// let p0 = cvt_f32x2_bf16x2(regs[0], regs[1]);
/// let p1 = cvt_f32x2_bf16x2(regs[2], regs[3]);
/// // ...
/// stmatrix_m8n8_x4_trans(smem_ptr, p0, p1, p2, p3);
/// ```
pub type TmemF32x32 = CuSimd<f32, 32>;

/// Load 32 f32 values from TMEM to registers (PURE - no SMEM store).
///
/// This is the **correct** way to load from TMEM for epilog processing.
/// Unlike `tcgen05_ld_16x256b_x8`, this function:
/// - Returns values in registers (as a CuSimd)
/// - Does NOT store to shared memory
/// - Allows subsequent in-register operations (convert, pack, stmatrix)
///
/// # Returns
///
/// `TmemF32x32` (`CuSimd<f32, 32>`) containing 32 f32 values per thread.
/// For a warp: 32 threads × 32 values = 1024 f32 = 16 rows × 64 columns.
///
/// # PTX
///
/// ```ptx
/// tcgen05.ld.sync.aligned.16x256b.x8.b32
///   {%r0,%r1,...,%r31}, [%tmem_addr];
/// ```
///
/// # Safety
///
/// - `tmem_addr` must be valid (from tcgen05_alloc)
/// - Must be called by ALL 32 threads in a warp together (warp-synchronous)
/// - Must call `tcgen05_load_wait()` before accessing returned values
#[inline(never)]
pub unsafe fn tcgen05_ld_16x256b_x8_pure(tmem_addr: u32) -> TmemF32x32 {
    let _ = tmem_addr;
    unreachable!("tcgen05_ld_16x256b_x8_pure called outside CUDA kernel context")
}

// =============================================================================
// Base LDTM (16dp256bit without .x multiplier)
// =============================================================================

/// Result of `tcgen05_ld_16x256b_pure` (base LDTM) - 4 f32 values per thread.
///
/// This is a type alias for `CuSimd<f32, 4>` which provides:
/// - Array-style indexing: `regs[i]` for runtime access
/// - Const generic access: `regs.get::<0>()` for compile-time access
/// - Shorthand accessors: `regs.x()`, `regs.y()`, `regs.z()`, `regs.w()`
///
/// # Thread Data Layout
///
/// For base LDTM.16dp256bit with 32dp (data parallelism = 32 threads):
/// - Each thread receives 4 f32 values = 16 bytes = 128 bits
/// - Total per warp: 32 threads × 4 values = 128 f32 values
/// - This corresponds to an 8×16 matrix tile (8 rows × 16 columns)
///
/// # Relationship to stmatrix.x2
///
/// The data layout from base LDTM matches perfectly with `stmatrix.m8n8.x2`:
/// - 128 bf16 (after conversion) = 2 matrices × 8×8 = 128 elements ✓
/// - Thread ownership aligns: each thread owns data for specific positions
///
/// # Usage
///
/// ```rust,ignore
/// let regs: TmemF32x4 = tcgen05_ld_16x256b_pure(tmem_addr);
/// tcgen05_load_wait();
///
/// // Access via index or shorthand
/// let p0 = cvt_f32x2_bf16x2(regs[0], regs[1]);
/// let p1 = cvt_f32x2_bf16x2(regs[2], regs[3]);
/// // Or: cvt_f32x2_bf16x2(regs.x(), regs.y())
///
/// // Store via stmatrix.x2 (non-trans)
/// stmatrix_m8n8_x2(smem_ptr, p0, p1);
/// ```
pub type TmemF32x4 = CuSimd<f32, 4>;

/// Load 4 f32 values from TMEM to registers (base LDTM, no multiplier).
///
/// This is the **base** TMEM load instruction without the .x8 multiplier.
/// It loads fewer values but with a simpler, more predictable layout.
///
/// # SASS Instruction
///
/// `LDTM.16dp256bit` (no .x8 suffix)
///
/// # Returns
///
/// `TmemF32x4` (`CuSimd<f32, 4>`) containing 4 f32 values per thread.
/// For a warp: 32 threads × 4 values = 128 f32 values.
///
/// # Data Layout
///
/// The base LDTM has a simpler layout than .x8:
/// - Each thread gets 4 consecutive f32 values from its assigned position
/// - Layout matches well with stmatrix.m8n8.x2 for 8×16 tile stores
///
/// # PTX
///
/// ```ptx
/// tcgen05.ld.sync.aligned.16x256b.b32
///   {%r0, %r1, %r2, %r3}, [%tmem_addr];
/// ```
///
/// # Safety
///
/// - `tmem_addr` must be valid (from tcgen05_alloc)
/// - Must be called by ALL 32 threads in a warp together (warp-synchronous)
/// - Must call `tcgen05_load_wait()` before accessing returned values
#[inline(never)]
pub unsafe fn tcgen05_ld_16x256b_pure(tmem_addr: u32) -> TmemF32x4 {
    let _ = tmem_addr;
    unreachable!("tcgen05_ld_16x256b_pure called outside CUDA kernel context")
}

// =============================================================================
// PTX Type Conversion Intrinsics
// =============================================================================

/// Convert two f32 values to packed bf16x2 using PTX.
///
/// This is the **proper** way to convert f32 → bf16 for stmatrix.
/// Uses the PTX `cvt.rn.bf16x2.f32` instruction which:
/// - Converts with round-to-nearest-even
/// - Packs two bf16 into one u32 register
///
/// # PTX
///
/// ```ptx
/// cvt.rn.bf16x2.f32 %result, %b, %a;
/// // Result: lower 16 bits = bf16(a), upper 16 bits = bf16(b)
/// ```
///
/// # Arguments
///
/// - `a`: First f32 value (goes to lower 16 bits)
/// - `b`: Second f32 value (goes to upper 16 bits)
///
/// # Returns
///
/// Packed u32 with two bf16 values: `(bf16(b) << 16) | bf16(a)`
///
/// Lane placement: the first argument (`a`) fills bits `[15:0]` and the
/// second argument (`b`) fills bits `[31:16]`, even though the PTX
/// operand list prints `b` first. This is the same first-arg-low
/// convention as [`cvt_f16x2_f32`](crate::convert::cvt_f16x2_f32),
/// which differs only in its destination element type (f16, not bf16)
/// and its `lo`/`hi` argument naming.
///
/// # Example
///
/// ```rust,ignore
/// let regs = tcgen05_ld_16x256b_x8_pure(addr);
/// tcgen05_load_wait();
///
/// // Convert pairs of f32 to packed bf16
/// let p0 = cvt_f32x2_bf16x2(regs[0], regs[1]);
/// let p1 = cvt_f32x2_bf16x2(regs[2], regs[3]);
/// let p2 = cvt_f32x2_bf16x2(regs[4], regs[5]);
/// let p3 = cvt_f32x2_bf16x2(regs[6], regs[7]);
///
/// // Store via stmatrix
/// stmatrix_m8n8_x4_trans(smem_ptr, p0, p1, p2, p3);
/// ```
#[inline(never)]
pub fn cvt_f32x2_bf16x2(a: f32, b: f32) -> u32 {
    let _ = (a, b);
    unreachable!("cvt_f32x2_bf16x2 called outside CUDA kernel context")
}

// =============================================================================
// TMEM Load/Store Synchronization
// =============================================================================

/// Wait for tensor memory loads to complete.
///
/// This is a **critical synchronization barrier** that must be called after
/// `tcgen05_ld_*` instructions before accessing the loaded data.
///
/// The `tcgen05.ld` instructions are **asynchronous** - they initiate the load
/// but don't wait for completion. Without this wait, you may read stale data!
///
/// # Usage
///
/// ```rust,ignore
/// // Load from TMEM to registers (asynchronous)
/// let regs = tcgen05_ld_16x256b_x8_pure(tmem_addr);
///
/// // MUST wait before accessing the loaded data!
/// tcgen05_load_wait();
///
/// // Now safe to read from smem_ptr
/// ```
///
/// # Safety
///
/// - Must be called by ALL 32 threads in a warp together (warp-synchronous)
///
/// # PTX
///
/// ```ptx
/// tcgen05.wait::ld.sync.aligned;
/// ```
#[inline(never)]
pub fn tcgen05_load_wait() {
    unreachable!("tcgen05_load_wait called outside CUDA kernel context")
}

/// Wait for tensor memory stores to complete.
///
/// Similar to `tcgen05_load_wait`, but for store operations (`tcgen05.st`).
///
/// # PTX
///
/// ```ptx
/// tcgen05.wait::st.sync.aligned;
/// ```
#[inline(never)]
pub fn tcgen05_store_wait() {
    unreachable!("tcgen05_store_wait called outside CUDA kernel context")
}

// =============================================================================
// Matrix Store Instructions
// =============================================================================

/// Store matrix tile from registers to shared memory using stmatrix.
///
/// This is the warp-cooperative matrix store instruction that properly handles
/// the matrix layout when storing MMA results to shared memory.
///
/// # Parameters
///
/// - `smem_ptr`: Destination in shared memory (16-byte aligned)
/// - `r0`, `r1`, `r2`, `r3`: Four u32 values, each containing two packed b16 elements
///
/// # Shape
///
/// `stmatrix.sync.aligned.m8n8.x4.shared.b16`:
/// - Stores four 8×8 matrix tiles
/// - Uses 4 registers per thread (packed as 16-bit elements)
/// - Cooperatively distributes across 32 threads
///
/// # Usage
///
/// Convert loaded values to packed b16 pairs before calling this function.
/// Each u32 is treated as bits; this function does not convert f32 to b16.
///
/// # Address lanes
///
/// ```text
/// lanes  0..7  -> rows 0..7 of matrix 0
/// lanes  8..15 -> rows 0..7 of matrix 1
/// lanes 16..23 -> rows 0..7 of matrix 2
/// lanes 24..31 -> rows 0..7 of matrix 3
/// ```
///
/// # Safety
///
/// - `smem_ptr` must be valid shared memory (16-byte aligned)
/// - Must be called by ALL 32 threads in a warp together
/// - Callers must use the appropriate fence or barrier before a dependent memory access
#[inline(never)]
pub unsafe fn stmatrix_m8n8_x4(smem_ptr: *mut u8, r0: u32, r1: u32, r2: u32, r3: u32) {
    let _ = (smem_ptr, r0, r1, r2, r3);
    unreachable!("stmatrix_m8n8_x4 called outside CUDA kernel context")
}

/// Store matrix tiles from registers to shared memory in column-major order.
///
/// This is the warp-cooperative matrix store instruction with the critical `.trans`
/// modifier, which selects column-major storage.
///
/// # PTX Instruction
///
/// `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [ptr], {r0, r1, r2, r3};`
///
/// # Parameters
///
/// - `smem_ptr`: Destination in shared memory (16-byte aligned)
/// - `r0`, `r1`, `r2`, `r3`: Four u32 values, each containing 2 packed bf16 elements
///
/// # Shape
///
/// - Stores four 8×8 matrix tiles (256 elements total per warp call)
/// - Each register contains 2 packed bf16 values
/// - `.trans` stores each matrix in column-major order
///
/// # Usage for 64×64 extraction
///
/// After `tcgen05_ld_16x256b_x8_pure` (32 f32 registers per thread):
/// 1. Convert f32 → bf16 (cvt.rn.bf16.f32)
/// 2. Pack bf16 pairs → u32 (mov.b32)
/// 3. Call `stmatrix_m8n8_x4_trans` 4 times with groups of 4 registers
///
/// Each warp stores 16 rows (1024 elements), so 4 calls × 256 = 1024 ✓
///
/// # Address lanes
///
/// The address-lane rule is unchanged by `.trans`: consecutive groups of
/// eight lanes provide rows 0-7 for matrices 0, 1, 2, and 3 respectively.
///
/// # Safety
///
/// - `smem_ptr` must be valid shared memory (16-byte aligned)
/// - Must be called by ALL 32 threads in a warp together (warp-synchronous)
/// - Registers must contain properly packed bf16 pairs
/// - Callers must use the appropriate fence or barrier before a dependent memory access
#[inline(never)]
pub unsafe fn stmatrix_m8n8_x4_trans(smem_ptr: *mut u8, r0: u32, r1: u32, r2: u32, r3: u32) {
    let _ = (smem_ptr, r0, r1, r2, r3);
    unreachable!("stmatrix_m8n8_x4_trans called outside CUDA kernel context")
}

/// Store 2 matrix tiles (8×8 each = 16 columns) from registers to shared memory.
///
/// This is the warp-cooperative matrix store instruction (NON-transpose version).
/// It stores **16 columns** (2 × 8×8 tiles) per call WITHOUT transpose.
///
/// # PTX Instruction
///
/// `stmatrix.sync.aligned.m8n8.x2.shared.b16 [ptr], {r0, r1};`
///
/// # SASS Instruction
///
/// `STSM.16.MT88.2 [Rx], Ry;`
///
/// # Parameters
///
/// - `smem_ptr`: Base address in shared memory (16-byte aligned)
/// - `r0`, `r1`: Two u32 values, each containing 2 packed bf16 elements
///
/// # Shape
///
/// - Stores two 8×8 matrix tiles (128 bf16 elements total per warp call)
/// - Each thread provides 2 registers = 4 bf16 values
/// - 32 threads × 4 bf16 = 128 bf16 ✓
///
/// # Thread-to-Position Mapping (Non-Trans)
///
/// Without `.trans`, each group of 4 threads stores one row:
/// ```text
/// Row-Major Output (8 rows × 16 columns):
/// ┌────────────────────────────────────┐
/// │ row 0: Threads 0-3's data         │
/// │ row 1: Threads 4-7's data         │
/// │ row 2: Threads 8-11's data        │
/// │ ...                                │
/// │ row 7: Threads 28-31's data       │
/// └────────────────────────────────────┘
/// ```
///
/// Per thread within a row (4 threads per row):
/// - Thread (lane % 4) == 0: columns 0-1 of matrix 0, columns 0-1 of matrix 1
/// - Thread (lane % 4) == 1: columns 2-3 of matrix 0, columns 2-3 of matrix 1
/// - Thread (lane % 4) == 2: columns 4-5 of matrix 0, columns 4-5 of matrix 1
/// - Thread (lane % 4) == 3: columns 6-7 of matrix 0, columns 6-7 of matrix 1
///
/// # Address Requirement
///
/// Only lanes 0-15 provide addresses. Lanes 0-7 provide rows 0-7 of the first
/// matrix; lanes 8-15 provide rows 0-7 of the second matrix. All 32 lanes still
/// execute the instruction and provide fragment registers.
///
/// ```text
/// lanes  0..7  -> base + (lane_id % 8) * row_stride
/// lanes  8..15 -> base + matrix_stride + (lane_id % 8) * row_stride
/// lanes 16..31 -> address operand ignored; fragment registers still used
/// ```
///
/// # Usage with Base LDTM
///
/// ```rust,ignore
/// // Load 4 f32 from TMEM using base LDTM
/// let regs: TmemF32x4 = tcgen05_ld_16x256b_pure(tmem_addr);
/// tcgen05_load_wait();
///
/// // Convert f32 pairs to packed bf16x2
/// let p0 = cvt_f32x2_bf16x2(regs[0], regs[1]);
/// let p1 = cvt_f32x2_bf16x2(regs[2], regs[3]);
///
/// let row_in_tile = lane_id % 8;
/// let matrix_offset = if lane_id < 8 { 0 } else { matrix_stride };
/// let addr = base_smem + matrix_offset + row_in_tile * row_stride + col_offset;
///
/// // Store via stmatrix.x2 (non-trans)
/// stmatrix_m8n8_x2(addr as *mut u8, p0, p1);
/// ```
///
/// # Safety
///
/// - `smem_ptr` must be valid shared memory (16-byte aligned)
/// - Must be called by ALL 32 threads in a warp together (warp-synchronous)
/// - Registers must contain properly packed bf16 pairs
/// - Callers must use the appropriate fence or barrier before a dependent memory access
#[inline(never)]
pub unsafe fn stmatrix_m8n8_x2(smem_ptr: *mut u8, r0: u32, r1: u32) {
    let _ = (smem_ptr, r0, r1);
    unreachable!("stmatrix_m8n8_x2 called outside CUDA kernel context")
}

/// Store two 8×8 bf16 matrices to shared memory in column-major order.
///
/// This is the `.trans` version that matches cuBLAS epilog: `STSM.16.MT88.2`.
///
/// # Generated PTX
///
/// `stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [ptr], {r0, r1};`
///
/// # Thread-to-Address Mapping (TRANSPOSE)
///
/// The address-lane rule is unchanged by `.trans`: lanes 0-7 provide rows
/// 0-7 of the first matrix and lanes 8-15 provide rows 0-7 of the second.
/// All 32 lanes still execute the instruction and provide fragment registers.
///
/// # Safety
///
/// - `smem_ptr` must be valid shared memory (16-byte aligned)
/// - Must be called by ALL 32 threads in a warp together
/// - Registers must contain properly packed bf16 pairs from LDTM
/// - Callers must use the appropriate fence or barrier before a dependent memory access
#[inline(never)]
pub unsafe fn stmatrix_m8n8_x2_trans(smem_ptr: *mut u8, r0: u32, r1: u32) {
    let _ = (smem_ptr, r0, r1);
    unreachable!("stmatrix_m8n8_x2_trans called outside CUDA kernel context")
}

// =============================================================================
// Type Conversion Helpers (for epilog)
// =============================================================================

/// Convert f32 to bf16 using simple truncation.
///
/// BF16 format: 1 sign + 8 exponent + 7 mantissa = 16 bits
/// This simply takes the upper 16 bits of the f32 representation.
///
/// Note: This is a simple truncation, not round-to-nearest-even.
/// For most ML workloads, the difference is negligible.
///
/// # Example
///
/// ```rust,ignore
/// let f32_val: f32 = 3.14159;
/// let bf16_bits: u16 = f32_to_bf16(f32_val);
/// ```
#[inline(always)]
pub fn f32_to_bf16(val: f32) -> u16 {
    // BF16 is just the upper 16 bits of f32
    // f32: 1 sign + 8 exp + 23 mantissa
    // bf16: 1 sign + 8 exp + 7 mantissa (truncate lower 16 bits)
    (val.to_bits() >> 16) as u16
}

/// Convert f32 to bf16 with round-to-nearest-even.
///
/// This provides slightly more accurate conversion than simple truncation
/// by rounding the truncated mantissa bits.
#[inline(always)]
pub fn f32_to_bf16_rne(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round to nearest even: add 0x7FFF + bit 16 (round up if bit 16 is 1)
    let round_bit = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + round_bit);
    (rounded >> 16) as u16
}

/// Pack two bf16 values (as u16) into a single u32.
///
/// The first bf16 goes in the lower 16 bits, the second in the upper 16 bits.
/// This is used to prepare data for `stmatrix` which expects packed 16-bit values.
///
/// # Example
///
/// ```rust,ignore
/// let bf16_0 = f32_to_bf16(f32_vals[0]);
/// let bf16_1 = f32_to_bf16(f32_vals[1]);
/// let packed = pack_bf16_pair(bf16_0, bf16_1);
/// // packed is now ready for stmatrix_m8n8_x4_trans
/// ```
#[inline(always)]
pub fn pack_bf16_pair(lo: u16, hi: u16) -> u32 {
    (lo as u32) | ((hi as u32) << 16)
}

/// Pack two f16 values (as u16) into a single u32.
///
/// The first f16 goes in the lower 16 bits, the second in the upper 16 bits.
#[inline(always)]
pub fn pack_f16_pair(lo: u16, hi: u16) -> u32 {
    (lo as u32) | ((hi as u32) << 16)
}

/// Convert two consecutive f32 values to bf16 and pack into u32.
///
/// This is a convenience function that combines conversion and packing
/// for the common epilog pattern.
///
/// # Example
///
/// ```rust,ignore
/// let f32_pair: [f32; 2] = [a, b];
/// let packed: u32 = f32_pair_to_packed_bf16(f32_pair[0], f32_pair[1]);
/// ```
#[inline(always)]
pub fn f32_pair_to_packed_bf16(a: f32, b: f32) -> u32 {
    pack_bf16_pair(f32_to_bf16(a), f32_to_bf16(b))
}

// =============================================================================
// CTA Pair (cta_group::2) Variants
// =============================================================================
//
// These are identical to their cta_group::1 counterparts but emit
// `cta_group::2` in PTX. Per the PTX ISA, ALL tcgen05 instructions in a
// kernel must use the same cta_group value — mixing is undefined.
//
// CTA pairs place two CTAs on adjacent SMs (a TPC). The pair cooperates
// on larger MMA tiles (e.g., 256×128 instead of 128×128) and shares
// barriers via DSMEM.

/// Allocate Tensor Memory for a CTA pair (`cta_group::2`).
///
/// Identical to [`tcgen05_alloc`] but emits `cta_group::2`.
/// Both CTAs in the pair must call this together.
///
/// # Safety
///
/// Must be called from a CUDA kernel context on sm_100a+.
/// `dst_smem` must be a valid shared-memory pointer. Both CTAs in the
/// pair must execute this cooperatively.
///
/// # PTX
///
/// ```ptx
/// tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], n_cols;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_alloc_cg2(dst_smem: *mut u32, n_cols: u32) {
    let _ = (dst_smem, n_cols);
    unreachable!("tcgen05_alloc_cg2 called outside CUDA kernel context")
}

/// Deallocate Tensor Memory for a CTA pair (`cta_group::2`).
///
/// Identical to [`tcgen05_dealloc`] but emits `cta_group::2`.
///
/// # Safety
///
/// Must be called from a CUDA kernel context on sm_100a+.
/// `tmem_addr` must be a previously allocated TMEM address.
///
/// # PTX
///
/// ```ptx
/// tcgen05.dealloc.cta_group::2.sync.aligned.b32 tmem_addr, n_cols;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_dealloc_cg2(tmem_addr: u32, n_cols: u32) {
    let _ = (tmem_addr, n_cols);
    unreachable!("tcgen05_dealloc_cg2 called outside CUDA kernel context")
}

/// Relinquish TMEM allocation permit for a CTA pair (`cta_group::2`).
///
/// # PTX
///
/// ```ptx
/// tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
/// ```
#[inline(never)]
pub fn tcgen05_relinquish_alloc_permit_cg2() {
    unreachable!("tcgen05_relinquish_alloc_permit_cg2 called outside CUDA kernel context")
}

/// tcgen05 MMA with f16 inputs for a CTA pair (`cta_group::2`).
///
/// Both A and B are consumed from SMEM descriptors. The two CTAs in the
/// pair cooperate on a larger tile (e.g., 256×128 with each SM contributing
/// half the rows).
///
/// # Safety
///
/// Must be called from a CUDA kernel context on sm_100a+.
/// Descriptors must reference valid SMEM regions. `d_tmem` must be a
/// valid TMEM address. Both CTAs must participate.
///
/// # PTX
///
/// ```ptx
/// tcgen05.mma.cta_group::2.kind::f16 [d], a_desc, b_desc, idesc, {0,0,0,0}, enable_d;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_mma_f16_cg2(
    d_tmem: u32,
    a_desc: u64,
    b_desc: u64,
    idesc: u32,
    enable_d: bool,
) {
    let _ = (d_tmem, a_desc, b_desc, idesc, enable_d);
    unreachable!("tcgen05_mma_f16_cg2 called outside CUDA kernel context")
}

/// Commit pending CTA-pair tcgen05 operations to an mbarrier (`cta_group::2`).
///
/// # Safety
///
/// Must be called from a CUDA kernel context on sm_100a+.
/// `mbar` must point to a valid mbarrier object in shared memory.
///
/// # PTX
///
/// ```ptx
/// tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [mbar];
/// ```
#[inline(never)]
pub unsafe fn tcgen05_commit_cg2(mbar: *mut u64) {
    let _ = mbar;
    unreachable!("tcgen05_commit_cg2 called outside CUDA kernel context")
}

/// Commit CTA-pair tcgen05 operations via `.shared::cluster` (`cta_group::2`).
///
/// # Safety
///
/// Must be called from a CUDA kernel context on sm_100a+.
/// `mbar` must point to a valid mbarrier in cluster-accessible shared memory.
///
/// # PTX
///
/// ```ptx
/// tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [mbar];
/// ```
#[inline(never)]
pub unsafe fn tcgen05_commit_shared_cluster_cg2(mbar: *mut u64) {
    let _ = mbar;
    unreachable!("tcgen05_commit_shared_cluster_cg2 called outside CUDA kernel context")
}

/// Commit CTA-pair tcgen05 operations with multicast to multiple CTAs' barriers.
///
/// When a cooperative MMA (`cta_group::2`) completes, this signals the
/// mbarrier in every CTA whose bit is set in `cta_mask`. This is how
/// Mojo's `mma_arrive_multicast` works — the MMA warp signals both its
/// own and its partner CTA's barrier in one instruction.
///
/// # Safety
///
/// Must be called from a CUDA kernel context on sm_100a+.
/// `mbar` must point to a valid mbarrier in cluster-accessible shared memory.
/// `cta_mask` must only reference valid CTA ranks within the cluster.
///
/// # Parameters
///
/// - `mbar`: pointer to mbarrier in shared memory
/// - `cta_mask`: 16-bit mask where bit N = signal CTA with cluster rank N
///
/// # PTX
///
/// ```ptx
/// tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [mbar], ctaMask;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_commit_multicast_cg2(mbar: *mut u64, cta_mask: u16) {
    let _ = (mbar, cta_mask);
    unreachable!("tcgen05_commit_multicast_cg2 called outside CUDA kernel context")
}

/// Copy SMEM to TMEM for a CTA pair (`cta_group::2`).
///
/// # Safety
///
/// Must be called from a CUDA kernel context on sm_100a+.
/// `tmem_addr` must be a valid TMEM address and `smem_desc` must be a
/// valid SMEM descriptor. Both CTAs must participate.
///
/// # PTX
///
/// ```ptx
/// tcgen05.cp.cta_group::2.128x256b [tmem_addr], smem_desc;
/// ```
#[inline(never)]
pub unsafe fn tcgen05_cp_smem_to_tmem_cg2(tmem_addr: u32, smem_desc: u64) {
    let _ = (tmem_addr, smem_desc);
    unreachable!("tcgen05_cp_smem_to_tmem_cg2 called outside CUDA kernel context")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `idesc_value()` returns the correct PTX ISA encoding
    /// for each element type under its corresponding kind.
    ///
    /// Reference: PTX ISA Table 45 (Instruction descriptor for
    /// .kind::tf32, .kind::f16, .kind::f8f6f4 and .kind::i8)
    #[test]
    fn idesc_value_matches_ptx_isa_encoding() {
        // kind::f16 family
        assert_eq!(Tcgen05ElementType::F16.idesc_value(), 0);
        assert_eq!(Tcgen05ElementType::BF16.idesc_value(), 1);

        // kind::tf32
        assert_eq!(Tcgen05ElementType::TF32.idesc_value(), 2);

        // kind::f8f6f4 family — note: encoding differs from enum discriminant!
        assert_eq!(Tcgen05ElementType::E4M3.idesc_value(), 0);
        assert_eq!(Tcgen05ElementType::E5M2.idesc_value(), 1);
        assert_eq!(Tcgen05ElementType::E2M3.idesc_value(), 3);
        assert_eq!(Tcgen05ElementType::E3M2.idesc_value(), 4);
        assert_eq!(Tcgen05ElementType::E2M1.idesc_value(), 5);
    }

    /// The enum discriminants must NOT equal the idesc encoding for
    /// f8f6f4 types. This is the core invariant of approach B: the
    /// discriminant is an identifier, `idesc_value()` is the encoding.
    #[test]
    fn f8f6f4_discriminants_differ_from_idesc_encoding() {
        // E4M3: discriminant=3, but idesc_value=0
        assert_ne!(
            Tcgen05ElementType::E4M3 as u8,
            Tcgen05ElementType::E4M3.idesc_value() as u8
        );
        // E5M2: discriminant=4, but idesc_value=1
        assert_ne!(
            Tcgen05ElementType::E5M2 as u8,
            Tcgen05ElementType::E5M2.idesc_value() as u8
        );
        // E2M3: discriminant=5, but idesc_value=3
        assert_ne!(
            Tcgen05ElementType::E2M3 as u8,
            Tcgen05ElementType::E2M3.idesc_value() as u8
        );
    }

    /// Verify that `build()` correctly encodes atype/btype bits (7–12)
    /// using `idesc_value()` rather than the raw enum discriminant.
    #[test]
    fn build_encodes_atype_btype_via_idesc_value() {
        // E4M3: idesc_value=0 → bits 7-9 should be 0
        let idesc_e4m3 = Tcgen05InstructionDescriptor::builder()
            .element_type(Tcgen05ElementType::E4M3)
            .build();
        let atype_bits = (idesc_e4m3.raw() >> 7) & 0x7;
        let btype_bits = (idesc_e4m3.raw() >> 10) & 0x7;
        assert_eq!(atype_bits, 0, "E4M3 atype must be 0 (f8f6f4 encoding)");
        assert_eq!(btype_bits, 0, "E4M3 btype must be 0 (f8f6f4 encoding)");

        // E5M2: idesc_value=1 → bits 7-9 should be 1
        let idesc_e5m2 = Tcgen05InstructionDescriptor::builder()
            .element_type(Tcgen05ElementType::E5M2)
            .build();
        let atype_bits = (idesc_e5m2.raw() >> 7) & 0x7;
        assert_eq!(atype_bits, 1, "E5M2 atype must be 1 (f8f6f4 encoding)");

        // E2M1: idesc_value=5 → bits 7-9 should be 5
        let idesc_e2m1 = Tcgen05InstructionDescriptor::builder()
            .element_type(Tcgen05ElementType::E2M1)
            .build();
        let atype_bits = (idesc_e2m1.raw() >> 7) & 0x7;
        assert_eq!(atype_bits, 5, "E2M1 atype must be 5 (f8f6f4 encoding)");

        // E3M2: idesc_value=4
        let idesc_e3m2 = Tcgen05InstructionDescriptor::builder()
            .element_type(Tcgen05ElementType::E3M2)
            .build();
        let atype_bits = (idesc_e3m2.raw() >> 7) & 0x7;
        assert_eq!(atype_bits, 4, "E3M2 atype must be 4 (f8f6f4 encoding)");

        // E2M3: idesc_value=3
        let idesc_e2m3 = Tcgen05InstructionDescriptor::builder()
            .element_type(Tcgen05ElementType::E2M3)
            .build();
        let atype_bits = (idesc_e2m3.raw() >> 7) & 0x7;
        assert_eq!(atype_bits, 3, "E2M3 atype must be 3 (f8f6f4 encoding)");
    }

    /// Verify that f16/bf16/tf32 still use their correct idesc encodings.
    #[test]
    fn build_preserves_f16_bf16_tf32_encoding() {
        // F16: idesc_value=0
        let idesc_f16 = Tcgen05InstructionDescriptor::new_f16();
        let atype_bits = (idesc_f16.raw() >> 7) & 0x7;
        assert_eq!(atype_bits, 0, "F16 atype must be 0");

        // BF16: idesc_value=1
        let idesc_bf16 = Tcgen05InstructionDescriptor::new_bf16();
        let atype_bits = (idesc_bf16.raw() >> 7) & 0x7;
        assert_eq!(atype_bits, 1, "BF16 atype must be 1");

        // TF32: idesc_value=2
        let idesc_tf32 = Tcgen05InstructionDescriptor::new_tf32();
        let atype_bits = (idesc_tf32.raw() >> 7) & 0x7;
        assert_eq!(atype_bits, 2, "TF32 atype must be 2");
    }

    /// Verify that `a_type()` and `b_type()` can set independent types
    /// and the idesc encodes them separately.
    #[test]
    fn build_supports_mixed_atype_btype() {
        // E4M3 A × E5M2 B (common mixed-precision pattern)
        let idesc = Tcgen05InstructionDescriptor::builder()
            .a_type(Tcgen05ElementType::E4M3)
            .b_type(Tcgen05ElementType::E5M2)
            .build();
        let atype_bits = (idesc.raw() >> 7) & 0x7;
        let btype_bits = (idesc.raw() >> 10) & 0x7;
        assert_eq!(atype_bits, 0, "E4M3 atype must be 0");
        assert_eq!(btype_bits, 1, "E5M2 btype must be 1");
    }

    /// Verify default constructors for new FP4/FP6 types produce valid
    /// descriptors with correct element type encoding.
    #[test]
    fn default_constructors_for_fp4_fp6() {
        let idesc_e2m1 = Tcgen05InstructionDescriptor::new_e2m1();
        assert_eq!((idesc_e2m1.raw() >> 7) & 0x7, 5, "E2M1");

        let idesc_e3m2 = Tcgen05InstructionDescriptor::new_e3m2();
        assert_eq!((idesc_e3m2.raw() >> 7) & 0x7, 4, "E3M2");

        let idesc_e2m3 = Tcgen05InstructionDescriptor::new_e2m3();
        assert_eq!((idesc_e2m3.raw() >> 7) & 0x7, 3, "E2M3");
    }

    /// Verify that the old FP8 constructors (new_e4m3, new_e5m2) now
    /// produce correct f8f6f4-encoded descriptors (Bug 3 regression test).
    #[test]
    fn fp8_constructors_use_f8f6f4_encoding() {
        // Before the fix, E4M3 used discriminant=3 as the idesc value,
        // which in f8f6f4 kind means E2M3. After the fix, it should use 0.
        let idesc_e4m3 = Tcgen05InstructionDescriptor::new_e4m3();
        let atype = (idesc_e4m3.raw() >> 7) & 0x7;
        assert_eq!(
            atype, 0,
            "E4M3 must encode as 0 in f8f6f4, not 3 (old discriminant)"
        );

        let idesc_e5m2 = Tcgen05InstructionDescriptor::new_e5m2();
        let atype = (idesc_e5m2.raw() >> 7) & 0x7;
        assert_eq!(
            atype, 1,
            "E5M2 must encode as 1 in f8f6f4, not 4 (old discriminant)"
        );
    }
}
