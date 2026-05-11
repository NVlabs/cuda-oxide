/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Async barrier primitives for Hopper+ architectures.
//!
//! Hardware barriers (`mbarrier`) enable efficient synchronization for async
//! operations like TMA copies. Unlike `sync_threads()`, barriers can track
//! transaction completion asynchronously.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  TMA (Hardware DMA)                                     │
//! │       │                                                 │
//! │       │ cp.async.bulk.tensor...                         │
//! │       ▼                                                 │
//! │  ┌─────────────┐    mbarrier.arrive    ┌─────────────┐  │
//! │  │   Shared    │◄─────────────────────►│   Barrier   │  │
//! │  │   Memory    │                       │  (64-bit)   │  │
//! │  └─────────────┘    mbarrier.wait      └─────────────┘  │
//! │       ▲                                      ▲          │
//! │       │                                      │          │
//! │  Threads read data              Threads check completion│
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, SharedArray};
//! use cuda_device::barrier::{Barrier, mbarrier_init, mbarrier_arrive, mbarrier_wait};
//!
//! #[kernel]
//! pub fn async_copy_kernel(...) {
//!     // Barrier in shared memory
//!     static mut BAR: Barrier = Barrier::UNINIT;
//!
//!     // Thread 0 initializes (expected arrivals = block size)
//!     if thread::threadIdx_x() == 0 {
//!         unsafe { mbarrier_init(&mut BAR, 128); }
//!     }
//!     thread::sync_threads();
//!
//!     // ... TMA copy would arrive at barrier when done ...
//!
//!     // All threads arrive and wait
//!     let token = unsafe { mbarrier_arrive(&BAR) };
//!     unsafe { mbarrier_wait(&BAR, token); }
//!
//!     // Barrier phase complete - safe to read data
//! }
//! ```
//!
//! # Hardware Support
//!
//! - **sm_80+ (Ampere)**: Basic mbarrier support
//! - **sm_90+ (Hopper)**: Full TMA integration with transaction tracking
//! - **sm_120 (Blackwell)**: Enhanced barrier operations

// =============================================================================
// Barrier Type
// =============================================================================

/// Hardware barrier for async synchronization.
///
/// This is a 64-bit value stored in shared memory that tracks:
/// - Expected arrival count
/// - Current arrival count
/// - Phase bit (for reuse across iterations)
///
/// # Memory Layout (conceptual)
///
/// ```text
/// [63:48] Phase + Hardware State
/// [47:32] Expected Arrival Count
/// [31:0]  Current Arrival Count
/// ```
///
/// # Safety
///
/// - Must be declared as `static mut` in shared memory
/// - Must be initialized before use with `mbarrier_init`
/// - All threads that will arrive must be accounted for in expected count
#[repr(C, align(8))]
#[derive(Copy, Clone)]
pub struct Barrier {
    /// Internal 64-bit state managed by hardware
    _state: u64,
}

impl Barrier {
    /// Uninitialized barrier constant for `static mut` declarations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BAR: Barrier = Barrier::UNINIT;
    /// ```
    pub const UNINIT: Self = Self { _state: 0 };
}

// =============================================================================
// Barrier Initialization
// =============================================================================

/// Initialize a barrier with the expected number of arrivals.
///
/// Must be called by exactly ONE thread before any other barrier operations.
/// Typically thread 0 initializes, then `sync_threads()` before use.
///
/// # Parameters
///
/// - `bar`: Pointer to barrier in shared memory
/// - `expected_count`: Number of threads/transactions that will arrive
///
/// # Safety
///
/// - `bar` must point to valid shared memory
/// - Must be called exactly once per barrier phase
/// - Other threads must not access barrier until init completes
///
/// # Example
///
/// ```rust,ignore
/// static mut BAR: Barrier = Barrier::UNINIT;
///
/// if thread::threadIdx_x() == 0 {
///     unsafe { mbarrier_init(&mut BAR, 128); }  // 128 threads will arrive
/// }
/// thread::sync_threads();
/// ```
///
/// # PTX
///
/// ```ptx
/// mbarrier.init.shared.b64 [addr], count;
/// ```
#[inline(never)]
pub unsafe fn mbarrier_init(bar: *mut Barrier, expected_count: u32) {
    let _ = (bar, expected_count);
    // Lowered to: call void @llvm.nvvm.mbarrier.init.shared(ptr %bar, i32 %count)
    unreachable!("mbarrier_init called outside CUDA kernel context")
}

// =============================================================================
// Barrier Arrive Operations
// =============================================================================

/// Arrive at barrier, signaling this thread's participation.
///
/// Returns a token (phase) that must be passed to `mbarrier_wait`.
/// The barrier completes when all expected arrivals have occurred.
///
/// # Parameters
///
/// - `bar`: Pointer to barrier in shared memory
///
/// # Returns
///
/// A 64-bit token representing the current barrier phase.
///
/// # Safety
///
/// - `bar` must be initialized
/// - Must be paired with a `mbarrier_wait` call
///
/// # Example
///
/// ```rust,ignore
/// let token = unsafe { mbarrier_arrive(&BAR) };
/// // ... do independent work ...
/// unsafe { mbarrier_wait(&BAR, token); }
/// ```
///
/// # PTX
///
/// ```ptx
/// mbarrier.arrive.shared.b64 token, [addr];
/// ```
#[inline(never)]
pub unsafe fn mbarrier_arrive(bar: *const Barrier) -> u64 {
    let _ = bar;
    // Lowered to: call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr %bar)
    unreachable!("mbarrier_arrive called outside CUDA kernel context")
}

/// Arrive at barrier without returning a token (fire-and-forget).
///
/// Use this when the thread will not wait on this barrier.
/// For producer threads that signal completion but don't need to wait.
///
/// # Safety
///
/// - `bar` must be initialized
/// - Only use when this thread will NOT call `mbarrier_wait`
///
/// # PTX
///
/// ```ptx
/// mbarrier.arrive.noComplete.shared.b64 _, [addr];
/// ```
#[inline(never)]
pub unsafe fn mbarrier_arrive_no_complete(bar: *const Barrier) {
    let _ = bar;
    // Lowered to: call void @llvm.nvvm.mbarrier.arrive.noComplete.shared(ptr %bar)
    unreachable!("mbarrier_arrive_no_complete called outside CUDA kernel context")
}

// =============================================================================
// Barrier Wait Operations
// =============================================================================

/// Test if barrier phase is complete (non-blocking).
///
/// Returns `true` if all expected arrivals have occurred for the given phase.
/// Use this for polling-style waits or to check completion without blocking.
///
/// # Parameters
///
/// - `bar`: Pointer to barrier in shared memory
/// - `token`: Phase token from `mbarrier_arrive`
///
/// # Returns
///
/// `true` if the barrier phase is complete, `false` otherwise.
///
/// # Safety
///
/// - `bar` must be initialized
/// - `token` must be from a matching `mbarrier_arrive` call
///
/// # PTX
///
/// ```ptx
/// mbarrier.test_wait.shared.b64 pred, [addr], token;
/// ```
#[inline(never)]
pub unsafe fn mbarrier_test_wait(bar: *const Barrier, token: u64) -> bool {
    let _ = (bar, token);
    // Lowered to: call i1 @llvm.nvvm.mbarrier.test_wait.shared(ptr %bar, i64 %token)
    unreachable!("mbarrier_test_wait called outside CUDA kernel context")
}

/// Try to wait for barrier phase to complete (with scheduling hints).
///
/// Similar to `mbarrier_test_wait` but provides better scheduling hints to the
/// hardware. The thread may be suspended briefly while waiting, improving
/// efficiency in spin-wait loops.
///
/// **This is the preferred wait operation for TMA synchronization.** nvcc uses
/// this instruction for barrier waits in TMA copy patterns.
///
/// # Parameters
///
/// - `bar`: Pointer to barrier in shared memory
/// - `token`: Phase token from `mbarrier_arrive`
///
/// # Returns
///
/// `true` if the barrier phase is complete, `false` otherwise.
///
/// # Safety
///
/// - `bar` must be initialized
/// - `token` must be from a matching `mbarrier_arrive` call
///
/// # Example
///
/// ```rust,ignore
/// let token = unsafe { mbarrier_arrive(&BAR) };
/// // Wait with scheduling hints
/// while !unsafe { mbarrier_try_wait(&BAR, token) } {
///     // Hardware may suspend thread briefly
/// }
/// ```
///
/// # PTX
///
/// ```ptx
/// mbarrier.try_wait.shared.b64 pred, [addr], token;
/// ```
#[inline(never)]
pub unsafe fn mbarrier_try_wait(bar: *const Barrier, token: u64) -> bool {
    let _ = (bar, token);
    // Lowered to inline PTX: mbarrier.try_wait.shared.b64 p, [%bar], %token;
    unreachable!("mbarrier_try_wait called outside CUDA kernel context")
}

/// Try to wait for barrier completion using parity-based wait.
///
/// This variant is used in patterns where the producer arrives
/// via operations that do **not** return a token (e.g. `tcgen05.commit`).
///
/// # PTX
///
/// ```ptx
/// mbarrier.try_wait.parity.shared::cta.b64 pred, [addr], parity;
/// ```
///
/// # Safety
///
/// - `bar` must be a valid pointer to a barrier in shared memory
/// - Must be called from within a CUDA kernel context
#[inline(never)]
pub unsafe fn mbarrier_try_wait_parity(bar: *const Barrier, parity: u32) -> bool {
    let _ = (bar, parity);
    unreachable!("mbarrier_try_wait_parity called outside CUDA kernel context")
}

/// Wait for barrier phase to complete (blocking).
///
/// Blocks until all expected arrivals have occurred for the given phase.
/// This is implemented as a loop over `mbarrier_test_wait`.
///
/// # Parameters
///
/// - `bar`: Pointer to barrier in shared memory
/// - `token`: Phase token from `mbarrier_arrive`
///
/// # Safety
///
/// - `bar` must be initialized
/// - `token` must be from a matching `mbarrier_arrive` call
/// - Calling thread must have arrived at the barrier
///
/// # Example
///
/// ```rust,ignore
/// let token = unsafe { mbarrier_arrive(&BAR) };
/// unsafe { mbarrier_wait(&BAR, token); }
/// // Barrier complete - safe to access synchronized data
/// ```
#[inline(always)]
pub unsafe fn mbarrier_wait(bar: *const Barrier, token: u64) {
    // Implemented as a spin loop on try_wait.
    // try_wait provides scheduling hints to the hardware, making it
    // more efficient than test_wait for actual waiting.
    while !unsafe { mbarrier_try_wait(bar, token) } {
        // spin
    }
}

// =============================================================================
// Transaction-Based Barrier Operations (for TMA)
// =============================================================================

/// Arrive at barrier expecting additional transaction bytes.
///
/// Used with TMA copies to track async completion. The barrier won't
/// complete until BOTH:
/// 1. All expected arrivals occur
/// 2. All expected transaction bytes are transferred
///
/// # Parameters
///
/// - `bar`: Pointer to barrier in shared memory
/// - `tx_count`: Number of transactions (typically 1 for single TMA op)
/// - `bytes`: Number of bytes expected from async transfers
///
/// # Returns
///
/// A 64-bit token representing the current barrier phase.
///
/// # Safety
///
/// - `bar` must be initialized with expected arrival count
/// - `bytes` must match the actual TMA transfer size
///
/// # Example (with TMA)
///
/// ```rust,ignore
/// // Thread 0 initiates TMA copy and arrives with transaction
/// if thread::threadIdx_x() == 0 {
///     tma::copy_async_2d(...);  // TMA will transfer `tile_bytes`
///     let token = unsafe { mbarrier_arrive_expect_tx(&BAR, 1, tile_bytes) };
/// }
///
/// // Other threads just arrive
/// let token = unsafe { mbarrier_arrive(&BAR) };
///
/// // All wait for TMA + arrivals
/// unsafe { mbarrier_wait(&BAR, token); }
/// ```
///
/// # PTX
///
/// ```ptx
/// mbarrier.arrive.expect_tx.shared.b64 token, [addr], bytes;
/// ```
#[inline(never)]
pub unsafe fn mbarrier_arrive_expect_tx(bar: *const Barrier, _tx_count: u32, bytes: u32) -> u64 {
    let _ = (bar, bytes);
    // Note: LLVM 20 may not have this intrinsic - may need inline PTX
    // Lowered to inline PTX: mbarrier.arrive.expect_tx.shared.b64 %rd, [%bar], %bytes;
    unreachable!("mbarrier_arrive_expect_tx called outside CUDA kernel context")
}

// =============================================================================
// Cluster-Scope Barrier Operations (for TMA Multicast)
// =============================================================================

/// Arrive at a barrier in another CTA's shared memory within the cluster.
///
/// This is the key synchronization primitive for TMA multicast. When rank 0
/// multicasts B to all CTAs, it must wait for ALL CTAs to consume before
/// reusing the buffer. Each CTA's MMA warp calls this to arrive at rank 0's
/// consumer barrier (MMA_BAR) via distributed shared memory (DSMEM).
///
/// Takes a raw `u64` address (from `map_shared_rank` cast to integer) rather
/// than a `*const Barrier` to avoid LLVM IR address-space mismatches in loop
/// phi nodes. The `mapa` instruction returns a generic-address-space pointer,
/// which conflicts with shared-address-space expectations in loop back-edges.
/// Using `u64` bypasses the pointer type system entirely.
///
/// # Usage Pattern (TMA multicast consumer barrier)
///
/// ```rust,ignore
/// // Before the K-loop (once):
/// let rank0_mma_bar_addr = unsafe {
///     cluster::map_shared_rank(&raw const MMA_BAR0, 0) as u64
/// };
///
/// // Inside the K-loop (after MMA consumes data):
/// unsafe { mbarrier_arrive_cluster(rank0_mma_bar_addr); }
/// ```
///
/// # PTX
///
/// ```ptx
/// mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$addr];
/// ```
///
/// # Safety
///
/// - `remote_bar_addr` must be a valid cluster-scope shared memory address
///   obtained from `map_shared_rank` (mapa instruction)
/// - The target barrier must be initialized and expecting arrivals
/// - Must be called from within a cluster launch context
#[inline(never)]
pub unsafe fn mbarrier_arrive_cluster(remote_bar_addr: u64) {
    let _ = remote_bar_addr;
    unreachable!("mbarrier_arrive_cluster called outside CUDA kernel context")
}

// =============================================================================
// Barrier Invalidation
// =============================================================================

/// Invalidate a barrier, releasing its resources.
///
/// Call this when done with a barrier to allow reinitialization.
/// Typically used when reusing barriers across kernel phases.
///
/// # Safety
///
/// - All threads must have completed their wait operations
/// - No threads should be using the barrier after invalidation
///
/// # PTX
///
/// ```ptx
/// mbarrier.inval.shared.b64 [addr];
/// ```
#[inline(never)]
pub unsafe fn mbarrier_inval(bar: *mut Barrier) {
    let _ = bar;
    // Lowered to: call void @llvm.nvvm.mbarrier.inval.shared(ptr %bar)
    unreachable!("mbarrier_inval called outside CUDA kernel context")
}

// =============================================================================
// Proxy Fence Operations (for TMA synchronization)
// =============================================================================

/// Fence to synchronize generic proxy with async proxy in shared memory.
///
/// This fence ensures that memory operations performed through the generic
/// proxy (normal thread operations like `mbarrier.init`) are visible to the
/// async proxy (hardware async operations like TMA `cp.async.bulk`).
///
/// **Critical for TMA:** Must be called after `mbarrier_init` and before
/// issuing TMA operations. Without this fence, TMA hardware may not see
/// the barrier initialization!
///
/// # Usage Pattern
///
/// ```rust,ignore
/// // Thread 0 initializes barrier
/// if thread::threadIdx_x() == 0 {
///     unsafe {
///         mbarrier_init(&raw mut BAR, count);
///         fence_proxy_async_shared_cta();  // <-- REQUIRED for TMA!
///     }
/// }
/// thread::sync_threads();
///
/// // Now safe to issue TMA operations
/// ```
///
/// # Why This Is Needed
///
/// NVIDIA GPUs have separate memory "proxies":
/// - **Generic Proxy**: Normal thread memory operations
/// - **Async Proxy**: Hardware async operations (TMA, cp.async)
///
/// Without this fence, writes through the generic proxy (like `mbarrier.init`)
/// may not be visible to the async proxy when TMA tries to signal the barrier.
///
/// # Hardware Support
///
/// - **PTX ISA 8.0+**
/// - **sm_90+ (Hopper and newer)**
///
/// # PTX
///
/// # Safety
///
/// Must be called from within a CUDA kernel context.
///
/// ```ptx
/// fence.proxy.async.shared::cta;
/// ```
#[inline(never)]
pub unsafe fn fence_proxy_async_shared_cta() {
    // Lowered to inline PTX: fence.proxy.async.shared::cta;
    unreachable!("fence_proxy_async_shared_cta called outside CUDA kernel context")
}

// =============================================================================
// Thread Scheduling Hints
// =============================================================================

/// Suspend the calling thread for approximately `ns` nanoseconds.
///
/// Used inside spin-wait loops to reduce memory bus contention and allow
/// pending asynchronous operations (like cluster-scope barrier arrivals)
/// to complete. The PTX ISA strongly recommends using this in
/// `mbarrier_try_wait` polling loops.
///
/// # Parameters
///
/// - `ns`: Approximate suspension time in nanoseconds. Values of 0x20–0x100
///   are typical. The actual delay is hardware-dependent and approximate.
///
/// # PTX
///
/// ```ptx
/// nanosleep.u32 N;
/// ```
///
/// # Safety
///
/// Must be called from within a CUDA kernel context.
#[inline(never)]
pub unsafe fn nanosleep(ns: u32) {
    let _ = ns;
    unreachable!("nanosleep called outside CUDA kernel context")
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Arrive and immediately wait (combined operation).
///
/// Convenience function for the common pattern of arriving and then
/// waiting on the same barrier. Equivalent to:
///
/// ```rust,ignore
/// let token = mbarrier_arrive(bar);
/// mbarrier_wait(bar, token);
/// ```
///
/// # Safety
///
/// - `bar` must be initialized
/// - All arriving threads must call this for the barrier to complete
#[inline(always)]
pub unsafe fn mbarrier_arrive_and_wait(bar: *const Barrier) {
    let token = unsafe { mbarrier_arrive(bar) };
    unsafe { mbarrier_wait(bar, token) };
}

// =============================================================================
// Typestate-Based Managed Barrier API
// =============================================================================
//
// This section provides a safer, typestate-based API for barrier management.
// It prevents common mistakes like using a barrier before initialization or
// double-invalidation through compile-time type checking.

use core::marker::PhantomData;

// =============================================================================
// State Markers
// =============================================================================

/// State: Barrier has been claimed but not initialized.
pub struct Uninit;

/// State: Barrier is initialized and ready for arrive/wait operations.
pub struct Ready;

/// State: Barrier has been invalidated and cannot be used.
pub struct Invalidated;

// =============================================================================
// Kind Markers (users can define their own)
// =============================================================================

/// Marker type for TMA-related barriers.
pub struct TmaBarrier;

/// Marker type for MMA/tcgen05 compute barriers.
pub struct MmaBarrier;

/// Marker type for general-purpose barriers.
pub struct GeneralBarrier;

// =============================================================================
// Barrier Token (Newtype)
// =============================================================================

/// Token returned from `arrive()`, must be passed to `wait()`.
///
/// This newtype prevents accidentally passing raw `u64` values
/// where a barrier token is expected.
///
/// # Example
///
/// ```rust,ignore
/// let token = barrier.arrive();
/// barrier.wait(token);  // Type-safe!
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct BarrierToken(u64);

impl BarrierToken {
    /// Get the raw token value (escape hatch for advanced patterns).
    #[inline(always)]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Create from a raw value (for interop with low-level APIs).
    #[inline(always)]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

// =============================================================================
// ManagedBarrier
// =============================================================================

/// A barrier with typestate lifecycle management.
///
/// This type uses Rust's type system to enforce correct barrier usage:
/// - Cannot `arrive()` on an uninitialized barrier
/// - Cannot double-initialize
/// - Cannot use after invalidation
/// - `inval()` consumes the barrier, preventing reuse
///
/// # Type Parameters
///
/// - `State`: Current lifecycle state (`Uninit`, `Ready`, `Invalidated`)
/// - `Kind`: Marker type distinguishing different barriers (`TmaBarrier`, `MmaBarrier`, etc.)
/// - `ID`: Const generic index for multiple barriers of the same kind (default 0)
///
/// # Thread Requirements
///
/// | Operation       | Thread Requirement        |
/// |-----------------|---------------------------|
/// | `from_static()` | Single thread (thread 0)  |
/// | `init()`        | Single thread (thread 0)  |
/// | `arrive()`      | All participating threads |
/// | `wait()`        | All participating threads |
/// | `inval()`       | Single thread (thread 0)  |
///
/// # Example
///
/// ```rust,ignore
/// // Thread 0: Create and initialize
/// let bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR);
/// let bar = unsafe { bar.init(128) };  // Now Ready
/// fence_proxy_async_shared_cta();
/// sync_threads();
///
/// // All threads: arrive and wait
/// let token = bar.arrive();
/// bar.wait(token);
///
/// // Thread 0: Cleanup
/// sync_threads();
/// unsafe { bar.inval(); }
/// ```
pub struct ManagedBarrier<State, Kind, const ID: usize = 0> {
    ptr: *const Barrier,
    _state: PhantomData<State>,
    _kind: PhantomData<Kind>,
}

// Safety: Barrier pointer is only accessed through synchronized operations
unsafe impl<S, K, const ID: usize> Send for ManagedBarrier<S, K, ID> {}

// =============================================================================
// Uninit State Implementation
// =============================================================================

impl<Kind, const ID: usize> ManagedBarrier<Uninit, Kind, ID> {
    /// Create an Uninit barrier from an explicit static declaration.
    ///
    /// Wrap a `static mut Barrier` in the typestate wrapper. Each barrier
    /// must be declared as a separate `static mut` variable, following the
    /// same pattern as `SharedArray`.
    ///
    /// # Thread Requirements
    ///
    /// **Single-thread**: Only call from the initialization thread (thread 0).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BAR0: Barrier = Barrier::UNINIT;
    /// static mut BAR1: Barrier = Barrier::UNINIT;
    ///
    /// if tid == 0 {
    ///     let bar0 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR0);
    ///     let bar1 = ManagedBarrier::<Uninit, GeneralBarrier>::from_static(&raw mut BAR1);
    ///
    ///     let bar0 = unsafe { bar0.init(32) };
    ///     let bar1 = unsafe { bar1.init(32) };
    /// }
    /// ```
    pub fn from_static(ptr: *mut Barrier) -> Self {
        Self {
            ptr,
            _state: PhantomData,
            _kind: PhantomData,
        }
    }

    /// Initialize the barrier with an expected arrival count.
    ///
    /// **All threads in the block should call this.** Only thread 0 performs
    /// the actual initialization; all threads synchronize and receive a `Ready` handle.
    ///
    /// This is a convenience wrapper for `init_by(count, 0)`.
    ///
    /// # Block-Scoped Barriers
    ///
    /// mbarrier operates at **block scope** - each block has its own barrier in
    /// shared memory. Exactly ONE thread per block must call `mbarrier_init`.
    ///
    /// # Safety
    ///
    /// - Must be called before any arrive/wait operations
    /// - All participating threads in the block must call this together
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BAR: Barrier = Barrier::UNINIT;
    ///
    /// // ALL threads call init - only thread 0 actually initializes
    /// let bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR);
    /// let bar = unsafe { bar.init(128) };  // All threads get Ready handle
    /// ```
    #[inline(always)]
    pub unsafe fn init(self, count: u32) -> ManagedBarrier<Ready, Kind, ID> {
        unsafe { self.init_by(count, 0) }
    }

    /// Initialize the barrier with a specific thread performing initialization.
    ///
    /// **All threads in the block should call this.** Only the thread with
    /// `threadIdx.x == init_thread` performs the actual initialization;
    /// all threads synchronize and receive a `Ready` handle.
    ///
    /// # Block-Scoped Barriers
    ///
    /// mbarrier operates at **block scope** - each block has its own barrier in
    /// shared memory. Exactly ONE thread per block must call `mbarrier_init`.
    /// Any thread can be the initializer, not just thread 0.
    ///
    /// # Parameters
    ///
    /// - `count`: Expected number of arrivals before barrier completes
    /// - `init_thread`: Thread ID (threadIdx.x) that performs initialization
    ///
    /// # Safety
    ///
    /// - Must be called before any arrive/wait operations
    /// - All participating threads in the block must call this together
    /// - `init_thread` must be a valid thread ID within the block
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BAR: Barrier = Barrier::UNINIT;
    ///
    /// // Use thread 31 (last thread in first warp) as initializer
    /// let bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR);
    /// let bar = unsafe { bar.init_by(128, 31) };
    /// ```
    #[inline(always)]
    pub unsafe fn init_by(self, count: u32, init_thread: u32) -> ManagedBarrier<Ready, Kind, ID> {
        if crate::thread::threadIdx_x() == init_thread {
            unsafe {
                mbarrier_init(self.ptr as *mut Barrier, count);
                fence_proxy_async_shared_cta();
            }
        }
        // All threads synchronize - ensures init is visible to all
        crate::thread::sync_threads();

        ManagedBarrier {
            ptr: self.ptr,
            _state: PhantomData,
            _kind: PhantomData,
        }
    }
}

// =============================================================================
// Ready State Implementation
// =============================================================================

impl<Kind, const ID: usize> ManagedBarrier<Ready, Kind, ID> {
    /// Get the raw pointer to the underlying barrier.
    ///
    /// Useful for interop with low-level APIs.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const Barrier {
        self.ptr
    }

    /// Arrive at the barrier.
    ///
    /// Returns a token that must be passed to `wait()` or `try_wait()`.
    ///
    /// # Thread Requirements
    ///
    /// All participating threads must call this.
    #[inline(always)]
    pub fn arrive(&self) -> BarrierToken {
        unsafe { BarrierToken(mbarrier_arrive(self.ptr)) }
    }

    /// Arrive at the barrier expecting TMA transaction bytes.
    ///
    /// Use when this barrier tracks TMA copy completion. The barrier won't
    /// complete until both all arrivals occur AND all expected bytes transfer.
    ///
    /// # Thread Requirements
    ///
    /// **Single-thread**: The thread that issued the TMA copy.
    #[inline(always)]
    pub fn arrive_expect_tx(&self, bytes: u32) -> BarrierToken {
        unsafe { BarrierToken(mbarrier_arrive_expect_tx(self.ptr, 1, bytes)) }
    }

    /// Wait for barrier completion (blocking).
    ///
    /// Blocks until all expected arrivals have occurred.
    ///
    /// # Thread Requirements
    ///
    /// All participating threads should wait.
    #[inline(always)]
    pub fn wait(&self, token: BarrierToken) {
        unsafe { mbarrier_wait(self.ptr, token.0) }
    }

    /// Try to wait for barrier completion (non-blocking).
    ///
    /// Returns `true` if the barrier phase is complete, `false` otherwise.
    /// Preferred over busy-looping on `test_wait` due to better scheduling hints.
    #[inline(always)]
    pub fn try_wait(&self, token: BarrierToken) -> bool {
        unsafe { mbarrier_try_wait(self.ptr, token.0) }
    }

    /// Test if barrier phase is complete (non-blocking).
    #[inline(always)]
    pub fn test_wait(&self, token: BarrierToken) -> bool {
        unsafe { mbarrier_test_wait(self.ptr, token.0) }
    }

    /// Try wait using parity (for tcgen05.commit patterns).
    ///
    /// Use when the producer arrives via operations that don't return tokens
    /// (like `tcgen05_commit`).
    #[inline(always)]
    pub fn try_wait_parity(&self, parity: u32) -> bool {
        unsafe { mbarrier_try_wait_parity(self.ptr, parity) }
    }

    /// Invalidate the barrier.
    ///
    /// **All threads in the block should call this.** Only thread 0 performs
    /// the actual invalidation; all threads synchronize before returning.
    ///
    /// This is a convenience wrapper for `inval_by(0)`.
    ///
    /// Consumes the `Ready` barrier and returns an `Invalidated` barrier.
    /// The underlying memory can be reused after this.
    ///
    /// # Safety
    ///
    /// - All threads must have completed their wait operations before calling
    /// - All participating threads in the block must call this together
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // ALL threads call inval - only thread 0 actually invalidates
    /// let _dead = unsafe { bar.inval() };  // Consumes Ready, returns Invalidated
    /// ```
    #[inline(always)]
    pub unsafe fn inval(self) -> ManagedBarrier<Invalidated, Kind, ID> {
        unsafe { self.inval_by(0) }
    }

    /// Invalidate the barrier with a specific thread performing invalidation.
    ///
    /// **All threads in the block should call this.** Only the thread with
    /// `threadIdx.x == inval_thread` performs the actual invalidation;
    /// all threads synchronize before returning.
    ///
    /// Consumes the `Ready` barrier and returns an `Invalidated` barrier.
    ///
    /// # Parameters
    ///
    /// - `inval_thread`: Thread ID (threadIdx.x) that performs invalidation
    ///
    /// # Safety
    ///
    /// - All threads must have completed their wait operations before calling
    /// - All participating threads in the block must call this together
    /// - `inval_thread` must be a valid thread ID within the block
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use thread 31 as the invalidator
    /// let _dead = unsafe { bar.inval_by(31) };
    /// ```
    #[inline(always)]
    pub unsafe fn inval_by(self, inval_thread: u32) -> ManagedBarrier<Invalidated, Kind, ID> {
        // Ensure all threads are done with the barrier before invalidating
        crate::thread::sync_threads();

        if crate::thread::threadIdx_x() == inval_thread {
            unsafe { mbarrier_inval(self.ptr as *mut Barrier) };
        }

        // All threads synchronize - ensures inval is complete
        crate::thread::sync_threads();

        ManagedBarrier {
            ptr: self.ptr,
            _state: PhantomData,
            _kind: PhantomData,
        }
    }
}

// =============================================================================
// Type Aliases for Convenience
// =============================================================================

/// TMA barrier handle (single instance, ID=0)
pub type TmaBarrierHandle<S> = ManagedBarrier<S, TmaBarrier, 0>;

/// MMA barrier handle (single instance, ID=0)
pub type MmaBarrierHandle<S> = ManagedBarrier<S, MmaBarrier, 0>;

/// Double-buffered TMA barrier #0
pub type TmaBarrier0<S> = ManagedBarrier<S, TmaBarrier, 0>;

/// Double-buffered TMA barrier #1
pub type TmaBarrier1<S> = ManagedBarrier<S, TmaBarrier, 1>;
