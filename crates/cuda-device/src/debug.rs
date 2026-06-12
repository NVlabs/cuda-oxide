/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU Debug and Profiling Intrinsics
//!
//! This module provides intrinsics for debugging and profiling GPU kernels:
//!
//! | Function           | Description                   | CUDA C++ Equivalent |
//! |--------------------|-------------------------------|---------------------|
//! | [`clock()`]        | Read 32-bit GPU clock counter | `clock()`           |
//! | [`clock64()`]      | Read 64-bit GPU clock counter | `clock64()`         |
//! | [`globaltimer()`]  | Read GPU global timer         | `%globaltimer`      |
//! | [`trap()`]         | Abort kernel execution        | `__trap()`          |
//! | [`breakpoint()`]   | Insert cuda-gdb breakpoint    | `__brkpt()`         |
//! | [`prof_trigger()`] | Signal NVIDIA profiler        | `__prof_trigger(N)` |
//!
//! # Example: Micro-benchmarking
//!
//! ```rust,ignore
//! use cuda_device::debug;
//!
//! let start = debug::clock64();
//! // ... computation to measure ...
//! let end = debug::clock64();
//! let cycles = end - start;
//! ```
//!
//! # Example: Runtime Assertion
//!
//! ```rust,ignore
//! use cuda_device::debug;
//!
//! if value < 0 {
//!     debug::trap();  // Kernel aborts
//! }
//! ```

// =============================================================================
// Clock/Timing Intrinsics
// =============================================================================

/// Read the GPU clock counter (32-bit).
///
/// Returns the current value of a per-SM clock counter. Useful for micro-benchmarking
/// kernel code by measuring elapsed cycles between operations.
///
/// # Returns
///
/// A 32-bit clock cycle count. Note that this wraps around relatively quickly
/// (~4 billion cycles). For longer measurements, use [`clock64()`].
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::debug;
///
/// let start = debug::clock();
/// // ... some computation ...
/// let end = debug::clock();
/// let cycles = end.wrapping_sub(start);
/// ```
///
/// # Notes
///
/// - Clock frequency varies by GPU and power state
/// - Different SMs may have slightly different clock values
/// - Use for relative timing within a kernel, not absolute time
#[inline(never)]
pub fn clock() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.clock()
    unreachable!("clock called outside CUDA kernel context")
}

/// Read the GPU clock counter (64-bit).
///
/// Returns the current value of a per-SM clock counter as a 64-bit value.
/// Preferred over [`clock()`] for measurements that might exceed 32-bit range.
///
/// # Returns
///
/// A 64-bit clock cycle count that won't wrap around for practical measurements.
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::debug;
///
/// let start = debug::clock64();
/// // ... some computation ...
/// let end = debug::clock64();
/// let cycles = end - start;  // No wrapping concerns
/// ```
///
/// # Notes
///
/// - Clock frequency varies by GPU and power state
/// - Different SMs may have slightly different clock values
/// - Use for relative timing within a kernel, not absolute time
#[inline(never)]
pub fn clock64() -> u64 {
    // Lowered to: call i64 @llvm.nvvm.read.ptx.sreg.clock64()
    unreachable!("clock64 called outside CUDA kernel context")
}

/// Read the GPU global timer.
///
/// Returns a 64-bit timer value from PTX `%globaltimer`. Unlike [`clock64()`],
/// this is a global timer source, so it is preferable when measuring interactions
/// that may span multiple SMs.
///
/// # Example
///
/// ```rust,no_run
/// use cuda_device::debug;
///
/// let start = debug::globaltimer();
/// // ... operation to measure ...
/// let end = debug::globaltimer();
/// let ticks = end.wrapping_sub(start);
/// ```
#[inline(never)]
pub fn globaltimer() -> u64 {
    // Lowered to: call i64 @llvm.nvvm.read.ptx.sreg.globaltimer()
    unreachable!("globaltimer called outside CUDA kernel context")
}

// =============================================================================
// Trap/Abort Intrinsics
// =============================================================================

/// Abort kernel execution immediately.
///
/// This is equivalent to `__trap()` in CUDA C/C++. When any thread executes
/// `trap()`, the kernel is terminated and an error is reported to the host.
///
/// # Use Cases
///
/// - Runtime error checking (like `assert` but unconditional)
/// - Detecting invalid states that should never occur
/// - Debugging to stop execution at a specific point
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::debug;
///
/// if invalid_condition {
///     debug::trap();  // Kernel dies here
/// }
/// ```
///
/// # Notes
///
/// - The kernel terminates with an error status
/// - No error message is provided (use `gpu_assert!` for messages)
/// - Host will see a CUDA error when synchronizing
#[inline(never)]
pub fn trap() -> ! {
    // Lowered to: call void @llvm.nvvm.trap()
    unreachable!("trap called outside CUDA kernel context")
}

// =============================================================================
// Debugging Intrinsics
// =============================================================================

/// Insert a breakpoint for cuda-gdb.
///
/// When debugging with cuda-gdb, execution will stop at this point.
/// This is equivalent to `__brkpt()` in CUDA C/C++.
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::debug;
///
/// // Only break on thread 0 to avoid overwhelming the debugger
/// if thread::index_1d().get() == 0 {
///     debug::breakpoint();
/// }
/// ```
///
/// # Notes
///
/// - Only effective when running under cuda-gdb
/// - Without cuda-gdb, this is typically a no-op or trap
/// - Be careful about placing breakpoints in divergent code
#[inline(never)]
pub fn breakpoint() {
    // Lowered to: call void @llvm.nvvm.brkpt() or inline PTX "brkpt;"
    unreachable!("breakpoint called outside CUDA kernel context")
}

/// Signal the NVIDIA profiler at a specific point.
///
/// Triggers a profiler event that can be viewed in Nsight Systems or
/// Nsight Compute. The counter `N` identifies the trigger point.
///
/// This is equivalent to `__prof_trigger(N)` in CUDA C/C++.
///
/// # Type Parameters
///
/// * `N` - The profiler counter ID (0-15 typically)
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::debug;
///
/// debug::prof_trigger::<0>();  // Mark start of region
/// // ... computation ...
/// debug::prof_trigger::<1>();  // Mark end of region
/// ```
///
/// # Notes
///
/// - Useful for marking regions of interest in profiler traces
/// - Counter IDs should be consistent across kernel invocations
/// - Has minimal overhead when not profiling
#[inline(never)]
pub fn prof_trigger<const N: u32>() {
    // Lowered to: inline PTX "pmevent N;"
    unreachable!("prof_trigger called outside CUDA kernel context")
}

// =============================================================================
// Assertion Macro
// =============================================================================

/// GPU-side assertion macro.
///
/// Checks a condition at runtime and aborts the kernel if it fails.
/// This is equivalent to `assert!()` but works on GPU kernels.
///
/// # Usage
///
/// ```rust,ignore
/// use cuda_device::gpu_assert;
///
/// // Simple assertion
/// gpu_assert!(x >= 0);
///
/// // With custom message (message is ignored in current impl)
/// gpu_assert!(idx < len, "Index out of bounds");
/// ```
///
/// # Behavior
///
/// When the condition is false:
/// - The kernel execution is aborted via `trap()`
/// - The CUDA driver reports an error to the host
/// - Other threads may continue briefly before the error propagates
///
/// # Notes
///
/// - Use sparingly in performance-critical code
/// - The message argument is currently ignored (will be supported with assertfail)
/// - For debugging, consider using [`breakpoint()`] instead
#[macro_export]
macro_rules! gpu_assert {
    ($cond:expr) => {
        if !$cond {
            $crate::debug::trap();
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            // TODO (npasham): Use llvm.nvvm.assertfail for better error messages with file/line
            $crate::debug::trap();
        }
    };
}

// =============================================================================
// Printf Support
// =============================================================================

/// Internal vprintf wrapper for GPU printf support.
///
/// This function is recognized by the cuda-oxide compiler and replaced with
/// an actual `vprintf` call in the generated PTX. Do not call directly.
///
/// # Arguments
///
/// * `format` - Pointer to null-terminated C format string (in global memory)
/// * `args` - Pointer to packed argument buffer (following C vararg ABI)
///
/// # Returns
///
/// Number of arguments on success, negative value on error.
/// Note: Unlike standard C printf which returns character count, CUDA's vprintf
/// returns the argument count because the GPU only marshals args to a buffer -
/// the host does the actual formatting later.
///
/// # Safety
///
/// This function only works within CUDA kernel context. The compiler replaces
/// calls with actual vprintf instructions. Calling from host code will panic.
#[doc(hidden)]
#[inline(never)]
pub fn __gpu_vprintf(_format: *const u8, _args: *const u8) -> i32 {
    unreachable!("__gpu_vprintf called outside CUDA kernel context")
}

/// Trait for GPU printf argument promotion.
///
/// Implements C vararg promotion rules:
/// - `i8`, `i16` → `i32`
/// - `u8`, `u16` → `u32`
/// - `f32` → `f64`
/// - `bool` → `i32`
/// - 64-bit types stay as-is
pub trait GpuPrintfArg {
    /// The promoted type for C vararg ABI
    type Promoted: Copy;

    /// C format specifier character for this type
    const FORMAT_CHAR: char;

    /// Whether this is a 64-bit type (needs `ll` modifier)
    const IS_64BIT: bool;

    /// Whether this is a floating point type
    const IS_FLOAT: bool;

    /// Promote the value to the vararg type
    fn promote(self) -> Self::Promoted;
}

// Signed integers
impl GpuPrintfArg for i8 {
    type Promoted = i32;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i32 {
        self as i32
    }
}

impl GpuPrintfArg for i16 {
    type Promoted = i32;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i32 {
        self as i32
    }
}

impl GpuPrintfArg for i32 {
    type Promoted = i32;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i32 {
        self
    }
}

impl GpuPrintfArg for i64 {
    type Promoted = i64;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i64 {
        self
    }
}

impl GpuPrintfArg for isize {
    type Promoted = i64;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i64 {
        self as i64
    }
}

// Unsigned integers
impl GpuPrintfArg for u8 {
    type Promoted = u32;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u32 {
        self as u32
    }
}

impl GpuPrintfArg for u16 {
    type Promoted = u32;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u32 {
        self as u32
    }
}

impl GpuPrintfArg for u32 {
    type Promoted = u32;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u32 {
        self
    }
}

impl GpuPrintfArg for u64 {
    type Promoted = u64;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u64 {
        self
    }
}

impl GpuPrintfArg for usize {
    type Promoted = u64;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u64 {
        self as u64
    }
}

// Floating point
impl GpuPrintfArg for f32 {
    type Promoted = f64;
    const FORMAT_CHAR: char = 'f';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = true;
    fn promote(self) -> f64 {
        self as f64
    }
}

impl GpuPrintfArg for f64 {
    type Promoted = f64;
    const FORMAT_CHAR: char = 'f';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = true;
    fn promote(self) -> f64 {
        self
    }
}

// Boolean
impl GpuPrintfArg for bool {
    type Promoted = i32;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i32 {
        self as i32
    }
}

// Pointers
impl<T> GpuPrintfArg for *const T {
    type Promoted = u64;
    const FORMAT_CHAR: char = 'p';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u64 {
        self as u64
    }
}

impl<T> GpuPrintfArg for *mut T {
    type Promoted = u64;
    const FORMAT_CHAR: char = 'p';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u64 {
        self as u64
    }
}

// Re-export the gpu_printf macro from cuda-macros
pub use cuda_macros::gpu_printf;
