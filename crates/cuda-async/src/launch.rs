/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA kernel launch builder with argument marshalling.
//!
//! [`AsyncKernelLaunch`] accumulates a kernel function reference, launch
//! configuration, and type-erased argument pointers, then submits the launch
//! through the CUDA driver when executed as a [`DeviceOperation`].
//!
//! Arguments are heap-allocated via [`KernelArgument::push_arg`] and freed when
//! the launcher is dropped after submission. This keeps the pointed-to values
//! alive until `cuLaunchKernel` / `cuLaunchKernelEx` has copied the launch
//! parameter values out of the host-side argument array.
//!
//! [`DeviceOperation`]: crate::device_operation::DeviceOperation

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOperation, ExecutionContext};
use crate::error::DeviceError;
use crate::scheduling_policies::SchedulingPolicy;
use cuda_core::{CudaFunction, CudaStream, LaunchConfig};
use std::ffi::c_void;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::sync::Arc;

/// Builder that accumulates kernel arguments and submits a CUDA kernel launch.
///
/// Implements [`DeviceOperation`] so it can be composed with other operations
/// and scheduled onto any stream. Also implements [`IntoFuture`] for `.await`
/// syntax.
#[derive(Debug)]
pub struct AsyncKernelLaunch<'a> {
    /// Handle to the compiled device function.
    pub func: Arc<CudaFunction>,
    /// Heap-allocated, type-erased argument storage passed to the CUDA driver.
    args: KernelArgStorage,
    /// Grid/block dimensions and shared memory size. Must be set before launch.
    cfg: Option<LaunchConfig>,
    /// Optional thread-block cluster dimensions for `cuLaunchKernelEx`.
    cluster_dim: Option<(u32, u32, u32)>,
    /// When `true`, launch via `cuLaunchKernelEx` with
    /// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` set, which guarantees the whole grid
    /// is co-resident on the device (required for `cuda_device::grid::sync()`).
    cooperative: bool,
    /// Ties borrowed device buffers to the lazy launch operation.
    _borrows: PhantomData<&'a mut ()>,
}

/// # Safety
///
/// The `*mut c_void` pointers in `args` are heap-allocated boxes that do not
/// alias mutable state. The `Arc<CudaFunction>` is `Send + Sync`.
unsafe impl<'a> Send for AsyncKernelLaunch<'a> {}

#[derive(Debug, Default)]
struct KernelArgStorage {
    ptrs: Vec<*mut c_void>,
    drops: Vec<unsafe fn(*mut c_void)>,
}

impl Drop for KernelArgStorage {
    fn drop(&mut self) {
        for (arg, drop_arg) in self.ptrs.drain(..).zip(self.drops.drain(..)) {
            unsafe { drop_arg(arg) };
        }
    }
}

impl KernelArgStorage {
    fn push_boxed_arg<T>(&mut self, arg: Box<T>) {
        unsafe fn drop_box<T>(arg: *mut c_void) {
            let _ = unsafe { Box::from_raw(arg as *mut T) };
        }

        self.ptrs.push(Box::into_raw(arg) as *mut c_void);
        self.drops.push(drop_box::<T>);
    }

    fn push_scalar_arg<T: Copy>(&mut self, arg: T) {
        self.push_boxed_arg(Box::new(arg));
    }

    fn as_mut_slice(&mut self) -> &mut [*mut c_void] {
        &mut self.ptrs
    }
}

impl<'a> AsyncKernelLaunch<'a> {
    /// Creates a launcher for `func` with no arguments and no launch config.
    pub fn new(func: Arc<CudaFunction>) -> Self {
        Self {
            func,
            args: KernelArgStorage::default(),
            cfg: None,
            cluster_dim: None,
            cooperative: false,
            _borrows: PhantomData,
        }
    }

    /// Appends a by-value `Copy` argument to the kernel launch packet.
    ///
    /// This is the typed-module path used for scalar, raw-pointer, custom
    /// `Copy` structs, and `Copy` closure arguments.
    #[inline(always)]
    pub fn push_scalar_arg<T: Copy + 'a>(&mut self, arg: T) -> &mut Self {
        self.args.push_scalar_arg(arg);
        self
    }

    /// Appends a kernel argument. The value is heap-allocated and its pointer
    /// stored for the driver call.
    ///
    /// Scalars like `u32`, `f32`, `u64` etc. are auto-boxed -- no need to
    /// wrap them in `Box::new`.
    ///
    /// The allocated storage remains alive until launch submission finishes or
    /// the builder is dropped without launching.
    #[inline(always)]
    pub fn push_arg<T: KernelArgument>(&mut self, arg: T) -> &mut Self {
        arg.push_arg(self);
        self
    }

    /// Appends multiple kernel arguments at once from a tuple.
    ///
    /// Equivalent to chained [`push_arg`](Self::push_arg) calls but allows
    /// grouping all arguments in a single expression:
    ///
    /// ```ignore
    /// launch.push_args((m, n, k, alpha, a_ptr, a_len, b_ptr, b_len, beta, c_ptr, c_len));
    /// ```
    ///
    /// Supports tuples up to 32 elements.
    #[inline(always)]
    pub fn push_args<T: KernelArguments>(&mut self, args: T) -> &mut Self {
        args.push_args(self);
        self
    }

    /// Sets the grid/block dimensions and shared memory size for the launch.
    pub fn set_launch_config(&mut self, cfg: LaunchConfig) -> &mut Self {
        self.cfg = Some(cfg);
        self
    }

    /// Sets thread-block cluster dimensions for a cluster launch.
    pub fn set_cluster_dim(&mut self, cluster_dim: (u32, u32, u32)) -> &mut Self {
        self.cluster_dim = Some(cluster_dim);
        self
    }

    /// Marks this launch as cooperative (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`).
    ///
    /// A cooperative launch guarantees every block in the grid is co-resident
    /// on the device, which is the precondition for grid-wide barriers like
    /// `cuda_device::grid::sync()`. May be combined with
    /// [`set_cluster_dim`](Self::set_cluster_dim); both attributes are then
    /// passed to `cuLaunchKernelEx` in the same call.
    pub fn set_cooperative(&mut self, cooperative: bool) -> &mut Self {
        self.cooperative = cooperative;
        self
    }

    /// Submits the kernel to `stream` via `cuLaunchKernel`, or via
    /// `cuLaunchKernelEx` when cluster dimensions or a cooperative launch
    /// were requested.
    ///
    /// # Safety
    ///
    /// - `self.func` must refer to a kernel loaded from the same CUDA context
    ///   that owns `stream`.
    /// - All argument pointers in `self.args` must point to correctly typed and
    ///   aligned host-side values for the kernel's formal parameters.
    /// - The pointed-to argument values must remain valid until launch
    ///   submission returns. The stream-aware launch helper binds the correct
    ///   context before calling into the CUDA driver.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::Launch`] if no launch config was set, or
    /// [`DeviceError::Driver`] if context binding or launch submission fails.
    unsafe fn launch(mut self, stream: &Arc<CudaStream>) -> Result<(), DeviceError> {
        let cfg = self
            .cfg
            .ok_or_else(|| DeviceError::Launch("Launch config not set.".to_string()))?;
        let result = match (self.cluster_dim, self.cooperative) {
            (Some(cluster_dim), true) => unsafe {
                cuda_core::launch_kernel_ex_cooperative_on_stream(
                    self.func.as_ref(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    cluster_dim,
                    stream.as_ref(),
                    self.args.as_mut_slice(),
                )
            },
            (Some(cluster_dim), false) => unsafe {
                cuda_core::launch_kernel_ex_on_stream(
                    self.func.as_ref(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    cluster_dim,
                    stream.as_ref(),
                    self.args.as_mut_slice(),
                )
            },
            (None, true) => unsafe {
                cuda_core::launch_kernel_cooperative_on_stream(
                    self.func.as_ref(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    stream.as_ref(),
                    self.args.as_mut_slice(),
                )
            },
            (None, false) => unsafe {
                cuda_core::launch_kernel_on_stream(
                    self.func.as_ref(),
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    stream.as_ref(),
                    self.args.as_mut_slice(),
                )
            },
        };
        result.map_err(DeviceError::Driver)?;
        Ok(())
    }
}

/// Owns resources for a lazy async kernel launch and returns them when the GPU
/// work has completed.
#[derive(Debug)]
pub struct OwnedAsyncKernelLaunch<R: Send> {
    launch: AsyncKernelLaunch<'static>,
    resources: R,
}

unsafe impl<R: Send> Send for OwnedAsyncKernelLaunch<R> {}

impl<R: Send> OwnedAsyncKernelLaunch<R> {
    /// Creates an owned async kernel operation from a prepared launch and the
    /// resources that must stay alive until completion.
    pub fn new(launch: AsyncKernelLaunch<'static>, resources: R) -> Self {
        Self { launch, resources }
    }
}

/// Trait for types that can be marshalled into a CUDA kernel argument list.
///
/// Implementors heap-allocate the value and push a `*mut c_void` into the
/// launcher's argument vector.
///
/// Implemented for all common scalar primitives (`u8`–`u64`, `i8`–`i64`,
/// `f32`, `f64`, `usize`, `isize`, `bool`) so callers can write
/// `.push_arg(42u32)` without manual `Box::new`.
pub trait KernelArgument {
    /// Heap-allocates `self` and appends the pointer to `launcher.args`.
    fn push_arg(self, launcher: &mut AsyncKernelLaunch<'_>);
}

/// Passes the box's raw pointer directly as a kernel argument.
///
/// This is the low-level escape hatch. Prefer passing scalars directly -- they
/// are auto-boxed via the blanket scalar impls.
impl<T: 'static> KernelArgument for Box<T> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch<'_>) {
        launcher.args.push_boxed_arg(self);
    }
}

macro_rules! impl_scalar_kernel_arg {
    ($($t:ty),*) => {
        $(
            impl KernelArgument for $t {
                #[inline(always)]
                fn push_arg(self, launcher: &mut AsyncKernelLaunch<'_>) {
                    launcher.push_scalar_arg(self);
                }
            }
        )*
    };
}

impl_scalar_kernel_arg!(
    u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64, bool
);

// ---------------------------------------------------------------------------
// KernelArguments — push multiple heterogeneous args in a single call
// ---------------------------------------------------------------------------

/// Trait for a group of kernel arguments that can be pushed together.
///
/// Implemented for tuples of [`KernelArgument`] types up to arity 32, enabling
/// `launch.push_args((dim_m, dim_n, alpha, ptr, len))` as an alternative to
/// chained `.push_arg()` calls.
#[diagnostic::on_unimplemented(
    message = "cannot push `{Self}` as kernel arguments",
    note = "KernelArguments is implemented for tuples of KernelArgument types up to 32 elements"
)]
pub trait KernelArguments {
    /// Pushes every element into `launcher` in order.
    fn push_args(self, launcher: &mut AsyncKernelLaunch<'_>);
}

macro_rules! impl_kernel_args_tuple {
    // Base case: empty tuple
    () => {
        impl KernelArguments for () {
            #[inline(always)]
            fn push_args(self, _launcher: &mut AsyncKernelLaunch<'_>) {}
        }
    };
    // Recursive case: (A, B, C, ...) where each element is a KernelArgument
    ($($idx:tt : $T:ident),+) => {
        impl<$($T: KernelArgument),+> KernelArguments for ($($T,)+) {
            #[inline(always)]
            fn push_args(self, launcher: &mut AsyncKernelLaunch<'_>) {
                $(launcher.push_arg(self.$idx);)+
            }
        }
    };
}

impl_kernel_args_tuple!();
impl_kernel_args_tuple!(0: A);
impl_kernel_args_tuple!(0: A, 1: B);
impl_kernel_args_tuple!(0: A, 1: B, 2: C);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD, 30: AE);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD, 30: AE, 31: AF);

/// Launches the kernel on the stream bound to the execution context.
impl<'a> DeviceOperation for AsyncKernelLaunch<'a> {
    type Output = ();

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        unsafe { self.launch(ctx.get_cuda_stream()) }
    }
}

/// Schedules the kernel launch via the thread-local default scheduling policy.
impl<'a> IntoFuture for AsyncKernelLaunch<'a> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), AsyncKernelLaunch<'a>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

impl<R: Send> DeviceOperation for OwnedAsyncKernelLaunch<R> {
    type Output = R;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<R, DeviceError> {
        let Self { launch, resources } = self;
        unsafe { launch.launch(ctx.get_cuda_stream()) }?;
        Ok(resources)
    }
}

impl<R: Send> IntoFuture for OwnedAsyncKernelLaunch<R> {
    type Output = Result<R, DeviceError>;
    type IntoFuture = DeviceFuture<R, OwnedAsyncKernelLaunch<R>>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct TestParams {
        scale: f32,
        bias: i32,
    }

    #[test]
    fn scalar_arg_storage_accepts_custom_copy_value() {
        let mut storage = KernelArgStorage::default();
        let params = TestParams {
            scale: 2.0,
            bias: 3,
        };

        storage.push_scalar_arg(params);

        assert_eq!(storage.ptrs.len(), 1);
        assert_eq!(unsafe { *(storage.ptrs[0] as *const TestParams) }, params);
    }

    #[test]
    fn arg_storage_drops_values_with_their_original_type() {
        struct DropCounter(Rc<Cell<usize>>);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Rc::new(Cell::new(0));
        let mut storage = KernelArgStorage::default();

        storage.push_boxed_arg(Box::new(DropCounter(Rc::clone(&drops))));
        assert_eq!(drops.get(), 0);

        drop(storage);
        assert_eq!(drops.get(), 1);
    }
}
