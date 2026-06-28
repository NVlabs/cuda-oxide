/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Semantic CUDA program graph support.
//!
//! This module intentionally stops before scheduling. A program graph describes
//! the CUDA work the user declared; binding attaches that structure to a loaded
//! CUDA module and produces an explicit executable description that another
//! runtime layer can launch or submit.

/// Target adapter used when binding a semantic graph to CUDA runtime handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramLowering {
    /// Preserve declaration order as plain CUDA launches.
    SequentialLaunches,
}

/// Aggregate role of a named program resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramResourceRole {
    /// Resource is read by the graph and not written.
    Input,
    /// Resource is written by the graph and not read.
    Output,
    /// Resource is both written and read within the graph.
    Scratch,
    /// Resource is passed by value as launch metadata or scalar kernel input.
    Scalar,
}

/// Per-operation access mode for a graph argument.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramArgumentRole {
    /// Argument is read as device data.
    Read,
    /// Argument is written as device data.
    Write,
    /// Argument is passed by value.
    Scalar,
}

/// Static metadata for a named graph resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramResourceMetadata {
    /// User-visible resource name from the graph declaration.
    pub name: &'static str,
    /// Rust type text captured from the graph declaration.
    pub type_name: &'static str,
    /// Aggregate role inferred from operation argument usage.
    pub role: ProgramResourceRole,
}

/// Static metadata for an argument of a graph operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramArgumentMetadata {
    /// Argument expression text as written in the graph declaration.
    pub expression: &'static str,
    /// Matched graph resource name, if the expression refers to one.
    pub resource: Option<&'static str>,
    /// Operation-local role inferred from the kernel signature.
    pub role: ProgramArgumentRole,
}

/// Static metadata for one operation node in a semantic program graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramOperationMetadata {
    /// Operation index in declaration order.
    pub index: usize,
    /// Kernel operation name.
    pub name: &'static str,
    /// Operation argument metadata.
    pub arguments: &'static [ProgramArgumentMetadata],
}

/// Static dependency edge derived from graph resource use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramDependencyMetadata {
    /// Producer operation index.
    pub from: usize,
    /// Consumer operation index.
    pub to: usize,
    /// Resource that creates the dependency.
    pub resource: &'static str,
}

/// Static metadata for a semantic program graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramGraphMetadata {
    /// User-visible graph name.
    pub name: &'static str,
    /// Ordered kernel operation names in the graph.
    pub operations: &'static [&'static str],
    /// Named graph resources.
    pub resources: &'static [ProgramResourceMetadata],
    /// Operation nodes with argument roles.
    pub operation_nodes: &'static [ProgramOperationMetadata],
    /// Dependency edges derived from read/write resource usage.
    pub dependencies: &'static [ProgramDependencyMetadata],
}

/// A semantic CUDA program bound to concrete CUDA module handles.
///
/// `BoundProgram` is not a scheduler. It is the executable description emitted
/// by the semantic graph layer after it has been bound to a loaded CUDA module.
/// The current adapter can launch directly on a stream; future adapters can hand
/// the same metadata to cuda-oxide's async or graph execution layers.
type LaunchThunk<'a> =
    dyn FnOnce(&cuda_core::CudaStream) -> Result<(), cuda_core::DriverError> + Send + 'a;

pub struct BoundProgram<'a> {
    metadata: ProgramGraphMetadata,
    lowering: ProgramLowering,
    launch: Box<LaunchThunk<'a>>,
}

impl<'a> BoundProgram<'a> {
    /// Creates a bound program from a launch adapter closure.
    pub fn new(
        metadata: ProgramGraphMetadata,
        lowering: ProgramLowering,
        launch: impl FnOnce(&cuda_core::CudaStream) -> Result<(), cuda_core::DriverError> + Send + 'a,
    ) -> Self {
        Self {
            metadata,
            lowering,
            launch: Box::new(launch),
        }
    }

    /// Returns static graph metadata for diagnostics and runtime adapters.
    pub fn metadata(&self) -> ProgramGraphMetadata {
        self.metadata
    }

    /// Returns the adapter this program was bound with.
    pub fn lowering(&self) -> ProgramLowering {
        self.lowering
    }

    /// Launches this bound program on `stream` using its adapter.
    pub fn launch(self, stream: &cuda_core::CudaStream) -> Result<(), cuda_core::DriverError> {
        (self.launch)(stream)
    }
}

#[cfg(feature = "async")]
impl<'a> cuda_async::device_operation::DeviceOperation for BoundProgram<'a> {
    type Output = ();

    unsafe fn execute(
        self,
        context: &cuda_async::device_operation::ExecutionContext,
    ) -> Result<(), cuda_async::error::DeviceError> {
        self.launch(context.get_cuda_stream().as_ref())
            .map_err(cuda_async::error::DeviceError::Driver)
    }
}

#[cfg(feature = "async")]
impl<'a> std::future::IntoFuture for BoundProgram<'a> {
    type Output = Result<(), cuda_async::error::DeviceError>;
    type IntoFuture = cuda_async::device_future::DeviceFuture<(), BoundProgram<'a>>;

    fn into_future(self) -> Self::IntoFuture {
        match cuda_async::device_context::with_default_device_policy(|policy| {
            cuda_async::scheduling_policies::SchedulingPolicy::schedule(policy, self)
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => cuda_async::device_future::DeviceFuture::failed(e),
        }
    }
}

/// Backwards-compatible alias for early users of the initial prototype.
pub type ExecutionPlan<'a> = BoundProgram<'a>;
/// Backwards-compatible alias for early users of the initial prototype.
pub type ExecutionPolicy = ProgramLowering;

#[cfg(test)]
mod tests {
    #[cfg(feature = "async")]
    #[test]
    fn bound_program_is_a_device_operation() {
        fn assert_device_operation<T>()
        where
            T: cuda_async::device_operation::DeviceOperation<Output = ()>,
        {
        }

        assert_device_operation::<super::BoundProgram<'static>>();
    }
}
