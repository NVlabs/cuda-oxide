use cuda_core::DeviceBuffer;
use cuda_host::{ProgramArgumentRole, ProgramResourceRole};
use cuda_macros::{cuda_program, kernel};

pub struct DisjointSlice<T>(*mut T);

#[cuda_program]
mod kernels {
    use super::*;

    #[kernel]
    pub fn first(input: &[f32], mut scratch: DisjointSlice<f32>) {
        let _ = input;
        let _ = &mut scratch;
    }

    #[kernel]
    pub fn second(scratch: &[f32], mut output: DisjointSlice<f32>, scale: f32) {
        let _ = scratch;
        let _ = &mut output;
        let _ = scale;
    }

    #[program]
    pub fn forward(
        input: &DeviceBuffer<f32>,
        scratch: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        scale: f32,
        n: u32,
    ) {
        first(input, scratch).grid_len(n);
        second(scratch, output, scale).grid_len(n);
    }
}

fn main() {
    let metadata = kernels::ForwardGraph::METADATA;
    assert_eq!(metadata.operations, &["first", "second"]);
    assert_eq!(metadata.resources[0].role, ProgramResourceRole::Input);
    assert_eq!(metadata.resources[1].role, ProgramResourceRole::Scratch);
    assert_eq!(metadata.resources[2].role, ProgramResourceRole::Output);
    assert_eq!(metadata.resources[3].role, ProgramResourceRole::Scalar);
    assert_eq!(metadata.dependencies[0].resource, "scratch");
    assert_eq!(metadata.operation_nodes[1].arguments[2].role, ProgramArgumentRole::Scalar);
}

