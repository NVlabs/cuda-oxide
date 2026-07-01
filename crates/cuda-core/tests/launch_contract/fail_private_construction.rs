use cuda_core::LaunchConfig1D;

fn main() {
    let _forged = LaunchConfig1D {
        grid_x: 1,
        block_x: 32,
        shared_mem_bytes: 0,
    };
}
