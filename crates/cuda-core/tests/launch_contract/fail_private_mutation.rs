use cuda_core::LaunchConfig1D;

fn main() {
    let mut valid = LaunchConfig1D::new(1, 32, 0);
    valid.block_x = 64;
}
