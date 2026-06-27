use cuda_macros::device;

#[device]
unsafe extern "Rust" {
    fn wrong_abi(value: *mut f32);
}

fn main() {}
