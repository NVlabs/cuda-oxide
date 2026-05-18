use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn k_sigmoid(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let x = a[idx_raw];
            *o = 1.0f32 / (1.0f32 + (-x).exp());
        }
    }
}

fn f32_to_hex(val: f32) -> String { format!("{:08x}", val.to_bits()) }

fn main() {
    println!("=== CUDA-oxide k_sigmoid on sm_61 ===\n");
    let ctx = CudaContext::new(0).expect("No CUDA device");
    let stream = ctx.default_stream();
    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 0.003 - 1.5).collect();
    let cpu: Vec<f32> = a.iter().map(|&x| 1.0f32 / (1.0f32 + (-x).exp())).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let module = kernels::load(&ctx).expect("Failed to load CUDA module");
    module.k_sigmoid(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &mut out_dev).unwrap();
    let result = out_dev.to_host_vec(&stream).unwrap();
    let mut max_ulp: u32 = 0;
    let mut diffs = 0;
    for i in 0..N {
        let g = result[i].to_bits(); let c = cpu[i].to_bits();
        if g != c { diffs += 1; let u = if g > c { g - c } else { c - g }; if u > max_ulp { max_ulp = u; } }
    }
    println!("First 3: gpu={} cpu={}", f32_to_hex(result[0]), f32_to_hex(cpu[0]));
    if diffs == 0 { println!("RESULT: BIT-IDENTICAL ({N} elements)"); }
    else { println!("RESULT: {diffs}/{N} differ, max ULP = {max_ulp}");
        if max_ulp <= 2 { println!("  PASS: within transcendental bound"); }
        else { println!("  FAIL"); std::process::exit(1); }
    }
}
