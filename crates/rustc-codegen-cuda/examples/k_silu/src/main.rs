use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn k_silu(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let x = a[idx_raw];
            // silu(x) = x / (1 + exp(-x))
            // Use GPU's expf via core::intrinsics or just f32 ops
            *o = x / (1.0f32 + (-x).exp());
        }
    }
}

fn f32_to_hex(val: f32) -> String { format!("{:08x}", val.to_bits()) }

fn main() {
    println!("=== CUDA-oxide k_silu on sm_61 ===\n");
    let ctx = CudaContext::new(0).expect("No CUDA device");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 0.003 - 1.5).collect();
    let cpu: Vec<f32> = a.iter().map(|&x| x / (1.0f32 + (-x).exp())).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load CUDA module");
    module.k_silu(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &mut out_dev).unwrap();
    let result = out_dev.to_host_vec(&stream).unwrap();

    // ULP comparison (silu uses expf — expect ≤2 ULP)
    let mut max_ulp: u64 = 0;
    let mut mismatches_strict = 0;
    for i in 0..N {
        let g = result[i].to_bits();
        let c = cpu[i].to_bits();
        if g != c {
            mismatches_strict += 1;
            let ulp = if g > c { g - c } else { c - g };
            if ulp as u64 > max_ulp { max_ulp = ulp as u64; }
        }
    }

    println!("First 5 (hex):");
    for i in 0..5 {
        let xor = result[i].to_bits() ^ cpu[i].to_bits();
        if xor == 0 {
            println!("  [{}] gpu={} cpu={} IDENTICAL", i, f32_to_hex(result[i]), f32_to_hex(cpu[i]));
        } else {
            println!("  [{}] gpu={} cpu={} xor={:08x}", i, f32_to_hex(result[i]), f32_to_hex(cpu[i]), xor);
        }
    }

    if mismatches_strict == 0 {
        println!("\nRESULT: BIT-IDENTICAL ({N} elements)");
    } else {
        println!("\nRESULT: {mismatches_strict}/{N} differ, max ULP = {max_ulp}");
        if max_ulp <= 2 {
            println!("  PASS: within ≤2 ULP transcendental bound");
        } else {
            println!("  FAIL: exceeds ≤2 ULP bound");
            std::process::exit(1);
        }
    }
}
