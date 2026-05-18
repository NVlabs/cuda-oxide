use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn k_sqrt(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let x = a[i];
            *o = if x >= 0.0f32 { x.sqrt() } else { 0.0f32 };
        }
    }
}
fn h(v: f32) -> String { format!("{:08x}", v.to_bits()) }
fn main() {
    let ctx = CudaContext::new(0).unwrap(); let s = ctx.default_stream();
    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 0.003 + 0.01).collect();
    let cpu: Vec<f32> = a.iter().map(|&x| x.sqrt()).collect();
    let ad = DeviceBuffer::from_host(&s, &a).unwrap();
    let mut od = DeviceBuffer::<f32>::zeroed(&s, N).unwrap();
    let m = kernels::load(&ctx).unwrap();
    m.k_sqrt(&s, LaunchConfig::for_num_elems(N as u32), &ad, &mut od).unwrap();
    let r = od.to_host_vec(&s).unwrap();
    let mut diffs = 0;
    let mut max_ulp: u32 = 0;
    for i in 0..N {
        let g = r[i].to_bits(); let c = cpu[i].to_bits();
        if g != c {
            diffs += 1;
            let u = if g > c { g - c } else { c - g };
            if u > max_ulp { max_ulp = u; }
        }
    }
    if diffs == 0 { println!("k_sqrt: BIT-IDENTICAL | {} elements", N); }
    else {
        println!("k_sqrt: {}/{} differ, max ULP = {} | {}", diffs, N, max_ulp,
                 if max_ulp <= 2 { "PASS (≤2 ULP)" } else { "FAIL" });
        if max_ulp > 2 { std::process::exit(1); }
    }
}
