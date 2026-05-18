use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// erf is not stable in std — we need to provide it for the GPU kernel
// On GPU, our from_cmath_callee intercept will map it to __nv_erff
// For CPU reference, use libm::erff

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn k_gelu(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let x = a[i];
            // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            // Use expf-based tanh approximation instead since erf
            // is not directly callable as an intrinsic or cmath function
            // in Rust stable. The tanh-gelu form IS the one llama.cpp uses.
            let c = 0.7978845608f32; // sqrt(2/pi)
            let x3 = x * x * x;
            *o = 0.5f32 * x * (1.0f32 + (c * (x + 0.044715f32 * x3)).tanh());
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap(); let s = ctx.default_stream();
    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| (i as f32)*0.003-1.5).collect();
    // CPU reference using same tanh-gelu form
    let cpu: Vec<f32> = a.iter().map(|&x| {
        let c = 0.7978845608f32;
        let x3 = x * x * x;
        0.5f32 * x * (1.0f32 + (c * (x + 0.044715f32 * x3)).tanh())
    }).collect();
    let ad = DeviceBuffer::from_host(&s,&a).unwrap();
    let mut od = DeviceBuffer::<f32>::zeroed(&s,N).unwrap();
    let m = kernels::load(&ctx).unwrap();
    m.k_gelu(&s,LaunchConfig::for_num_elems(N as u32),&ad,&mut od).unwrap();
    let r = od.to_host_vec(&s).unwrap();
    let mut d=0u32; let mut mx=0u32;
    for i in 0..N { let g=r[i].to_bits(); let c=cpu[i].to_bits();
        if g!=c { d+=1; let u=if g>c{g-c}else{c-g}; if u>mx{mx=u;} } }
    if d==0 { println!("k_gelu: BIT-IDENTICAL | {N}"); }
    else { println!("k_gelu: {d}/{N} differ, max ULP={mx} | {}",if mx<=2{"PASS"}else if mx<=400{"allclose"}else{"FAIL"}); }
}
