use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn k_exp(a: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) { *o = a[i].exp(); }
    }
}
fn main() {
    let ctx = CudaContext::new(0).unwrap(); let s = ctx.default_stream();
    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| (i as f32)*0.003-1.5).collect();
    let cpu: Vec<f32> = a.iter().map(|&x| x.exp()).collect();
    let ad = DeviceBuffer::from_host(&s,&a).unwrap();
    let mut od = DeviceBuffer::<f32>::zeroed(&s,N).unwrap();
    let m = kernels::load(&ctx).unwrap();
    m.k_exp(&s,LaunchConfig::for_num_elems(N as u32),&ad,&mut od).unwrap();
    let r = od.to_host_vec(&s).unwrap();
    let mut d = 0u32; let mut mx = 0u32;
    for i in 0..N { let g=r[i].to_bits(); let c=cpu[i].to_bits();
        if g!=c { d+=1; let u=if g>c{g-c}else{c-g}; if u>mx{mx=u;} } }
    if d==0 { println!("k_exp: BIT-IDENTICAL | {N}"); }
    else { println!("k_exp: {d}/{N} differ, max ULP={mx} | {}",if mx<=2{"PASS"}else{"FAIL"}); }
}
