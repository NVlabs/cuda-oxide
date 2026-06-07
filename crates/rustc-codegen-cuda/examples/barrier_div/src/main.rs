// Minimal repro v2: lane-0 work as a separate (non-inlined) fn + TWO sequential
// divergent-barrier sections + a parallel section between (mirrors nexus mb kernel).
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn single_threaded_fill(s: &mut SharedArray<u32, 64>, mul: u32) {
        let mut i = 0u32;
        while i < 64 { s[i as usize] = i.wrapping_mul(mul); i += 1; }
    }

    #[kernel]
    pub fn barrier_div(mut out: DisjointSlice<u32>) {
        static mut S: SharedArray<u32, 64> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        // Section 1: lane-0 single-threaded fn call + barrier
        if tid == 0 { unsafe { single_threaded_fill(&mut *(&raw mut S), 7); } }
        thread::sync_threads();

        // Parallel section (all lanes), then read
        let a = unsafe { S[tid as usize] };

        // Section 2: another lane-0 single-threaded fn call + barrier
        if tid == 0 { unsafe { single_threaded_fill(&mut *(&raw mut S), 3); } }
        thread::sync_threads();
        let b = unsafe { S[tid as usize] };

        if let Some(o) = out.get_mut(gid) { *o = a.wrapping_add(b); }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = ctx.load_module_from_file("barrier_div.ptx").expect("load ptx");
    let module = kernels::from_module(module).expect("typed module");
    let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (64,1,1), shared_mem_bytes: 0 };
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 64).unwrap();
    unsafe { module.barrier_div((stream).as_ref(), cfg, &mut out_dev) }.expect("launch");
    stream.synchronize().expect("sync");
    let r = out_dev.to_host_vec(&stream).unwrap();
    let ok = (0..64u32).all(|i| r[i as usize] == i.wrapping_mul(7).wrapping_add(i.wrapping_mul(3)));
    println!("out[0..8]={:?} (expect i*7+i*3=i*10); PASS={}", &r[..8], ok);
}
