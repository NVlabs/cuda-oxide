# Supported Features

This appendix presents the cuda-oxide feature matrix: every compiler capability,
runtime API, and hardware feature along with its current support status. The
data is drawn from the compiler/runtime sources and the test suite.

**Legend:** **Full** = tested and working, **Partial** = ships and works but
has a known gap (called out in the row description), **Planned** = on the
roadmap, **N/A** = not applicable or no identified need.

---

## Compiler: Memory Model

| Feature | Status | Description |
|:--------|:-------|:------------|
| HMM / Unified Memory Management | **Full** | GPU directly reads/writes host memory without `cudaMemcpy`. Reference captures in closures leverage HMM for host pointer access. Requires Turing+ GPU, Linux 6.1.24+, CUDA 12.2+. |
| Unified Struct ABI (no `#[repr(C)]`) | **Full** | Device struct layout matches host exactly. The compiler queries rustc's actual layout and reproduces it with explicit padding in LLVM IR. Works with `#[repr(Rust)]` default. |
| Dynamic Layout Matching | **Full** | Compiler queries rustc's `fields_by_offset_order()` and byte offsets, builds LLVM structs with correct field order and explicit padding bytes. Independent of LLVM's datalayout. |
| Pointer Distance (`offset_from`) | **Full** | `ptr_offset_from` / `ptr_offset_from_unsigned` intrinsics (and the `offset_from`, `offset_from_unsigned`, `byte_offset_from`, `byte_offset_from_unsigned` methods) lower to an address difference divided by the rustc-reported pointee size, returning `isize` (signed) or `usize` (unsigned). Errors on a zero-sized pointee. |
| Volatile Load/Store | **Full** | `core::ptr::read_volatile` / `write_volatile` carry an explicit volatile bit through MIR import, mem2reg (volatile accesses are never promoted), MIR-to-LLVM lowering, and textual export (`load volatile` / `store volatile`). Emits `ld.volatile` / `st.volatile` in PTX. |
| Bulk Copy (`copy_nonoverlapping`) | **Full** | `core::ptr::copy_nonoverlapping` lowers to a `mir.memcpy` op and then `llvm.memcpy`, with the element count scaled to bytes for the pointee. The intrinsic overload suffix is derived from the operand address spaces and length width. |

## Compiler: Type System

| Feature | Status | Description |
|:--------|:-------|:------------|
| Generics and Monomorphization | **Full** | Generic kernels and device functions with trait bounds. Monomorphized instances collected from rustc MIR. Const generics supported. |
| Enums (`Option<T>`, `Result<T,E>`, custom) | **Full** | Full enum support including discriminant extraction and payload access. Pattern matching on enums works. |
| Struct Construction and Field Access | **Full** | Struct literals, field access, pass-by-value and return values. User-defined structs supported without annotations. |
| Array Types (`[T; N]`) | **Full** | Static construction, constant- and runtime-index access. Array value constants (bare and nested) materialized. Mutable arrays auto-promoted to memory-backed. |
| `CuSimd<T, N>` SIMD Type | **Full** | Generic SIMD register type with named accessors (`x`/`y`/`z`/`w`), runtime and compile-time indexing, `to_array` conversion. |
| ABI Scalarization | **Full** | Slices are scalarized at kernel boundaries (`&[T]` -> `(ptr, len)`, reconstructed inside the function). Structs and closures pass by value as one byval `.param`; field flattening still applies on internal device-to-device calls. |

Array value constants support primitive leaves (integers, `f16`, `f32`,
`f64`), nested arrays, and tuples recursively composed from supported scalar,
enum, tuple, and zero-sized fields. Tuple element strides and field offsets
come from rustc layout, including internal and trailing padding; direct tuple
value constants use the same layout-aware decoder. Arrays whose elements are
structs or initialized unions are not yet materialized as constants.
Pointer-bearing array values are rejected until aggregate relocations can be
represented without losing provenance.

## Compiler: Closures

| Feature | Status | Description |
|:--------|:-------|:------------|
| Move Closures (`FnOnce`) | **Full** | Closures that capture by value. The whole closure struct is pushed as one byval kernel argument. `move \|x\| x * factor` pattern. |
| Reference Closures (`Fn`/`FnMut`) | **Full** | Non-move closures that capture by reference. The closure struct (containing host pointers) still travels as one byval argument; the GPU reads through those pointers via HMM. |
| Host-to-Device Closures | **Full** | Closures defined on host passed to generic kernels. Polynomial evaluation with captured coefficients tested. |
| Device-Internal Closures | **Full** | Closures created and used entirely on device, including closures passed to device functions. |

## Compiler: Control Flow

| Feature | Status | Description |
|:--------|:-------|:------------|
| Match Expressions (integer switch) | **Full** | Multi-way match on integers. Generates chain of conditional branches. |
| Match on Enums | **Full** | Pattern matching on `Option<T>` and custom enums. Discriminant extraction + payload access. |
| For Loops (range, iterator, enumerate) | **Full** | Full iterator desugaring: range-based, `slice.iter()`, `enumerate()`, nested loops, `break`, `continue`. |
| While Loops / If-Else | **Full** | Baseline control flow fully supported. |
| Break and Continue | **Full** | `break` and `continue` in for/while loops, including early exit. |
| Loop Unroll Annotations | **Partial** | `#[unroll]` and `#[unroll(N)]` request unrolling of explicit counted `while` loops. Nested loops and multiple `continue` paths work; full unrolling preserves `break` paths and multiple exit targets, while partial unrolling requires a positive-step `<`/`<=` loop with an invariant limit and only the normal header exit. Requests are capped at 1,024 copies, 8,192 cloned blocks, and 65,536 cloned operations. |

## Compiler: Arithmetic and Casting

| Feature | Status | Description |
|:--------|:-------|:------------|
| 64-bit Arithmetic | **Full** | Full 64-bit integer arithmetic including shifts, bitwise ops, and descriptor field packing. |
| Type Casting (all kinds) | **Full** | IntToInt, IntToFloat, FloatToInt, FloatToFloat, Transmute (bitcast), PtrToPtr, PtrToInt, IntToPtr, pointer coercions. |
| Packed bf16x2 FMA | **Full** | `bf16x2::fma_bf16x2(a, b, c)` lowers to PTX `fma.rn.bf16x2`, two bf16 lanes per `u32`. sm_80+. |

## Compiler: Interop

| Feature | Status | Description |
|:--------|:-------|:------------|
| Bi-directional LTOIR Support | **Full** | Rust kernels call CUDA C++ device functions **and** C++ calls Rust device functions. Via NVVM IR → libNVVM → LTOIR → nvJitLink. |
| Device FFI (`extern "C"`) | **Full** | `#[device] extern "C" { fn ... }` declarations for external LTOIR functions. CUB/CCCL integration demonstrated. |
| MathDx FFI (cuFFTDx / cuBLASDx) | **Full** | cuFFTDx (8/16/32-point thread-level FFT), cuBLASDx (32x32x32 block-level GEMM) via LTOIR. |
| Tile interop | **Experimental** | Inter-kernel interop works today: a [cutile-rs Tile kernel](https://github.com/NVlabs/cutile-rs) and a cuda-oxide SIMT PTX kernel can run in one host process on the same CUDA stream over shared device tensors. Intra-kernel Tile interop is work in progress and tracked in [#96](https://github.com/NVlabs/cuda-oxide/issues/96). |
| Cross-Crate Kernels | **Full** | Kernels and device functions defined in library crates with monomorphization at the binary crate use site. |

## Compiler: Functions

| Feature | Status | Description |
|:--------|:-------|:------------|
| `#[kernel]` Attribute | **Full** | Marks functions as GPU kernel entry points (`ptx_kernel` calling convention). Multiple kernels per file. |
| `#[device]` Helper Functions | **Full** | Device-side helper functions callable from kernels. `#[inline(always)]` is preserved as the LLVM `alwaysinline` attribute (emitted alongside the convergent group and any `!dbg` scope), so `opt` honors the inline intent. |
| Standalone `#[device]` Functions | **Full** | Device functions compiled without any kernel present. Clean export names for C++ consumption. |
| Multi-Kernel Modules | **Full** | Multiple `#[kernel]` functions in a single source file compile to a single PTX module. |

## Compiler: Compilation Pipeline

| Feature | Status | Description |
|:--------|:-------|:------------|
| Unified Single-Source Compilation | **Full** | Host and device code in the same file. Custom rustc codegen backend intercepts codegen. No `#[cfg]` needed. |
| PTX Output | **Full** | Default output: Rust MIR → `dialect-mir` → `mem2reg` → annotated loop unroll → LLVM dialect → LLVM IR → `llc` → PTX. Targets sm_80 through sm_100a. |
| NVVM IR Output | **Full** | Selects LLVM 7 typed-pointer syntax for pre-Blackwell GPUs and opaque-pointer syntax for Blackwell and newer GPUs. The generated module is verified by libNVVM, and unsupported legacy operations produce a compile error. |
| LTOIR Linking | **Full** | Device-side LTO via libNVVM and nvJitLink. |
| Float Math Intrinsics (libdevice) | **Full** | Rust `f32`/`f64` math methods (`sin`, `cos`, `exp`, `pow`, `sqrt`, ...) lower to CUDA libdevice (`__nv_*`) on pre-Blackwell and Blackwell GPUs. cuda-oxide selects the matching NVVM IR syntax automatically. On Blackwell, the runtime can also JIT PTX produced from a standard pre-Blackwell target such as `sm_86`. |
| Pipeline Inspection | **Full** | `cargo oxide pipeline <example>` shows imported and post-`mem2reg` MIR, LLVM dialect, exported LLVM IR, and PTX. |
| Compute Sanitizer Wrapper | **Full** | `cargo oxide sanitize <example>` builds the example and runs the host binary under NVIDIA Compute Sanitizer (`memcheck`, `racecheck`, `initcheck`, or `synccheck`). |
| cuda-gdb Source Debugging | **Full** | `cargo oxide debug` builds device debug information on the PTX path and launches `cuda-gdb`. Legacy NVVM IR does not yet support debug metadata. |
| cuda-gdb Local / Argument Inspection | **Partial** | `CUDA_OXIDE_DEBUG=full` is a `-G`-style build (optimization off, locals kept in memory) so `info args`/`info locals` show real values for scalars, pointers/references, and structs/tuples/arrays with their fields. Enums, ABI-split bare slices, closures, and projections (`x.0`) are not yet described. |

## Compiler: Inline PTX

| Feature | Status | Description |
|:--------|:-------|:------------|
| `ptx_asm!` Macro | **Partial** | CUDA inline PTX with `%0` operands, `in`, up to 8 `out` operands (each with an `=`-prefixed constraint; two or more `out`s return a tuple destructured into the output places in declaration order), up to 16 inputs, CUDA register constraints `h`, `r`, `l`, `q`, `f`, and `d`, immediate integer constraint `n`, `clobber("memory")`, and `options(register_only)` for pure register snippets. By default, snippets are treated as side-effecting and stay inside their current control flow. Use `options(register_only, may_diverge)` only for pure snippets that are safe to move across divergent control flow; **never** use it for `.sync` instructions or collectives. More than 8 outputs, read-write operands, and the `"C"` constraint are not implemented yet. |

---

## Runtime Library: Safety

| Feature | Status | Description |
|:--------|:-------|:------------|
| `DisjointSlice<T, IndexSpace>` | **Full** | Bounds-checked parallel write output slice. `IndexSpace` rejects mismatched layouts; uniqueness also requires matching prepared launch geometry (or a raw unsafe proof). |
| `ThreadIndex<'kernel, IndexSpace>` | **Full** | Opaque, non-transferable witness. `index_1d` uniqueness requires inactive Y/Z dimensions, proven by a `domain = 1` prepared launch or by the caller of a raw unsafe launch. |
| Proof-carrying static views | **Full** | A checked `u32` thread index proves one complete element or tile, then `at_const` accesses compile-time positions without another runtime bounds check. |
| `PreparedLaunch<K>` | **Full** | Checked, reusable launch geometry branded for the exact kernel. Raw `LaunchConfig` generated methods are unsafe. |
| `ManagedBarrier` Typestate | **Full** | Compile-time barrier lifecycle: `Uninit → Ready → Invalidated`. Invalid transitions are compile errors. |

## Runtime Library: Atomics

| Feature | Status | Description |
|:--------|:-------|:------------|
| Device-Scope Atomics | **Full** | `DeviceAtomic{U32,I32,U64,I64,F32,F64}` with `.gpu` scope. All 5 orderings. |
| Block-Scope Atomics | **Full** | `BlockAtomic{U32,I32,U64,I64,F32,F64}` with `.cta` scope. |
| System-Scope Atomics | **Full** | `SystemAtomic{U32,I32,U64,I64,F32,F64}` with `.sys` scope. For CPU-GPU shared data. |
| `core::sync::atomic` Support | **Full** | Standard library atomic types lowered to PTX `atom.sys` instructions. |

## Runtime Library: Shared Memory

| Feature | Status | Description |
|:--------|:-------|:------------|
| Static Shared Memory | **Full** | `SharedArray<T, N, ALIGN>` — compile-time sized, block-scoped. Optional alignment up to 256B. |
| Dynamic Shared Memory | **Full** | `DynamicSharedArray<T, ALIGN>` — runtime-sized, set via `LaunchConfig::shared_mem_bytes`. |
| Distributed Shared Memory (DSMEM) | **Full** | Direct access to other blocks' shared memory within a cluster. `map_shared_rank()` for address mapping. sm_90+. |

## Runtime Library: Thread and Synchronization

| Feature | Status | Description |
|:--------|:-------|:------------|
| Thread/Block/Grid Intrinsics | **Full** | `threadIdx`, `blockIdx`, `blockDim`, `gridDim`. Index witnesses are layout-typed; their uniqueness also depends on matching launch dimensionality. `index_2d_runtime(s)` adds a caller-proved stride. See [The Safety Model](../gpu-safety/the-safety-model.md). |
| Block Synchronization | **Full** | `sync_threads()` — thread block barrier. |
| Async Barriers (mbarrier) | **Full** | Hardware async barriers for Hopper+: init, arrive, test_wait, try_wait, inval. |
| Cluster Synchronization | **Full** | `cluster_sync()` for all blocks in a cluster. sm_90+. |
| Fence Operations | **Full** | `fence_proxy_async_shared_cta()` for TMA visibility, `nanosleep(ns)`. |

## Runtime Library: Warp

| Feature | Status | Description |
|:--------|:-------|:------------|
| Warp Shuffle Operations | **Full** | `shuffle`, `shuffle_xor`, `shuffle_down`, `shuffle_up` for `i32` and `f32`. |
| Warp Vote Operations | **Full** | `all(pred)`, `any(pred)`, `ballot(pred)` → bitmask. |
| Lane/Warp ID | **Full** | `lane_id()` (0–31), `warp_id()`. Direct register reads. |

## Runtime Library: Cooperative Groups

| Feature | Status | Description |
|:--------|:-------|:------------|
| Typed Group Handles | **Full** | `Grid`, `Cluster`, `ThreadBlock`, `WarpTile<N>` (N ∈ {1,2,4,8,16,32}), `CoalescedThreads`. |
| Group Universal API | **Full** | `size()`, `thread_rank()`, `sync()` on every group handle. |
| Warp Tile Partitioning | **Full** | `ThreadBlock::tiled_partition::<N>()` carves a sub-warp `WarpTile<N>`. `coalesced_threads()` materialises the active-lane group. |
| Warp Collectives | **Full** | `ballot`, `all`, `any`, `shfl`, `shfl_xor`, `shfl_down`, `shfl_up` (`i32` and `f32`); `match_any` / `match_all` (`i32` and `i64`); `active_mask`. |
| Warp Reductions / Scans | **Full** | `warp_reduce`, `warp_scan` (inclusive). `Sum`/`Min`/`Max` for `u32`/`i32`/`f32`; `BitAnd`/`BitOr`/`BitXor` for `u32`. |
| Block Reductions / Scans | **Full** | `block_reduce`, `block_scan` (inclusive). Const-generic over `NUM_WARPS`; same op/type matrix as warp variants; uses `__shared__` scratch. |
| Cooperative Kernel Launch | **Full** | `#[cooperative_launch]` on a `#[cuda_module]` kernel (or `unsafe { cuda_launch! { cooperative: true, ... } }`) enables `Grid::sync()` for grid-wide barriers. |

## Runtime Library: Debug

| Feature | Status | Description |
|:--------|:-------|:------------|
| `gpu_printf!` Macro | **Full** | Formatted GPU output with full format specifier support. Lowers to `vprintf`. |
| `gpu_assert!` Macro | **Full** | Runtime GPU assertion. Calls `trap()` if condition is false. |
| Debug Intrinsics | **Full** | `clock()`, `clock64()`, `trap()`, `breakpoint()`, `prof_trigger::<N>()`. |

## Runtime Library: Kernel Launch

| Feature | Status | Description |
|:--------|:-------|:------------|
| `#[cuda_module]` Typed Launch | **Full** | Embedded module loading with typed sync/async arguments. Raw configuration methods are unsafe. |
| `#[launch_contract]` / `PreparedLaunch<K>` | **Full** | Checked dimensionality, exact block shape, resources, capabilities, context, and kernel identity. |
| `cuda_launch!` Macro | **Full** | Unsafe lower-level launch for runtime-loaded modules; requires `unsafe { }`. |
| `cuda_launch_async!` Macro | **Full** | Unsafe lower-level lazy launch; requires `unsafe { }`. |
| `#[launch_bounds]` | **Full** | Occupancy hints: max threads per block, min blocks per SM. |
| `#[cluster_launch]` | **Full** | Compile-time cluster dimensions. Emits `.reqnctapercluster` in PTX. |

## Runtime Library: TMA

| Feature | Status | Description |
|:--------|:-------|:------------|
| TMA Bulk Tensor Copy (1D–5D) | **Full** | `cp_async_bulk_tensor_{1..5}d_g2s`. 128-byte TMA descriptors. sm_90+. |
| TMA Multicast | **Full** | Single TMA load broadcast to all CTAs in cluster. sm_100a for full multicast. |
| TMA Commit/Wait Groups | **Full** | `cp_async_bulk_commit_group`, `cp_async_bulk_wait_group` for async completion tracking. |

---

## Not Yet Implemented

| Feature | Status | Notes |
|:--------|:-------|:------|
| Rust `asm!` macro | **Planned** | Use `ptx_asm!` for CUDA inline PTX. Direct lowering of Rust MIR `InlineAsm` is not implemented. |
| FP8 / MX Data Types | **Planned** | Roadmap item for Blackwell. No architectural limitation. |
| Dynamic Dispatch (`dyn Trait`) | **N/A** | Use generics with static dispatch. Haven't found a real need for this. |
| Heap Allocation (`Box`, `Vec`) | **N/A** | CUDA has a device-side heap (`malloc`/`free` in kernels), and the compiler allows the `alloc` crate through -- but no device-side `#[global_allocator]` is wired up today. Even if it were, device `malloc` is extremely slow (serialized, fragmented, uncoalesced). Use slices and `SharedArray`. |
| `String` / `format_args!` | **N/A** | Use `gpu_printf!` for formatted output. |
| Panic / Unwinding | **N/A** | Panic paths exist in MIR but the compiler strips `core::panicking::*` and all unwind edges. The GPU hardware *can* support unwinding (absolute branches + per-thread call stack tracking post-Volta), but the CUDA toolchain (nvcc/ptxas) doesn't expose it today -- no landing pads survive to PTX. If a panic path is reached at runtime the GPU traps (same as `panic=abort`). NVIDIA has an active project to add C++ exception support to CUDA for automotive safety; the current cuda-oxide design is forward-compatible with that work. Use `gpu_assert!()` + `trap()` for explicit runtime checks today. |
| Standard Library (`std`/`alloc`) | **N/A** | `std` is forbidden. `alloc` is allowed by the collector but has no backing allocator. Only `core` is fully functional. `Option`, `Result`, iterators all work. |
| Texture Memory | **N/A** | Lower priority given TMA availability on Hopper+. |
