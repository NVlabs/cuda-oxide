# device_global

Tests ordinary Rust `static mut` values in CUDA global memory and non-zero
immutable Rust static tables.

Run with:

```bash
cargo oxide run device_global
```

The first kernel updates two ordinary device statics:

```rust
static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MARKER: u32 = 0;
```

The second kernel reads a non-zero immutable static table through a flattened
pointer, matching generated coefficient-table access patterns:

```rust
static STATIC_WEIGHTS: [[f32; 2]; 4] = [[0.25, 0.5], ...];
fn get_static_weights() -> &'static [[f32; 2]; 4] { &STATIC_WEIGHTS }
let weights = get_static_weights();
let pair = load_pair(&weights[0][0], 2);
```

Expected behavior:

| Static kind                 | Memory space       |
|----------------------------|--------------------|
| Ordinary `static mut`      | Global `addrspace(1)` |
| `SharedArray` / `Barrier`  | Shared `addrspace(3)` |
| `DynamicSharedArray::get()`| Shared `addrspace(3)` |

The example launches the kernel twice. `DEVICE_COUNTER` should persist across
launches, proving it is global device storage and not per-block shared memory.

Non-zero immutable static initializers are emitted into the generated LLVM/PTX
global so device code can read compile-time coefficient tables directly.
