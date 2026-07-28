# error_static_slice_addend

Negative test for an interior pointer into a device static whose pointee is
unsized:

```rust
static TABLE: [[f32; 2]; 4] = [[0.25, 0.5], [1.0, 2.0], [4.0, 8.0], [16.0, 32.0]];
const PAIR_SLICE: &'static [f32] = &TABLE[2];
```

`&TABLE[2]` is `&[f32; 2]`, and the unsize coercion to `&[f32]` keeps the
16-byte addend that selects element 2 while adding a length. Interior
pointers into device statics are materialized as a thin pointer (base
address plus byte addend), which cannot carry the length half of a fat
pointer. Emitting the constant anyway would type a thin pointer as a slice
carrier and misread the fat-pointer layout downstream, so cuda-oxide must
stop with a clear error.

Run:

```bash
cargo oxide build error_static_slice_addend
```

The build must fail with a message similar to:

```text
constant pointer into device static error_static_slice_addend::TABLE has
byte offset 16, but pointee type [f32] is unsized; cuda-oxide does not
yet preserve the fat-pointer metadata an interior slice or DST pointer
needs
```
