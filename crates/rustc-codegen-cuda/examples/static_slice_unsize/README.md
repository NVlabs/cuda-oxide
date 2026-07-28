# static_slice_unsize

Positive test for zero-addend device-static array→slice unsize:

```rust
static TABLE: [f32; 4] = [0.25, 0.5, 1.0, 2.0];
const TABLE_SLICE: &[f32] = &TABLE;
```

`&TABLE` is `&[f32; 4]`. The unsize coercion to `&[f32]` keeps a zero addend
and adds length metadata. cuda-oxide materializes a fat pointer via
`mir.construct_slice` (thin global pointer + array length).

Interior addends such as `&TABLE[2] as &[f32]` remain unsupported
(`error_static_slice_addend`).

```bash
cargo oxide run static_slice_unsize
```
