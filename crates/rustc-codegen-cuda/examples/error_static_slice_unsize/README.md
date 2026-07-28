# error_static_slice_unsize

Negative test for a zero-addend unsize coercion of a device static to a
slice:

```rust
static TABLE: [f32; 4] = [0.25, 0.5, 1.0, 2.0];
const TABLE_SLICE: &[f32] = &TABLE;
```

`&TABLE` is `&[f32; 4]`, and the unsize coercion to `&[f32]` points at the
full static (byte addend 0) while adding a length. Static constants are
materialized as a thin pointer, which cannot carry the length half of a
fat pointer. Emitting the constant anyway would type a thin pointer as a
slice carrier and misread the fat-pointer layout downstream, so cuda-oxide
must stop with a clear error instead of an ill-typed thin-to-fat cast.

This is the zero-addend sibling of `error_static_slice_addend`, which pins
the same rejection for an interior pointer (`&TABLE[2]`, addend 16).

Run:

```bash
cargo oxide build error_static_slice_unsize
```

The build must fail with a message similar to:

```text
constant pointer to device static error_static_slice_unsize::TABLE has
pointee type [f32], but the full static has type [f32; 4]; zero-addend
pointee reinterpretations and unsized coercions are not supported
```
