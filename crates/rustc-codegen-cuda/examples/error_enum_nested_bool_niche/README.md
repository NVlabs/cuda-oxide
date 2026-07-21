# `error_enum_nested_bool_niche`

Negative test for a niche carrier inside an aggregate payload:

```rust
#[repr(C)]
struct Wrapper {
    pad: u32,
    flag: bool,
}

type MaybeWrapper = Option<Wrapper>;
```

Rust stores `MaybeWrapper` in eight bytes and uses the `bool` byte at offset 4
as the niche carrier. An LLVM `bool` is only `i1`, however, so storing the
whole lowered wrapper and then reading that byte as `i8` does not prove that
the upper seven bits are zero.

Until cuda-oxide recursively materializes the physical byte for nested bools,
it must fail closed:

```bash
cargo oxide build error_enum_nested_bool_niche
```

Expected diagnostic:

```text
contains nested bool/i1 storage
```
