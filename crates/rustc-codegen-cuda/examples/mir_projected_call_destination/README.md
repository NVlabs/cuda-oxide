# mir_projected_call_destination

An intrinsic call whose destination carries a projection, in the three shapes
that reach code generation.

## What it pins

rustc lowers an ordinary call with a projected destination into a call to a
temporary followed by a store:

```text
_9 = f(const 10_i32) -> [return: bb3, unwind continue]
(*_8) = move _9
```

An intrinsic keeps its destination instead, so the projection survives to the
importer. Surface Rust cannot write that, which is why all three bodies here
are `#[custom_mir]`:

| body | destination | shape |
|---|---|---|
| `through_deref` | `(*p) = bswap(x)` | dereferenced raw pointer |
| `through_field` | `RET.1 = bswap(x)` | field of a `(f64, u8)` tuple |
| `through_index` | `RET[i] = bswap(x)` | element of a `[i32; 3]` |

Each result has to land at the address the projection names. The device and
the host run the same bodies and their results are compared, so a store aimed
at the local instead of the place shows up as a disagreement rather than as a
value nobody checks. The array case leaves its other two elements at `11` and
`33`, which a store over the whole array would not.

## Running it

```bash
cargo oxide run mir_projected_call_destination
```

Before the importer took the projection into account, the field case asked for
a cast from a byte to `{ double, i8, [7 x i8] }` and the deref case asked for
the width of a pointer, so this example did not compile.
