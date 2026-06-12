# dialect-mir

A [pliron](https://github.com/vaivaswatha/pliron) dialect that represents Rust's Mid-level Intermediate Representation (MIR). This is the first IR in the cuda-oxide pipeline -- `mir-importer` translates rustc's MIR into this dialect, then `mir-lower` lowers it to the LLVM dialect for PTX generation.

```text
rustc MIR ──► mir-importer ──► dialect-mir ──► mir-lower ──► LLVM dialect ──► LLVM IR ──► PTX
```

## Types

The dialect defines seven types that preserve Rust-level semantics:

| Type                  | Description                                        | Example                               |
|-----------------------|----------------------------------------------------|---------------------------------------|
| `MirTupleType`        | Heterogeneous tuples                               | `mir.tuple<i32, f32, i64>`            |
| `MirPtrType`          | Pointers with address space and mutability         | `mir.ptr<f32, mutable, addrspace: 3>` |
| `MirSliceType`        | Fat pointers (`&[T]` = ptr + len)                  | `mir.slice<f32, addrspace: 1>`        |
| `MirDisjointSliceType`| `DisjointSlice<T>` -- per-thread unique access     | `mir.disjoint_slice<f32, ...>`        |
| `MirStructType`       | Named structs with layout metadata                 | `mir.struct<"Point", [f32, f32]>`     |
| `MirEnumType`         | Rust enums with discriminant + variant payloads    | `mir.enum<"Ordering", i8, ...>`       |
| `MirArrayType`        | Fixed-size arrays                                  | `mir.array<f32, 256>`                 |

`MirEnumType` records the enum's layout the way rustc computed it: the
tag's type (width and signedness), the variant names, the declared
discriminant VALUES (not variant positions), per-variant field counts,
the field types, where each field and the tag live (byte positions),
and total size / alignment in bytes (0 = layout not recorded). In
textual order:

```text
mir.enum<"Ordering", si8, ["Less", "Equal", "Greater"], [255, 0, 1], [0, 0, 0], [], [], 0, 1, 1>
```

`Ordering::Less` is declared as -1, stored as the unsigned i8 bit
pattern 255. The tag slot of a lowered enum always holds these declared
values; using variant indices instead made `Ordering::Less` match the
`Equal` arm (issue #146).

### Address Spaces

Pointers and slices carry an NVPTX address space:

| Space      | ID | PTX Qualifier | Use                           |
|------------|----|---------------|-------------------------------|
| Generic    | 0  | (none)        | Default, resolved at runtime  |
| Global     | 1  | `.global`     | Device VRAM                   |
| Shared     | 3  | `.shared`     | Per-block scratchpad          |
| Constant   | 4  | `.const`      | Read-only cached              |
| Local      | 5  | `.local`      | Per-thread stack/spill        |
| TensorMem  | 6  | `.param`      | Blackwell+ tcgen05 operands   |

## Operations

54 operations across 11 modules:

| Module         | Ops | Description                                                                             |
|----------------|-----|-----------------------------------------------------------------------------------------|
| `function`     | 1   | `MirFuncOp` -- function definition                                                      |
| `control_flow` | 5   | return, goto, cond_branch, assert, unreachable                                          |
| `memory`       | 9   | alloca, load, store, ref, assign, ptr_offset, shared_alloc, global_alloc, extern_shared |
| `constants`    | 3   | integer, float, and undef constants                                                     |
| `arithmetic`   | 15  | add/sub/mul/div/rem, checked variants, bitwise, shifts                                  |
| `comparison`   | 6   | lt, le, gt, ge, eq, ne                                                                  |
| `aggregate`    | 8   | construct/extract/insert for structs, tuples, and arrays; field and element address     |
| `enum_ops`     | 3   | construct_enum, get_discriminant, enum_payload                                          |
| `cast`         | 1   | type conversions (kind tracked via `MirCastKindAttr`)                                   |
| `storage`      | 2   | storage_live, storage_dead (lifetime markers)                                           |
| `call`         | 1   | function calls                                                                          |

`MirAllocaOp` implements `PromotableAllocationInterface` and `MirLoadOp` / `MirStoreOp` implement `PromotableOpInterface`, so pliron's `mem2reg` pass can promote scalar stack slots back into SSA. `MirUndefOp` is the default reaching definition the pass materialises when a load is not dominated by any store.

## Verification

Every operation implements pliron's `Verify` trait to catch bugs early during the import phase:

| Category     | What's Checked                                             |
|--------------|------------------------------------------------------------|
| Function     | Entry block args match function signature                  |
| Control flow | Condition is `i1`, successor block args match              |
| Memory       | Pointer types, pointee types, address spaces consistent    |
| Arithmetic   | Operands same type, result type matches                    |
| Comparison   | Operands same type, result is `i1`                         |
| Aggregate    | Struct/tuple types, index within bounds, element types     |
| Enum         | Discriminant type valid, payload types match variant       |
| Cast         | Cast kind attribute present (full validation at lowering)  |
| Constants    | Type attribute present and well-formed                     |
| Call         | Callee exists, argument count and types match              |

This catches mismatches immediately after `mir-importer` translates from rustc, rather than deferring errors to LLVM.

## Attributes

The dialect defines four domain-specific attribute types (following the pliron best practice of avoiding overloaded `IntegerAttr`):

| Attribute           | Rust Type          | Description                                                                                                          |
|---------------------|--------------------|----------------------------------------------------------------------------------------------------------------------|
| `mir.cast_kind`     | `MirCastKindAttr`  | Preserves Rust cast intent (e.g. `IntToFloat`, `PtrToPtr`, `Transmute`) so lowering picks the right LLVM instruction |
| `mir.mutability`    | `MutabilityAttr`   | Boolean: `&` vs `&mut` for `mir.ref`                                                                                 |
| `mir.field_index`   | `FieldIndexAttr`   | Structural field index for `extract_field`, `insert_field`, `field_addr`, `enum_payload`                             |
| `mir.variant_index` | `VariantIndexAttr` | Enum variant index for `construct_enum`, `enum_payload`                                                              |

## Registration

```rust
use pliron::context::Context;
use dialect_mir::register;

let mut ctx = Context::new();
register(&mut ctx);  // Registers all ops, types, and attributes
```

## Source Layout

```text
src/
├── lib.rs                       # Dialect registration
├── types.rs                     # 7 MIR types + address_space constants
├── attributes.rs                # 4 domain-specific attributes
├── ops/
│   ├── mod.rs                   # Op module registry + re-exports
│   ├── function.rs              # MirFuncOp
│   ├── control_flow.rs          # Terminators and branches
│   ├── memory.rs                # Load, store, alloc, shared memory
│   ├── constants.rs             # Integer and float literals
│   ├── arithmetic.rs            # Math, bitwise, shifts, checked ops
│   ├── comparison.rs            # Relational and equality
│   ├── aggregate.rs             # Struct, tuple, array manipulation
│   ├── enum_ops.rs              # Enum construction and inspection
│   ├── cast.rs                  # Type conversions
│   ├── storage.rs               # Lifetime markers
│   └── call.rs                  # Function calls
```

## Further Reading

- [llvm-export](../llvm-export/) -- pliron-llvm shim + textual `.ll` exporter (lowering target)
- [dialect-nvvm](../dialect-nvvm/) -- NVVM GPU intrinsics
- [mir-importer](../mir-importer/) -- translates rustc MIR → `dialect-mir`
- [mir-lower](../mir-lower/) -- lowers `dialect-mir` → LLVM dialect
