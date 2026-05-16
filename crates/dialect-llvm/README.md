# dialect-llvm

A [pliron](https://github.com/vaivaswatha/pliron) dialect for LLVM IR, targeting the NVPTX backend. `mir-lower` lowers `dialect-mir` into this dialect, which is then exported to textual LLVM IR (`.ll`) for PTX generation via `llc`.

```text
dialect-mir ──► mir-lower ──► dialect-llvm ──► export ──► .ll file ──► llc ──► .ptx
```

## Types

| Type          | Description                         | LLVM Syntax                               |
|---------------|-------------------------------------|-------------------------------------------|
| `StructType`  | Named or anonymous structs          | `{ i32, float }`, `%Point = type { ... }` |
| `PointerType` | Opaque pointers with address space  | `ptr`, `ptr addrspace(3)`                 |
| `ArrayType`   | Fixed-size arrays                   | `[256 x float]`                           |
| `VectorType`  | SIMD vectors                        | `<4 x float>`                             |
| `FuncType`    | Function signatures                 | `(i32, ptr) -> void`                      |
| `VoidType`    | Void (no return value)              | `void`                                    |

### Address Spaces

| Space      | ID | Description                       |
|------------|----|-----------------------------------|
| Generic    | 0  | Default, resolved at runtime      |
| Global     | 1  | Device VRAM                       |
| Shared     | 3  | Per-block scratchpad              |
| Constant   | 4  | Read-only cached                  |
| Local      | 5  | Per-thread stack/spill            |
| TensorMem  | 6  | Blackwell+ tcgen05 operands       |

## Operations

63 operations across 13 modules:

| Module         | Ops | Description                                                                                                                   |
|----------------|-----|-------------------------------------------------------------------------------------------------------------------------------|
| `arithmetic`   | 19  | Integer (add, sub, mul, sdiv, udiv, srem, urem, and, or, xor, shl, lshr, ashr) and float (fadd, fsub, fmul, fdiv, frem, fneg) |
| `atomic`       | 5   | atomic load, store, RMW, cmpxchg, fence                                                                                       |
| `comparison`   | 2   | `icmp` (integer) and `fcmp` (float)                                                                                           |
| `cast`         | 13  | sext, zext, trunc, fpext, fptrunc, sitofp, uitofp, fptosi, fptoui, bitcast, ptrtoint, inttoptr, addrspacecast                 |
| `memory`       | 4   | alloca, load, store, getelementptr                                                                                            |
| `control_flow` | 5   | br, cond_br, switch, ret, unreachable                                                                                         |
| `aggregate`    | 3   | extractvalue, insertvalue, extractelement                                                                                     |
| `constants`    | 3   | constant, zeroinitializer, undef                                                                                              |
| `symbol`       | 3   | func, global, addressof                                                                                                       |
| `call`         | 2   | call, call_intrinsic                                                                                                          |
| `select`       | 1   | conditional selection                                                                                                         |
| `asm`          | 2   | inline assembly (single-result and multi-result)                                                                              |
| `va`           | 1   | variadic argument handling                                                                                                    |

## Verification

All operations implement comprehensive verification via pliron's `Verify` trait:

| Category     | What's Checked                                                        |
|--------------|-----------------------------------------------------------------------|
| Arithmetic   | Both operands same type, result matches, integer-only for bitwise ops |
| Comparison   | Predicate attribute valid, operands match, result is `i1`             |
| Cast         | Source/target width relationships (e.g. sext requires wider target)   |
| Memory       | Pointer types, element types, GEP index validity                      |
| Control flow | Successor blocks valid, condition is `i1`, switch cases               |
| Aggregate    | Index within bounds, element type matches                             |
| Symbol       | Function type consistent, entry block args match signature            |
| Call         | Argument count and types match callee signature                       |
| Atomic       | Ordering and scope attributes valid, pointer operand                  |

This catches lowering bugs before the LLVM IR export phase.

## Attributes

| Attribute                       | Description                                     |
|---------------------------------|-------------------------------------------------|
| `IntegerOverflowFlagsAttr`      | `nsw` / `nuw` flags on integer arithmetic       |
| `FastmathFlagsAttr`             | Fast-math flags (`nnan`, `ninf`, `nsz`, etc.)   |
| `ICmpPredicateAttr`             | Integer comparison predicate                    |
| `FCmpPredicateAttr`             | Float comparison predicate                      |
| `GepIndicesAttr`                | GEP index list (constant or operand per index)  |
| `InsertExtractValueIndicesAttr` | Indices for `insertvalue` / `extractvalue`      |
| `CaseValuesAttr`                | Switch case values                              |
| `LinkageAttr`                   | Linkage kind for globals and functions          |
| `LlvmAtomicOrdering`            | Atomic memory ordering                          |
| `LlvmSyncScope`                 | Synchronization scope for atomics               |
| `LlvmAtomicRmwKind`             | Atomic RMW operation kind                       |

## Export to LLVM IR

The `export` module serializes the dialect-llvm module to textual LLVM IR. Two backend configurations control the output format:

| Configuration      | Use Case    | `@llvm.used` | `!nvvmir.version` | `!nvvm.annotations`    |
|--------------------|-------------|--------------|-------------------|------------------------|
| `PtxExportConfig`  | PTX via llc | No           | No                | Launch bounds only     |
| `NvvmExportConfig` | NVVM IR     | Yes          | Yes               | All kernels            |

```rust
use dialect_llvm::export::{export_module_to_string, export_module_to_string_with_config, NvvmExportConfig};

// Default (PTX path)
let ll = export_module_to_string(&ctx, &module)?;

// NVVM IR path (for libNVVM / LTOIR)
let nvvm_ir = export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig)?;
```

The export handles block-arg to PHI-node translation, grouped intrinsic declarations, `convergent` attribute on synchronization ops, kernel metadata (`!nvvm.annotations`), launch bounds, cluster config, and device extern FFI declarations.

### Target Configuration

| Setting        | Value                                    |
|----------------|------------------------------------------|
| Target triple  | `nvptx64-nvidia-cuda`                    |
| Data layout    | 64-bit pointers, 128-bit i128 alignment  |
| PTX version    | 8.7+ (for sm_120)                        |

## Registration

```rust
use pliron::context::Context;
use dialect_llvm::register;

let mut ctx = Context::new();
register(&mut ctx);  // Registers all ops, types, and attributes
```

## Source Layout

```text
src/
├── lib.rs              # Dialect registration
├── types.rs            # 6 LLVM types + address_space constants
├── attributes.rs       # 11 attribute types
├── op_interfaces.rs    # Shared op interfaces
├── export/             # LLVM IR text export
│   ├── mod.rs          # Public API + function/block/op emission
│   ├── config.rs       # Backend configuration
│   ├── externs.rs      # Device extern declarations
│   ├── literals.rs     # Constant literal formatting
│   ├── metadata.rs     # NVVM metadata emission
│   ├── names.rs        # Exported symbol naming
│   └── type_printing.rs # LLVM type printing
└── ops/
    ├── mod.rs           # Op module registry
    ├── arithmetic.rs    # Integer and float binary ops
    ├── atomic.rs        # Atomic operations
    ├── comparison.rs    # icmp, fcmp
    ├── cast.rs          # 13 cast operations
    ├── memory.rs        # alloca, load, store, GEP
    ├── control_flow.rs  # Terminators and branches
    ├── aggregate.rs     # extractvalue, insertvalue
    ├── constants.rs     # Constant values
    ├── symbol.rs        # Functions, globals, addressof
    ├── call.rs          # call, call_intrinsic
    ├── select.rs        # Conditional selection
    ├── asm.rs           # Inline assembly
    └── va.rs            # Variadic args
```

## Further Reading

- [dialect-mir](../dialect-mir/) -- pliron dialect modelling Rust MIR (lowering source)
- [dialect-nvvm](../dialect-nvvm/) -- NVVM GPU intrinsics
- [mir-lower](../mir-lower/) -- lowers `dialect-mir` → `dialect-llvm`
