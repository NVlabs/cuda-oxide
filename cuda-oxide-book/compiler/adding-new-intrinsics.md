# Adding New Intrinsics

So you want to teach cuda-oxide a new GPU trick. Maybe NVIDIA just shipped a new
instruction, or you need an existing PTX operation that nobody has wired up yet.
Good news: the process is mechanical. Five crates, five steps, and roughly thirty
minutes once you have done it before.

This chapter walks through the full pipeline using two real examples -- one
trivially simple, one with a few twists -- so you can see exactly what happens at
each stage.

---

## The Five-Stage Pipeline

Every GPU intrinsic follows the same path through the compiler:

```text
cuda-device          User writes:  thread::threadIdx_x()
    │
    ▼
mir-importer       Compiler sees the call, emits a `dialect-nvvm` op
    │
    ▼
dialect-nvvm       The op lives here as a verified IR node
    │
    ▼
mir-lower          Converts the `dialect-nvvm` op into an LLVM dialect op
    │
    ▼
llvm-export        Exports textual LLVM IR  →  llc turns it into PTX
```

At the `mir-lower` stage you pick one of two strategies:

| Strategy            | When to use                                                                  | Examples                               |
| :------------------ | :--------------------------------------------------------------------------- | :------------------------------------- |
| LLVM intrinsic call | LLVM already has a built-in for it                                           | `threadIdx_x`, warp shuffles, barriers |
| Inline PTX assembly | No LLVM intrinsic exists, or you need exact control over the PTX instruction | `trap`, `wgmma`, `tcgen05`, `mbarrier` |

Both strategies are demonstrated below.

---

## Example 1: `threadIdx_x` (the Simple Case)

`threadIdx_x()` is the "Hello, World" of GPU intrinsics: zero arguments, one
`u32` result, maps directly to a single LLVM NVVM intrinsic. If you can follow
this example, you can add any simple intrinsic.

### Stage 1 -- Declare in `cuda-device`

**File:** `crates/cuda-device/src/thread.rs`

```rust
#[inline(never)]
pub fn threadIdx_x() -> u32 {
    unreachable!("threadIdx_x called outside CUDA kernel context")
}
```

Two rules that might look odd until you understand the trick:

- **`#[inline(never)]`** keeps the function visible as a distinct call in MIR.
  If rustc inlined it, the compiler would see `unreachable!()` instead of a
  call it can intercept. That would be... less than helpful.

- **The body is `unreachable!()`** because this function never actually runs.
  The compiler replaces the entire call with a `dialect-nvvm` operation
  before any GPU code executes. Think of the function as a placeholder -- a
  "dear compiler, please insert a GPU instruction here" note.

```{tip}
Document what LLVM IR or PTX the intrinsic maps to in a comment above the
function. Future you (or future contributors) will thank present you.
```

### Stage 2 -- Define the `dialect-nvvm` Op

**File:** `crates/dialect-nvvm/src/ops/thread.rs`

```rust
#[pliron_op(name = "nvvm.read_ptx_sreg_tid_x", dialect = "nvvm", format)]
pub struct ReadPtxSregTidXOp;

impl Verify for ReadPtxSregTidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 {
            return verify_err!(op.loc(), "expected 0 operands");
        }
        if op.get_num_results() != 1 {
            return verify_err!(op.loc(), "expected 1 result");
        }
        Ok(())
    }
}
```

Then register it so pliron knows the op exists:

```rust
pub(super) fn register(ctx: &mut Context) {
    ReadPtxSregTidXOp::register(ctx, ReadPtxSregTidXOp::parser_fn);
}
```

The `Verify` trait catches structural bugs early. If something accidentally
creates this op with two operands, verification fails with a clear message
instead of producing garbage PTX three stages later.

### Stage 3 -- Recognize in `mir-importer`

**File:** `crates/mir-importer/src/translator/terminator/mod.rs`

When the translator processes MIR, every function call passes through
`try_dispatch_intrinsic()`. The callee's fully qualified domain name (FQDN)
comes from `CrateDef::name()` via `extract_func_info()`, producing paths like
`cuda_device::thread::threadIdx_x`. The match checks this FQDN:

```rust
match name {
    "cuda_device::threadIdx_x" | "cuda_device::thread::threadIdx_x" => {
        Ok(Some(helpers::emit_nvvm_intrinsic(
            ctx,
            ReadPtxSregTidXOp::get_concrete_op_info(),
            destination, target, block_ptr, prev_op,
            value_map, block_map, loc,
        )?))
    }
    // ... hundreds of other intrinsics ...
}
```

We match both the re-exported name (`cuda_device::threadIdx_x`) and the full
module path (`cuda_device::thread::threadIdx_x`) so both work regardless of how
the user imports the function. The FQDN is used as-is for matching -- no `::` to
`__` conversion happens before the intrinsic check.

The `emit_nvvm_intrinsic()` helper is a generic function that works for *any*
zero-argument, single-result NVVM intrinsic. It creates the operation, stores
the result in the value map, and emits a branch to the next basic block. For
simple intrinsics, you never need to write a custom emitter.

### Stage 4 -- Lower to the LLVM dialect

**File:** `crates/mir-lower/src/convert/intrinsics/basic.rs`

The op implements the `MirToLlvmConversion` op interface. In
`convert/interface_impls.rs`, the impl dispatches to a converter function:

```rust
impl MirToLlvmConversion for ReadPtxSregTidXOp {
    fn rewrite(ctx, rewriter, op, operands_info) -> Result<()> {
        basic::convert_read_tid_x(ctx, rewriter, op, operands_info)
    }
}
```

Then `basic::convert_read_tid_x` emits the LLVM intrinsic call:

```rust
pub fn convert_read_tid_x(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let intrinsic_name = "llvm_nvvm_read_ptx_sreg_tid_x";
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let func_ty = FuncType::get(ctx, i32_ty.into(), vec![], false);

    let call_op = call_intrinsic(ctx, rewriter, intrinsic_name, func_ty, vec![])?;
    rewriter.replace_operation_with_values(ctx, op, vec![result]);
    Ok(())
}
```

```{note}
LLVM intrinsic names use dots (`llvm.nvvm.read.ptx.sreg.tid.x`), but pliron
identifiers cannot contain dots. Internally we use underscores
(`llvm_nvvm_read_ptx_sreg_tid_x`). The export stage converts them back.
```

### Stage 5 -- Export (Nothing to Change)

**File:** `crates/llvm-export/src/export/` (the export module)

The `CallOp` exporter already handles the underscore-to-dot conversion:

```rust
let fixed_name = if name.starts_with("llvm_nvvm") {
    name.replace('_', ".")
} else {
    strip_device_prefix(&name)
};
```

Since `threadIdx_x` is *not* convergent (it is a per-thread register read,
not a collective operation), there is nothing to add to
`is_convergent_intrinsic()`.

**Final output:**

```llvm
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%v5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
```

After `llc`:

```text
mov.u32  %r1, %tid.x;
```

Five stages, one PTX instruction. Not bad.

---

## Example 2: `shuffle_xor` (the Complex Case)

Warp shuffles are more interesting. The user passes two arguments, but the
underlying LLVM intrinsic expects *four*. The compiler quietly fills in the
extras. The operation is also **convergent**, meaning LLVM must not move,
duplicate, or speculate it across control flow.

### Stage 1 -- Declare in `cuda-device`

**File:** `crates/cuda-device/src/warp.rs`

```rust
#[inline(never)]
pub fn shuffle_xor(var: u32, lane_mask: u32) -> u32 {
    let _ = (var, lane_mask);
    unreachable!("shuffle_xor called outside CUDA kernel context")
}
```

Same pattern as before. The `let _ = (var, lane_mask);` suppresses
unused-variable warnings -- a small courtesy that costs nothing.

### Stage 2 -- Define the `dialect-nvvm` Op

**File:** `crates/dialect-nvvm/src/ops/warp.rs`

```rust
#[pliron_op(name = "nvvm.shfl_sync_bfly_i32", dialect = "nvvm", format)]
pub struct ShflSyncBflyI32Op;

impl Verify for ShflSyncBflyI32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 {
            return verify_err!(op.loc(), "expected 2 operands");
        }
        if op.get_num_results() != 1 {
            return verify_err!(op.loc(), "expected 1 result");
        }
        Ok(())
    }
}
```

Two operands this time (value and lane mask), one result.

### Stage 3 -- Recognize in `mir-importer`

**File:** `crates/mir-importer/src/translator/terminator/mod.rs`

Unlike `threadIdx_x`, this calls a **specialized emitter** because it needs to
handle the user's two arguments:

```rust
"cuda_device::warp::shuffle_xor" => Ok(Some(
    intrinsics::warp::emit_warp_shuffle_i32(
        ctx, body,
        ShflSyncBflyI32Op::get_concrete_op_info(),
        args, destination, target, ...
    )?
)),
```

**File:** `crates/mir-importer/src/translator/terminator/intrinsics/warp.rs`

```rust
pub fn emit_warp_shuffle_i32(ctx, body, shuffle_opid, args, ...) {
    if args.len() != 2 { return error; }

    let (val, _) = translate_operand(ctx, body, &args[0], ...);
    let (lane_or_mask, _) = translate_operand(ctx, body, &args[1], ...);

    let shuffle_op = Operation::new(ctx, shuffle_opid,
        vec![u32_type.to_ptr()],          // results: [u32]
        vec![val, lane_or_mask],          // operands: [value, mask]
        vec![], 0,
    );

}
```

The emitter translates the user's MIR operands into pliron values --
typically the results of `mir.load`s from each operand's alloca slot, or
`mir.constant`s for literal arguments -- and wires them into the NVVM
operation. (These are not SSA values yet; `pliron::opts::mem2reg` will
collapse the load/store chains once translation is complete.) For intrinsics
with different argument patterns, you write a similar custom emitter.

### Stage 4 -- Lower to the LLVM dialect

**File:** `crates/mir-lower/src/convert/intrinsics/warp.rs`

Here is where it gets interesting. The LLVM intrinsic for butterfly shuffle
takes **four** arguments, not two:

```text
User's API:     shuffle_xor(value, lane_mask)            →  2 args
LLVM intrinsic: shfl.sync.bfly.i32(mask, value, lane_mask, clamp) →  4 args
                                    ^^^^                   ^^^^^
                                    always -1              always 31
```

The converter adds the compiler-supplied constants:

```rust
fn convert_shuffle_i32(op, ctx, intrinsic_name, clamp: i32) {
    let operands = get_operands(op, ctx)?;
    let (val, lane_or_delta) = (operands[0], operands[1]);

    let mask_val  = create_i32_const(ctx, -1);   // all 32 lanes participate
    let clamp_val = create_i32_const(ctx, clamp); // full warp width

    let func_ty = FuncType::get(ctx.ctx, i32_ty.into(),
        vec![i32_ty, i32_ty, i32_ty, i32_ty], false);

    let call_op = call_intrinsic(ctx, intrinsic_name, func_ty,
        vec![mask_val, val, lane_or_delta, clamp_val]);

    map_result(op, call_op, ctx);
}
```

The mask value of `-1` (all bits set) means all 32 lanes in the warp
participate. The clamp value of `31` means the shuffle wraps at the full warp
width. These are the right defaults for almost every use case, and exposing
them to the user would just be noise.

### Stage 5 -- Export (Add Convergent)

Warp shuffles are **convergent** -- LLVM must not reorder them relative to
control flow. The export step checks `is_convergent_intrinsic()`:

```rust
fn is_convergent_intrinsic(name: &str) -> bool {
    name == "llvm.nvvm.barrier0"
        || name.starts_with("llvm.nvvm.shfl")      // shuffles
        || name.starts_with("llvm.nvvm.vote")       // votes
        || name.starts_with("llvm.nvvm.mbarrier")   // async barriers
        || name.starts_with("llvm.nvvm.cp.async.bulk")
        // ...
}
```

If your new intrinsic is convergent, add it here. If you forget, LLVM might
hoist it out of an `if` block, and your warp-level code will produce wrong
results or deadlock. Not great.

**Final output:**

```llvm
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)

%v8 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(
    i32 -1, i32 %v3, i32 %v4, i32 31) #0

attributes #0 = { convergent }
```

After `llc`:

```text
shfl.sync.bfly.b32  %r3, %r1, %r2, 31;
```

---

## The Inline PTX Path

Some operations do not have LLVM intrinsics. For those, we emit inline PTX
assembly directly. The helper `inline_asm_convergent()` handles the boilerplate:

```rust
// wgmma fence: no inputs, no outputs, just a side effect
inline_asm_convergent(
    ctx, void_ty.into(), vec![],
    "wgmma.fence.sync.aligned;", ""
);

// mbarrier arrive: one input (pointer), one output (token)
inline_asm_convergent(
    ctx, i64_ty.into(), vec![ptr_val],
    "mbarrier.arrive.shared.b64 $0, [$1];", "=l,r"
);
```

The constraint string follows LLVM inline assembly syntax:

| Constraint | Meaning                                 |
| :--------- | :-------------------------------------- |
| `=l`       | Output: 64-bit register                 |
| `=r`       | Output: 32-bit register                 |
| `r`        | Input: 32-bit register                  |
| `l`        | Input: 64-bit register                  |
| (empty)    | No inputs or outputs (side-effect only) |

The `sideeffect convergent` markers on the inline assembly tell LLVM to leave
it alone -- do not move it, do not delete it, do not duplicate it.

---

## End-to-End: The Full Journey

Here is every representation `threadIdx_x` passes through, top to bottom:

```text
Rust:         thread::threadIdx_x()
MIR:          _3 = threadIdx_x() -> bb1
Pliron MIR:   %v = nvvm.read_ptx_sreg_tid_x : i32
Pliron LLVM:  %v = call i32 @llvm_nvvm_read_ptx_sreg_tid_x()
LLVM IR:      %v5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
PTX:          mov.u32 %r1, %tid.x;
```

And `shuffle_xor`:

```text
Rust:         warp::shuffle_xor(val, mask)
MIR:          _5 = shuffle_xor(_3, _4) -> bb2
Pliron MIR:   %v = nvvm.shfl_sync_bfly_i32 %val, %mask : i32
Pliron LLVM:  %v = call i32 @llvm_nvvm_shfl_sync_bfly_i32(i32 -1, %val, %mask, i32 31)
LLVM IR:      %v8 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, %v3, %v4, i32 31) #0
PTX:          shfl.sync.bfly.b32 %r3, %r1, %r2, 31;
```

Six representations. One Rust function call becomes one PTX instruction. The
intermediate steps exist so that each transformation is small, verifiable,
and independently testable.

---

## Quick-Reference Checklist

Every file you need to touch, in order:

1. **`cuda-device/src/<module>.rs`** -- `pub fn` with `#[inline(never)]` and
   `unreachable!()` body. This is the user-facing API.

2. **`dialect-nvvm/src/ops/<module>.rs`** -- `#[pliron_op(name = "nvvm.<name>", ...)]` struct,
   `Verify` impl (check operand/result counts), `register()` call.

3. **`mir-importer/src/translator/terminator/mod.rs`** -- `match` arm in
   `try_dispatch_intrinsic()`. Use `helpers::emit_nvvm_intrinsic()` for
   zero-arg intrinsics, or write a custom emitter for anything with arguments.

4. **`mir-lower/src/convert/interface_impls.rs`** -- `MirToLlvmConversion`
   impl for the new op, dispatching to the converter function.

5. **`mir-lower/src/convert/intrinsics/<module>.rs`** -- conversion logic.
   Use `call_intrinsic()` for LLVM intrinsics, or `inline_asm_convergent()`
   for inline PTX.

6. **`llvm-export/src/export/`** -- *only if convergent*: add to
   `is_convergent_intrinsic()`.

---

## Side-by-Side Comparison

| Stage        | `threadIdx_x` (simple) | `shuffle_xor` (complex)   |
| :----------- | :--------------------- | :------------------------ |
| cuda-device  | `fn() -> u32`          | `fn(u32, u32) -> u32`     |
| dialect-nvvm | 0 operands, 1 result   | 2 operands, 1 result      |
| mir-importer | Generic helper         | Custom emitter            |
| mir-lower    | `call @intrinsic()`    | `call @intrinsic(4 args)` |
| Convergent?  | No                     | Yes (`#0`)                |
| PTX          | `mov.u32 %r1, %tid.x`  | `shfl.sync.bfly.b32 ...`  |

---

That is the entire process. Five files, each with a clear and narrow
responsibility. The pattern is mechanical enough that adding a new intrinsic
should take about thirty minutes once you have done it once -- most of that
time spent reading the PTX ISA spec to figure out exactly what instruction
you want.
