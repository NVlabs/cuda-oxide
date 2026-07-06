# Enzyme Autodiff via `CUDA_OXIDE_POST_IR`

This example differentiates a Rust `#[device]` function with
[Enzyme](https://enzyme.mit.edu/) — automatic differentiation performed on the
compiler's LLVM IR — using the `CUDA_OXIDE_POST_IR` hook. No cuda-oxide fork,
no manual IR authoring: an external script edits the exported `.ll` between IR
export and PTX generation, and the pipeline finishes code generation as usual.

**Status: ✅ verified on TITAN V (sm_70); derivative matches the analytic one
at 1024 points**

---

## Quick start

```bash
cargo oxide run enzyme_autodiff
```

Without an Enzyme plugin installed the hook leaves the IR untouched and the
example opts out gracefully:

```text
skipping: LLVMEnzyme plugin not available (enzyme.sh left the IR untouched)
  build Enzyme against LLVM 21 and set LLVMENZYME=/path/to/LLVMEnzyme-21.so
```

With one (see [Building Enzyme](#building-enzyme) below):

```bash
LLVMENZYME=/path/to/LLVMEnzyme-21.so cargo oxide run enzyme_autodiff
```

```text
=== Enzyme autodiff via CUDA_OXIDE_POST_IR ===

f(x)  = x^3 + 2x^2 - 5x + 1   checked at 1024 points: ok
f'(x) = 3x^2 + 4x - 5 (Enzyme) checked at 1024 points: ok
  e.g. f'(-1.336) = -4.989563 (analytic -4.989563)

✓ SUCCESS: Enzyme's device derivative matches the analytic one
```

## How it works

The Rust source defines three things:

- `poly` — an ordinary `#[device]` function, the primal `f(x) = x³ + 2x² − 5x + 1`;
- `poly_dx` — a stub with the derivative's signature whose body just returns a
  sentinel value (`#[inline(never)]` keeps its call sites intact);
- `poly_grad` — a normal `#[kernel]` that maps `poly_dx` over an array.

`.cargo/config.toml` sets `CUDA_OXIDE_POST_IR=enzyme.sh` for the device build
(the pipeline runs inside rustc, which cargo spawns from this directory, so the
`[env]` entry reaches it). The pipeline invokes the hook between IR export and
PTX generation as

```text
enzyme.sh <ll_path> <output_dir> <output_name> <target>
```

and `enzyme.sh` performs the whole transform on `<ll_path>` in place:

1. **Find** the `@poly` and `@poly_dx` defines in the exported IR (device
   functions keep their Rust path, with `::` lowered to `__`).
2. **Retarget the stub**: rename `@poly_dx`'s `define` line out of the way —
   call sites still refer to the original name — and append a replacement
   definition whose body is Enzyme's forward-mode request:

   ```llvm
   declare double @__enzyme_fwddiff(...)

   define double @poly_dx(double %x) {
     %r = call double (...) @__enzyme_fwddiff(ptr @poly, double %x, double 1.0)
     ret double %r
   }
   ```

3. **Run Enzyme**: `opt -load-pass-plugin=LLVMEnzyme-21.so
   -passes='enzyme,default<O2>'` replaces the `__enzyme_fwddiff` call with
   generated derivative code.

The pipeline then finishes its normal `opt`/`llc` PTX generation on the edited
IR, and the host launches `poly_grad` through the typed `#[cuda_module]` API —
the host-side code never knows the derivative was machine-generated. If the
stub still returns its sentinel at runtime, the host knows the hook skipped
(no plugin) and prints `skipping:`; if `opt` fails, the hook's non-zero exit
aborts the build with its stderr.

The primal is a polynomial on purpose: no libdevice (`__nv_*`) math, so the
edited IR stays self-contained for `llc`. Device functions that call libdevice
math need the hook to also `llvm-link` `libdevice.10.bc` (and internalize +
DCE the leftovers) — see the pipeline's libdevice handling for the pattern.

## Configuration

| Variable     | Meaning                                                                                     |
|--------------|---------------------------------------------------------------------------------------------|
| `LLVMENZYME` | Path to the Enzyme plugin. Unset/missing: probe common install paths, else skip gracefully. |
| `ENZYME_OPT` | The `opt` matching the plugin's LLVM major. Default: `opt-21`, `/usr/lib/llvm-21/bin/opt`, `opt`. |

## Building Enzyme

Enzyme is built out-of-tree against an LLVM installation; the plugin's LLVM
major must match the `opt` that loads it (LLVM 21 below):

```bash
git clone https://github.com/EnzymeAD/Enzyme
cmake -S Enzyme/enzyme -B Enzyme/enzyme/build -G Ninja \
      -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm
cmake --build Enzyme/enzyme/build
# → Enzyme/enzyme/build/Enzyme/LLVMEnzyme-21.so
```

## Notes

- **Forward mode** is used here (`__enzyme_fwddiff`, input tangent seeded with
  `1.0`): one active scalar in, its tangent out — the right shape for
  parameter sensitivities. Reverse mode (`__enzyme_autodiff`) works through
  the same hook but needs `Duplicated` shadow buffers for accumulation; see
  the [Enzyme docs](https://enzyme.mit.edu/getting_started/CallingConvention/).
- Enzyme differentiates plain device *functions*, not `ptx_kernel` entry
  points — which is exactly what the `#[device]`-function-plus-stub layout
  provides. Keeping the per-thread indexing in the Rust kernel means the hook
  only ever touches scalar code.
- The same hook mechanism fits other IR-stage tools: custom LLVM
  passes/plugins, instrumentation, or linking external bitcode. See the
  `post_ir_hook` example for the minimal version of the wiring.
