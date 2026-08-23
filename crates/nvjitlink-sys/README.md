# nvjitlink-sys

Runtime (`dlopen`) bindings to NVIDIA's nvJitLink. nvJitLink links one or more LTOIR modules (and other input forms — PTX, cubin, fatbin) into a final cubin or PTX.

## What this crate provides

- `LibNvJitLink` — RAII wrapper around the loaded library + resolved function pointers.
- `Linker` — RAII wrapper around an `nvJitLinkHandle`, with `add` / `finish` methods.
- `InputType` — supported input formats (`Ltoir`, `Ptx`, `Cubin`, `Fatbin`, ...).
- `NvJitLinkError` — typed errors with the nvJitLink error log captured.

## Build requirements

None. The library is loaded at runtime, so the CUDA Toolkit only needs to be present when the program runs (not when it compiles).

## Library discovery

`LibNvJitLink::load()` tries (in order):

1. `LIBNVJITLINK_PATH` env var, if set.
2. CUDA Toolkit roots from `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, and `CUDA_PATH`.
   - Linux: `<root>/lib64/libnvJitLink.so`.
   - Windows: versioned `nvJitLink_*.dll` files under `<root>/bin/x64`
     and `<root>/bin`, plus `<root>/lib/x64/nvJitLink.dll`.
3. Platform loader locations.
   - Linux: `libnvJitLink.so.13`, `libnvJitLink.so.12`,
     `libnvJitLink.so`.
   - Windows: versioned `nvJitLink_*.dll` files found on `PATH`, then the
     known CUDA 13/12 loader names as fallbacks.

Linux additionally probes `/usr/local/cuda` and `/opt/cuda` as Toolkit roots.
Windows does not inherit those Unix defaults.

nvJitLink ships with the standard CUDA Toolkit. No separate download.

## Symbol naming

`nvJitLink.h` maps every public function to a versioned mangled name (e.g. `nvJitLinkCreate -> __nvJitLinkCreate_13_0`). Toolkit installations may export either the public unsuffixed name or only the mangled name, so the binding probes both forms.

## Usage

This crate is low-level. Most users want the higher-level `cuda_host::ltoir::load_kernel_module` helper, which combines libNVVM + libdevice + nvJitLink behind one call. Use this crate directly only if you need explicit control over the link.

```rust
use nvjitlink_sys::{LibNvJitLink, Linker, InputType};

let nvj = LibNvJitLink::load()?;
let mut linker = Linker::new(&nvj, &["-arch=sm_120", "-lto"])?;
linker.add(InputType::Ltoir, &ltoir_bytes, "kernel.ltoir")?;
let cubin = linker.finish()?;
```

## Companion crate

[`libnvvm-sys`](../libnvvm-sys/) — same pattern, for libNVVM. Together they cover the NVVM IR → LTOIR → cubin pipeline.
