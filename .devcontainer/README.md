# cuda-oxide Dev Container

This dev container provides the toolchain expected by cuda-oxide:

- Ubuntu 24.04
- CUDA Toolkit 13.0
- LLVM 21 with NVPTX support
- Clang 21 resource headers for `bindgen`
- Rust `nightly-2026-04-03` with `rust-src` and `rustc-dev`

Open the repository in a devcontainer-aware editor and choose "Reopen in
Container". The container requests GPU access with `--gpus=all` and uses
`updateRemoteUserUID` so generated Cargo artifacts and exported PTX files stay
writable from the host checkout.

Inside the container:

```bash
cargo oxide doctor
cargo oxide run vecadd
```

The CUDA Toolkit is provided by the container image; it does not need to be
installed on the host. The host must provide an NVIDIA GPU, a driver compatible
with CUDA 13.0, and the NVIDIA Container Toolkit so Docker can expose the GPU
to the container.

If the host driver is too old, GPU commands such as `nvidia-smi`,
`cargo oxide doctor`, or `cargo oxide run vecadd` will fail inside the
container. Update the host NVIDIA driver rather than installing a different
CUDA Toolkit in the container.

The image does not set `LD_LIBRARY_PATH`; the NVIDIA Container Toolkit should
provide the host driver libraries at runtime.
