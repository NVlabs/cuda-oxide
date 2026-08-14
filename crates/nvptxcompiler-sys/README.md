# nvptxcompiler-sys

Runtime-loaded, driver-independent bindings to NVIDIA's PTX Compiler API.

The crate owns dynamic library discovery, symbol resolution, compiler-handle
lifetime, option marshaling, and raw log/program retrieval. It does not link
the CUDA Driver and requires the CUDA Toolkit only at run time.
