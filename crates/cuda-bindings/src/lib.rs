/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Licensed under the NVIDIA Software License (see LICENSE-NVIDIA at the
 * repository root). This crate, unlike the rest of the workspace, is not
 * Apache-2.0.
 */

//! Low-level FFI to the CUDA Driver API (`cuda.h`).
//!
//! Bindings are generated at build time by [`bindgen`](https://docs.rs/bindgen) from `wrapper.h`,
//! which includes the toolkit `cuda.h`. The build script passes `-I$CUDA_TOOLKIT_PATH/include` to
//! Clang, emits `cargo:rustc-link-search` for discovered library directories, and links
//! `libcuda` (`dylib=cuda`). Generated Rust lives under `OUT_DIR` as `bindings.rs` and is pulled in
//! via [`include!`].
//!
//! **Toolkit path:** set `CUDA_TOOLKIT_PATH` (or, failing that, `CUDA_HOME`) to the root of your
//! CUDA installation (the directory that contains `include/cuda.h`). If neither is set, the build
//! script and [`cuda_toolkit_dir`] both use `/usr/local/cuda`. Changing either variable or
//! `wrapper.h` triggers a rebuild.
//!
//! Types and functions in the generated module are `unsafe` where required by Rust; each carries
//! the usual CUDA API preconditions (valid handles, device state, stream ordering, etc.).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// The generated bindings carry CUDA's C doxygen comments verbatim. Those contain
// `[...]` spans, bare URLs, HTML-ish tables, and `\brief`-style code that rustdoc
// flags as broken intra-doc links, bare URLs, unclosed HTML tags, and unparseable
// Rust code blocks. We keep the comments (they are useful API docs) but silence
// these lints for this generated FFI crate; its doctests are excluded from the
// `--doc` gate in CI.
#![allow(rustdoc::broken_intra_doc_links)]
#![allow(rustdoc::bare_urls)]
#![allow(rustdoc::invalid_html_tags)]
#![allow(rustdoc::invalid_rust_codeblocks)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::env;

/// Reports the elapsed time between two recorded events, dispatching to the
/// event elapsed-time driver entry point declared by this build's toolkit
/// headers.
///
/// CUDA 12.8 renamed the entry point to `cuEventElapsedTime_v2`; earlier
/// toolkits only declare `cuEventElapsedTime`. The build script probes
/// `cuda.h` and sets the `cuda_has_cuEventElapsedTime_v2` cfg accordingly, so
/// callers stay source-compatible across toolkit versions.
///
/// # Safety
///
/// Same contract as the underlying driver call: `elapsed_ms` must be valid
/// for a `f32` write, and `start`/`end` must be valid event handles recorded
/// in the current context.
pub unsafe fn cu_event_elapsed_time(
    elapsed_ms: *mut f32,
    start: CUevent,
    end: CUevent,
) -> CUresult {
    #[cfg(cuda_has_cuEventElapsedTime_v2)]
    {
        unsafe { cuEventElapsedTime_v2(elapsed_ms, start, end) }
    }
    #[cfg(not(cuda_has_cuEventElapsedTime_v2))]
    {
        unsafe { cuEventElapsedTime(elapsed_ms, start, end) }
    }
}

/// Root directory of the CUDA toolkit used for this build, for host code that must agree with
/// compile-time include and link paths (e.g. loading companion libraries or probing layout).
///
/// Resolution matches `build.rs`: the first set variable among `CUDA_TOOLKIT_PATH` and
/// `CUDA_HOME` (taken verbatim); when neither is present (or the value is not Unicode),
/// returns `/usr/local/cuda`.
pub fn cuda_toolkit_dir() -> String {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .find_map(|var| env::var(var).ok())
        .unwrap_or_else(|| "/usr/local/cuda".to_string())
}
