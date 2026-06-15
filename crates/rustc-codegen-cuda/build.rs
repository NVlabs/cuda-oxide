// build.rs for rustc_codegen_cuda (Windows fix)
//
// The codegen backend is built as a `dylib` which on Windows means Rust
// auto-generates a .def file exporting every public symbol (~66953).
// This exceeds the PE/COFF limit of 65535 exports.
//
// This build script tells the linker to use our minimal .def file instead,
// which exports only `__rustc_codegen_backend` — the single entry point
// rustc calls to load the backend.

fn main() {
    #[cfg(target_os = "windows")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let def_path = std::path::Path::new(&manifest_dir).join("codegen_backend.def");
        
        if def_path.exists() {
            // Tell rustc to pass our .def file to the linker
            println!("cargo:rustc-link-arg=/DEF:{}", def_path.display());
            // Tell rustc NOT to generate its own .def file for this DLL
            println!("cargo:rustc-link-arg=/NODEFAULTLIB:__rust_no_alloc_shim_is_unstable");
        }
        
        // Add our stub ffi.lib to the search path
        println!("cargo:rustc-link-search=native={}", manifest_dir);
    }
}
