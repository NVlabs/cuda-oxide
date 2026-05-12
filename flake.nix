{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      systems = [ "aarch64-linux" "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      mkDevShell = system:
        let
          # allowUnfree is required because CUDA is unfree.
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [ rust-overlay.overlays.default ];
          };
          lib = pkgs.lib;

          # ---- CUDA toolkit ----
          # cuda-oxide does NOT link libNVVM. We only need cuda.h + libcuda
          # stubs (for cuda-bindings bindgen) and ptxas / cuobjdump (for runtime
          # PTX → SASS via the driver). CUDA 12.x+ is fine; 13.2 is what was
          # already pinned.
          cudaRoot = pkgs.cudaPackages_13_2.cudatoolkit;

          # Single source of truth for channel + components lives in
          # rust-toolchain.toml. Update there, not here.
          toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

          # ---- LLVM 21 ----
          # cuda-oxide emits LLVM IR with NVPTX intrinsics (TMA / tcgen05 /
          # WGMMA) that require llc 21+. cuda-oxide auto-discovers `llc-22`
          # then `llc-21` on PATH; we pin to 21 via CUDA_OXIDE_LLC for
          # determinism. `bindgen` (in cuda-bindings) needs libclang from the
          # same family.
          llvm21 = pkgs.llvmPackages_21;
          llvm21Bin = lib.getBin llvm21.llvm;
          llvm21Dev = lib.getDev llvm21.llvm;
          # cuda-oxide auto-discovers `llc-21` / `llc-22` but upstream LLVM
          # builds only provide unversioned `llc`. Shim a `llc-21` so PATH
          # discovery works regardless of the CUDA_OXIDE_LLC override.
          llvm21CompatTools = pkgs.symlinkJoin {
            name = "llvm21-compat-tools";
            paths = [
              (pkgs.writeShellScriptBin "llc-21" ''exec ${llvm21Bin}/bin/llc "$@"'')
              (pkgs.writeShellScriptBin "llvm-as-21" ''exec ${llvm21Bin}/bin/llvm-as "$@"'')
              (pkgs.writeShellScriptBin "llvm-dis-21" ''exec ${llvm21Bin}/bin/llvm-dis "$@"'')
              (pkgs.writeShellScriptBin "opt-21" ''exec ${llvm21Bin}/bin/opt "$@"'')
            ];
          };
        in
        pkgs.mkShell {
          CUDA_HOME = "${cudaRoot}";
          CUDA_ROOT = "${cudaRoot}";
          CUDA_PATH = "${cudaRoot}";
          CUDA_TOOLKIT_ROOT_DIR = "${cudaRoot}";
          # cuda-bindings (cuda-oxide) reads CUDA_TOOLKIT_PATH specifically;
          # default fallback would be /usr/local/cuda which doesn't exist on nix.
          CUDA_TOOLKIT_PATH = "${cudaRoot}";
          # Cover both lib/ (nix-style) and lib64/ (FHS-style) so downstream
          # bindgen / build.rs scripts that probe either layout resolve
          # libcudart + stubs.
          CUDA_LIBRARY_PATH =
            "${cudaRoot}/lib:${cudaRoot}/lib64:${cudaRoot}/lib/stubs:${cudaRoot}/lib64/stubs";

          # cuda-bindings runs bindgen, which loads libclang at *runtime* of
          # the build script. Point at LLVM 21's libclang.so.
          LIBCLANG_PATH = "${lib.getLib llvm21.libclang}/lib";
          LLVM_CONFIG = "${llvm21Dev}/bin/llvm-config";
          # Pin the llc invoked by cuda-oxide's pipeline. cuda-oxide's auto-
          # discovery looks for llc-22, then llc-21; honoring this env var
          # short-circuits that search.
          CUDA_OXIDE_LLC = "${llvm21Bin}/bin/llc";

          nativeBuildInputs = [
            toolchain
            pkgs.pkg-config
            pkgs.cmake
            pkgs.ninja
            pkgs.patchelf
            cudaRoot
            llvm21.clang
            llvm21.libclang
            llvm21Bin
            llvm21Dev
            llvm21CompatTools
          ];
          buildInputs = [
            pkgs.openssl
            pkgs.libxml2
            pkgs.zlib
            pkgs.ncurses
            pkgs.libffi
            pkgs.stdenv.cc.cc.lib
          ];

          shellHook = ''
            export CARGO_TARGET_DIR="$PWD/target"
            export PATH="${llvm21CompatTools}/bin:${llvm21Bin}/bin:${llvm21Dev}/bin:${cudaRoot}/bin:$PATH"
            # libcuda.so.1 is provided by the host NVIDIA driver, not nix.
            # LD_LIBRARY_PATH covers libcudart + libcublas etc. for runtime.
            export LD_LIBRARY_PATH="${cudaRoot}/lib64:${cudaRoot}/lib:${lib.getLib llvm21.libclang}/lib:${pkgs.ncurses.out}/lib:${pkgs.libxml2.out}/lib:${pkgs.zlib.out}/lib:${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            # cuda-bindings' build.rs only adds lib64/{,stubs} to the rustc link
            # search path; on aarch64 CUDA toolkits the stubs live under
            # lib/stubs, so add both layouts to LIBRARY_PATH so -lcuda resolves.
            export LIBRARY_PATH="${cudaRoot}/lib64:${cudaRoot}/lib64/stubs:${cudaRoot}/lib:${cudaRoot}/lib/stubs''${LIBRARY_PATH:+:$LIBRARY_PATH}"

            # Point cargo-oxide at the codegen backend .so that its `setup`
            # command produces. Our shellHook sets CARGO_TARGET_DIR, so the
            # build artifacts land here regardless of where setup ran. Without
            # this env var, cargo-oxide looks for the .so under the codegen
            # crate's nested target dir and won't find it.
            export CUDA_OXIDE_BACKEND="$PWD/target/debug/librustc_codegen_cuda.so"

            echo "cuda-oxide llvm21 shell"
            echo "  CUDA_HOME=$CUDA_HOME"
            echo "  LIBCLANG_PATH=$LIBCLANG_PATH"
            echo "  CUDA_OXIDE_LLC=$CUDA_OXIDE_LLC"
          '';
        };
    in
    {
      devShells = forAllSystems (system: {
        default = mkDevShell system;
      });
    };
}
