{
  description = "Development shell for cuda-oxide";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config = {
            allowUnfree = true;
          };
        };

        # LLVM
        llvmPkgs = pkgs.llvmPackages_21;

        # Nightly Rust
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        # cuda
        cudaSymlinked = pkgs.symlinkJoin {
          name = "cuda-symlinked";
          paths = with pkgs.cudaPackages_13; [
            cuda_nvcc
            cuda_gdb.bin
            cuda_cudart
            libnvvm
            libnvjitlink.lib
          ];
        };
      in
      {
        formatter = pkgs.nixfmt;

        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain

            llvmPkgs.clang
            llvmPkgs.libclang
            llvmPkgs.llvm

            cudaSymlinked
          ];

          CUDA_HOME = cudaSymlinked;
          LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";
          CUDA_OXIDE_LLC = "${llvmPkgs.llvm}/bin/llc";

          shellHook = ''
            # GPU driver setup borrowed from https://github.com/NVlabs/cutile-rs
            # NixOS provides /run/opengl-driver/lib
            if [ -d /run/opengl-driver/lib ]; then
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            else
              # Non-NixOS Linux (Ubuntu, Arch, etc. running Nix)
              # Symlink only the NVIDIA driver to avoid host glibc pollution
              _nv_drv_dir=$(mktemp -d /tmp/nix-nvidia-driver.XXXXXX)

              for d in /usr/lib/x86_64-linux-gnu \
                       /lib/x86_64-linux-gnu \
                       /usr/lib/aarch64-linux-gnu \
                       /lib/aarch64-linux-gnu \
                       /usr/lib \
                       /usr/lib64; do
                if [ -e "$d/libcuda.so.1" ]; then
                  for lib in "$d"/libcuda.so* "$d"/libnvidia-ptxjitcompiler.so* "$d"/libnvidia-gpucomp.so*; do
                    [ -e "$lib" ] && ln -sf "$lib" "$_nv_drv_dir/"
                  done
                  break
              fi
              done

              if [ -n "$(ls -A "$_nv_drv_dir" 2>/dev/null)" ]; then
                export LD_LIBRARY_PATH="$_nv_drv_dir:$LD_LIBRARY_PATH"
                # cleanup when the user exits the nix shell
                trap 'rm -rf "$_nv_drv_dir"' EXIT
              else
                # Clean up immediately if no NVIDIA drivers were found
                rm -rf "$_nv_drv_dir"
              fi
            fi

            echo "🦀 cuda-oxide dev environment loaded"
            echo " ✓ LLVM $(llc   --version | grep 'LLVM version' | awk '{print $3}')"
            echo " ✓ CUDA $(nvcc  --version | grep 'release'      | awk '{print $6}' | cut -c 2-)"
            echo " ✓ Rust $(rustc --version |                       awk '{print $2}')"
          '';
        };
      }
    );
}
