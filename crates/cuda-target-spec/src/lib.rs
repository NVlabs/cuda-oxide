/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared CUDA target parsing and recorded target-to-PTX policy.
//!
//! The floors in [`RECORDED_PTX_FLOORS`] describe the defaults emitted by the
//! pinned LLVM 22 NVPTX backend. They are not backend-independent CUDA facts;
//! in particular, LLVM 21 does not accept every target recorded here.

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaArch {
    capability: u32,
    suffix: Option<char>,
}

impl CudaArch {
    pub fn capability(&self) -> u32 {
        self.capability
    }
    pub fn suffix(&self) -> Option<char> {
        self.suffix
    }
    pub fn uses_legacy_llvm(&self) -> bool {
        self.capability < 100
    }
    pub fn sm(&self) -> String {
        self.render("sm_")
    }
    pub fn compute(&self) -> String {
        self.render("compute_")
    }
    fn render(&self, prefix: &str) -> String {
        match self.suffix {
            Some(suffix) => format!("{prefix}{}{suffix}", self.capability),
            None => format!("{prefix}{}", self.capability),
        }
    }
}

impl FromStr for CudaArch {
    type Err = CudaArchParseError;
    fn from_str(target: &str) -> Result<Self, Self::Err> {
        let rest = target
            .strip_prefix("sm_")
            .or_else(|| target.strip_prefix("compute_"))
            .ok_or_else(|| CudaArchParseError::new(target, "expected `sm_XX` or `compute_XX`"))?;
        let digit_count = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        if digit_count < 2 {
            return Err(CudaArchParseError::new(
                target,
                "compute capability must contain at least two digits",
            ));
        }
        let (digits, suffix_text) = rest.split_at(digit_count);
        let suffix = match suffix_text {
            "" => None,
            "a" => Some('a'),
            "f" => Some('f'),
            _ => {
                return Err(CudaArchParseError::new(
                    target,
                    "the only supported architecture suffixes are `a` and `f`",
                ));
            }
        };
        let capability = digits.parse::<u32>().map_err(|_| {
            CudaArchParseError::new(target, "compute capability is not a valid integer")
        })?;
        Ok(Self { capability, suffix })
    }
}

impl fmt::Display for CudaArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.sm())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaArchParseError {
    target: String,
    reason: &'static str,
}
impl CudaArchParseError {
    fn new(target: &str, reason: &'static str) -> Self {
        Self {
            target: target.to_string(),
            reason,
        }
    }
}
impl fmt::Display for CudaArchParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid CUDA target `{}`: {}", self.target, self.reason)
    }
}
impl std::error::Error for CudaArchParseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetPtxFloor {
    pub capability: u32,
    pub suffix: Option<char>,
    pub floor: u16,
}

pub const RECORDED_PTX_FLOORS: &[TargetPtxFloor] = &[
    TargetPtxFloor {
        capability: 70,
        suffix: None,
        floor: 60,
    },
    TargetPtxFloor {
        capability: 72,
        suffix: None,
        floor: 61,
    },
    TargetPtxFloor {
        capability: 75,
        suffix: None,
        floor: 63,
    },
    TargetPtxFloor {
        capability: 80,
        suffix: None,
        floor: 70,
    },
    TargetPtxFloor {
        capability: 86,
        suffix: None,
        floor: 71,
    },
    TargetPtxFloor {
        capability: 87,
        suffix: None,
        floor: 74,
    },
    TargetPtxFloor {
        capability: 88,
        suffix: None,
        floor: 90,
    },
    TargetPtxFloor {
        capability: 89,
        suffix: None,
        floor: 78,
    },
    TargetPtxFloor {
        capability: 90,
        suffix: None,
        floor: 78,
    },
    TargetPtxFloor {
        capability: 90,
        suffix: Some('a'),
        floor: 80,
    },
    TargetPtxFloor {
        capability: 100,
        suffix: None,
        floor: 86,
    },
    TargetPtxFloor {
        capability: 100,
        suffix: Some('a'),
        floor: 86,
    },
    TargetPtxFloor {
        capability: 100,
        suffix: Some('f'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 101,
        suffix: None,
        floor: 86,
    },
    TargetPtxFloor {
        capability: 101,
        suffix: Some('a'),
        floor: 86,
    },
    TargetPtxFloor {
        capability: 101,
        suffix: Some('f'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 103,
        suffix: None,
        floor: 88,
    },
    TargetPtxFloor {
        capability: 103,
        suffix: Some('a'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 103,
        suffix: Some('f'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 110,
        suffix: None,
        floor: 90,
    },
    TargetPtxFloor {
        capability: 110,
        suffix: Some('a'),
        floor: 90,
    },
    TargetPtxFloor {
        capability: 110,
        suffix: Some('f'),
        floor: 90,
    },
    TargetPtxFloor {
        capability: 120,
        suffix: None,
        floor: 87,
    },
    TargetPtxFloor {
        capability: 120,
        suffix: Some('a'),
        floor: 87,
    },
    TargetPtxFloor {
        capability: 120,
        suffix: Some('f'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 121,
        suffix: None,
        floor: 88,
    },
    TargetPtxFloor {
        capability: 121,
        suffix: Some('a'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 121,
        suffix: Some('f'),
        floor: 88,
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsupportedTargetError {
    target: String,
}
impl fmt::Display for UnsupportedTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA target `{}` has no recorded PTX ISA floor",
            self.target
        )
    }
}
impl std::error::Error for UnsupportedTargetError {}

pub fn recorded_ptx_floor(arch: &CudaArch) -> Result<u16, UnsupportedTargetError> {
    RECORDED_PTX_FLOORS
        .iter()
        .find(|entry| entry.capability == arch.capability && entry.suffix == arch.suffix)
        .map(|entry| entry.floor)
        .ok_or_else(|| UnsupportedTargetError {
            target: arch.to_string(),
        })
}

pub const PTX_ISA_SPELLINGS: &[u16] = &[62, 65, 70, 71, 73, 78, 80, 86, 87, 88, 90];
pub fn spelling_at_least(floor: u16) -> Option<u16> {
    PTX_ISA_SPELLINGS
        .iter()
        .copied()
        .find(|spelling| *spelling >= floor)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_arch_parses_and_renders_api_specific_spellings() {
        for (input, capability, suffix, sm, compute, legacy) in [
            ("sm_75", 75, None, "sm_75", "compute_75", true),
            ("compute_90a", 90, Some('a'), "sm_90a", "compute_90a", true),
            ("sm_100f", 100, Some('f'), "sm_100f", "compute_100f", false),
            ("compute_120", 120, None, "sm_120", "compute_120", false),
        ] {
            let arch: CudaArch = input.parse().unwrap();
            assert_eq!((arch.capability(), arch.suffix()), (capability, suffix));
            assert_eq!(
                (arch.sm(), arch.compute()),
                (sm.to_string(), compute.to_string())
            );
            assert_eq!(arch.uses_legacy_llvm(), legacy);
        }
    }
    #[test]
    fn cuda_arch_rejects_ambiguous_or_malformed_targets() {
        for input in [
            "", "86", "sm_", "sm_9", "sm_90x", "sm_90aa", "SM_90", "gfx90a",
        ] {
            assert!(input.parse::<CudaArch>().is_err(), "{input}");
        }
    }

    #[cfg(unix)]
    mod backend {
        use super::*;
        use std::fs;
        use std::path::{Path, PathBuf};
        use std::process::{Command, Output};

        struct TestDir(PathBuf);
        impl TestDir {
            fn new() -> Self {
                let path = std::env::temp_dir().join(format!(
                    "cuda-target-spec-{}-{:?}",
                    std::process::id(),
                    std::thread::current().id()
                ));
                fs::create_dir_all(&path).unwrap();
                Self(path)
            }
        }
        impl Drop for TestDir {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }

        fn rust_toolchain_llc() -> PathBuf {
            let sysroot = Command::new("rustc")
                .args(["--print", "sysroot"])
                .output()
                .unwrap();
            assert!(sysroot.status.success(), "rustc --print sysroot failed");
            let verbose = Command::new("rustc").arg("-vV").output().unwrap();
            assert!(verbose.status.success(), "rustc -vV failed");
            let host = String::from_utf8_lossy(&verbose.stdout)
                .lines()
                .find_map(|line| line.strip_prefix("host: "))
                .expect("rustc -vV did not report a host")
                .to_owned();
            let path = PathBuf::from(String::from_utf8_lossy(&sysroot.stdout).trim())
                .join("lib/rustlib")
                .join(host)
                .join("bin/llc");
            assert!(
                path.is_file(),
                "rust toolchain has no llc at {}",
                path.display()
            );
            path
        }

        fn llvm_22() -> Option<PathBuf> {
            let llc = rust_toolchain_llc();
            let output = Command::new(&llc).arg("--version").output().unwrap();
            assert!(output.status.success(), "llc --version failed");
            let version = String::from_utf8_lossy(&output.stdout);
            let major = version
                .lines()
                .find_map(|line| {
                    line.trim()
                        .strip_prefix("LLVM version ")?
                        .split('.')
                        .next()?
                        .parse::<u32>()
                        .ok()
                })
                .expect("llc --version did not report an LLVM version");
            if major != 22 {
                eprintln!("skipping LLVM-derived PTX-floor test: expected LLVM 22, found {major}");
                return None;
            }
            Some(llc)
        }

        fn module(directory: &Path) -> PathBuf {
            let module = directory.join("probe.ll");
            fs::write(&module, "target triple = \"nvptx64-nvidia-cuda\"\n\ndefine void @probe() {\nentry:\n  ret void\n}\n").unwrap();
            module
        }

        fn lower(
            llc: &Path,
            module: &Path,
            target: &str,
            feature: Option<&str>,
            output: &Path,
        ) -> Output {
            let mut command = Command::new(llc);
            command
                .arg("-mtriple=nvptx64-nvidia-cuda")
                .arg(format!("-mcpu={target}"));
            if let Some(feature) = feature {
                command.arg(format!("-mattr={feature}"));
            }
            command
                .arg("-filetype=asm")
                .arg(module)
                .arg("-o")
                .arg(output)
                .output()
                .unwrap()
        }

        fn emitted_ptx_isa(path: &Path) -> u16 {
            let ptx = fs::read_to_string(path).unwrap();
            let version = ptx
                .lines()
                .find_map(|line| line.trim().strip_prefix(".version "))
                .expect("emitted PTX carries no .version");
            let (major, minor) = version.split_once('.').unwrap();
            major.parse::<u16>().unwrap() * 10 + minor.parse::<u16>().unwrap()
        }

        #[test]
        fn recorded_floors_match_llvm_22_defaults() {
            let Some(llc) = llvm_22() else { return };
            let directory = TestDir::new();
            let module = module(&directory.0);
            for entry in RECORDED_PTX_FLOORS {
                let target = match entry.suffix {
                    Some(s) => format!("sm_{}{s}", entry.capability),
                    None => format!("sm_{}", entry.capability),
                };
                let output = directory.0.join(format!("{target}.ptx"));
                let result = lower(&llc, &module, &target, None, &output);
                assert!(
                    result.status.success(),
                    "{target}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert_eq!(emitted_ptx_isa(&output), entry.floor, "{target}");
            }
        }

        #[test]
        fn sm_90a_requires_ptx_80() {
            let Some(llc) = llvm_22() else { return };
            let directory = TestDir::new();
            let module = module(&directory.0);
            let output = directory.0.join("sm_90a.ptx");
            let pass = lower(&llc, &module, "sm_90a", Some("+ptx80"), &output);
            assert!(
                pass.status.success(),
                "{}",
                String::from_utf8_lossy(&pass.stderr)
            );
            assert_eq!(emitted_ptx_isa(&output), 80);
            let reject = lower(&llc, &module, "sm_90a", Some("+ptx78"), &output);
            assert!(!reject.status.success());
            assert!(
                String::from_utf8_lossy(&reject.stderr).contains("Minimum required PTX version")
            );
        }

        #[test]
        fn sm_103a_rejects_ptx_86_near_miss() {
            let Some(llc) = llvm_22() else { return };
            let directory = TestDir::new();
            let module = module(&directory.0);
            let output = directory.0.join("sm_103a.ptx");
            let reject = lower(&llc, &module, "sm_103a", Some("+ptx86"), &output);
            assert!(!reject.status.success());
            let pass = lower(&llc, &module, "sm_103a", Some("+ptx88"), &output);
            assert!(
                pass.status.success(),
                "{}",
                String::from_utf8_lossy(&pass.stderr)
            );
            assert_eq!(emitted_ptx_isa(&output), 88);
        }
    }
}
