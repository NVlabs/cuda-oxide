/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use libnvvm_sys::CudaArch;

/// Amount of device debug information preserved during finalization.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum DebugPolicy {
    /// Do not request debug information from the CUDA compiler tools.
    #[default]
    None,
    /// Preserve source line mappings without disabling optimization.
    LineTables,
    /// Emit full debug information and disable libNVVM optimization.
    Full,
}

/// Typed options shared by the libNVVM and nvJitLink stages.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FinalizationOptions {
    target: CudaArch,
    allow_fma_contraction: bool,
    debug: DebugPolicy,
}

impl FinalizationOptions {
    /// Start with cuda-oxide's ordinary optimized compilation policy.
    pub fn new(target: CudaArch) -> Self {
        Self {
            target,
            allow_fma_contraction: true,
            debug: DebugPolicy::None,
        }
    }

    /// Select whether multiply-add contraction is permitted.
    #[must_use]
    pub fn with_fma_contraction(mut self, allow: bool) -> Self {
        self.allow_fma_contraction = allow;
        self
    }

    /// Select the device debug-information policy.
    #[must_use]
    pub fn with_debug_policy(mut self, debug: DebugPolicy) -> Self {
        self.debug = debug;
        self
    }

    /// Concrete CUDA architecture used by both compiler stages.
    pub fn target(&self) -> &CudaArch {
        &self.target
    }

    /// Whether multiply-add contraction is permitted.
    pub fn allow_fma_contraction(&self) -> bool {
        self.allow_fma_contraction
    }

    /// Device debug-information policy.
    pub fn debug_policy(&self) -> DebugPolicy {
        self.debug
    }

    pub(crate) fn nvvm_verify_options(&self) -> Vec<String> {
        vec![format!("-arch={}", self.target.compute())]
    }

    pub(crate) fn nvvm_compile_options(&self) -> Vec<String> {
        let mut options = vec![
            format!("-arch={}", self.target.compute()),
            "-gen-lto".to_string(),
            self.fma_option().to_string(),
        ];
        if self.debug == DebugPolicy::Full {
            options.push("-g".to_string());
            options.push("-opt=0".to_string());
        }
        options
    }

    pub(crate) fn nvjitlink_options(&self, output: FinalizerOutput) -> Vec<String> {
        let mut options = vec![format!("-arch={}", self.target.sm()), "-lto".to_string()];
        if output == FinalizerOutput::Ptx {
            options.push("-ptx".to_string());
        }
        options.push(self.fma_option().to_string());
        match self.debug {
            DebugPolicy::None => {}
            DebugPolicy::LineTables => options.push("-lineinfo".to_string()),
            DebugPolicy::Full => options.push("-g".to_string()),
        }
        options
    }

    fn fma_option(&self) -> &'static str {
        if self.allow_fma_contraction {
            "-fma=1"
        } else {
            "-fma=0"
        }
    }
}

/// Final artifact requested from nvJitLink.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FinalizerOutput {
    /// Native, target-specific CUDA ELF image.
    Cubin,
    /// Forward-compatible PTX assembly.
    Ptx,
}

/// One named linker input. Slice order is preserved exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NamedInput<'a> {
    /// Name shown by CUDA-tool diagnostics and included in provenance.
    pub name: &'a str,
    /// Complete LTOIR input bytes.
    pub bytes: &'a [u8],
}

impl<'a> NamedInput<'a> {
    /// Construct a named input without copying its bytes.
    pub const fn new(name: &'a str, bytes: &'a [u8]) -> Self {
        Self { name, bytes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_order_preserves_target_lto_output_fma_and_debug_policy() {
        let target: CudaArch = "sm_90a".parse().unwrap();
        let base = FinalizationOptions::new(target).with_fma_contraction(false);

        assert_eq!(
            base.nvvm_compile_options(),
            ["-arch=compute_90a", "-gen-lto", "-fma=0"]
        );
        assert_eq!(
            base.nvjitlink_options(FinalizerOutput::Cubin),
            ["-arch=sm_90a", "-lto", "-fma=0"]
        );
        assert_eq!(
            base.clone()
                .with_debug_policy(DebugPolicy::LineTables)
                .nvjitlink_options(FinalizerOutput::Ptx),
            ["-arch=sm_90a", "-lto", "-ptx", "-fma=0", "-lineinfo"]
        );
        assert_eq!(
            base.clone()
                .with_debug_policy(DebugPolicy::LineTables)
                .nvvm_compile_options(),
            ["-arch=compute_90a", "-gen-lto", "-fma=0"]
        );
        assert_eq!(
            base.clone()
                .with_debug_policy(DebugPolicy::Full)
                .nvvm_compile_options(),
            ["-arch=compute_90a", "-gen-lto", "-fma=0", "-g", "-opt=0"]
        );
        assert_eq!(
            base.with_debug_policy(DebugPolicy::Full)
                .nvjitlink_options(FinalizerOutput::Cubin),
            ["-arch=sm_90a", "-lto", "-fma=0", "-g"]
        );
    }

    #[test]
    fn fma_policy_is_explicit_in_both_stages() {
        let target: CudaArch = "sm_120".parse().unwrap();
        for allow in [false, true] {
            let options = FinalizationOptions::new(target.clone()).with_fma_contraction(allow);
            let expected = if allow { "-fma=1" } else { "-fma=0" };
            assert_eq!(
                options
                    .nvvm_compile_options()
                    .iter()
                    .filter(|option| option.starts_with("-fma="))
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                [expected]
            );
            assert_eq!(
                options
                    .nvjitlink_options(FinalizerOutput::Cubin)
                    .iter()
                    .filter(|option| option.starts_with("-fma="))
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                [expected]
            );
        }
    }
}
