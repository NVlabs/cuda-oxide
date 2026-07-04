/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::llvm_tools::{LlvmToolchain, OptTool, probe_runnable};
use crate::options::BackendOptions;
use crate::pipeline::{
    ModuleArtifactKind, ModulePipelineRequest, OutputFiles, compile_translated_module,
};
use libnvvm_sys::CudaArch;
use llvm_export::export::DebugKind;
use pliron::builtin::{
    attributes::{IdentifierAttr, StringAttr},
    op_interfaces::ATTR_KEY_SYM_NAME,
    ops::ModuleOp,
};
use pliron::context::Context;
use pliron::identifier::Identifier;
use pliron::irbuild::{
    cloning::{IrMapping, clone_operation},
    listener::DummyListener,
    rewriter::IRRewriter,
};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

static COMPILATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A validated CUDA GPU target such as `sm_80`, `sm_90a`, or `sm_120`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Target(CudaArch);

impl Target {
    /// Parse and validate a CUDA target.
    pub fn parse(target: &str) -> Result<Self, CompileError> {
        target.parse()
    }

    /// Render the canonical `sm_XX` spelling used by LLVM and PTX.
    pub fn sm(&self) -> String {
        self.0.sm()
    }

    /// Numeric CUDA compute capability (`80`, `90`, `120`, ...).
    pub fn capability(&self) -> u32 {
        self.0.capability()
    }

    /// Optional architecture-family suffix (`a` or `f`).
    pub fn suffix(&self) -> Option<char> {
        self.0.suffix()
    }
}

impl FromStr for Target {
    type Err = CompileError;

    fn from_str(target: &str) -> Result<Self, Self::Err> {
        let trimmed = target.trim();
        let arch = trimmed
            .parse::<CudaArch>()
            .map_err(|error| CompileError::InvalidTarget {
                target: target.to_string(),
                reason: error.to_string(),
            })?;
        Ok(Self(arch))
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// LLVM middle-end policy for one compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Optimization {
    /// Feed verified, unoptimized LLVM IR directly to `llc`.
    None,
    /// Require a same-major `opt` and run `opt -O2`.
    #[default]
    O2,
}

/// Device debug information policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugInfo {
    /// Do not emit debug metadata.
    #[default]
    None,
    /// Preserve variables and compile at `-O0` for cuda-gdb-style debugging.
    Full,
}

/// Typed options for one standalone PTX compilation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CompileOptions {
    target: Target,
    optimization: Optimization,
    fma_contraction: bool,
    debug_info: DebugInfo,
}

impl CompileOptions {
    /// Create optimized, non-debug options for `target`.
    pub fn new(target: Target) -> Self {
        Self {
            target,
            optimization: Optimization::O2,
            fma_contraction: true,
            debug_info: DebugInfo::None,
        }
    }

    /// Select the LLVM optimization policy.
    pub fn with_optimization(mut self, optimization: Optimization) -> Self {
        self.optimization = optimization;
        self
    }

    /// Allow or forbid ordinary multiply/add contraction into FMA.
    pub fn with_fma_contraction(mut self, allow: bool) -> Self {
        self.fma_contraction = allow;
        self
    }

    /// Select device debug information.
    ///
    /// [`DebugInfo::Full`] must be paired with [`Optimization::None`].
    pub fn with_debug_info(mut self, debug_info: DebugInfo) -> Self {
        self.debug_info = debug_info;
        self
    }

    /// Requested CUDA target.
    pub fn target(&self) -> &Target {
        &self.target
    }

    /// Requested LLVM optimization policy.
    pub fn optimization(&self) -> Optimization {
        self.optimization
    }

    /// Whether ordinary floating-point operations may contract into FMA.
    pub fn fma_contraction(&self) -> bool {
        self.fma_contraction
    }

    /// Requested device debug-information tier.
    pub fn debug_info(&self) -> DebugInfo {
        self.debug_info
    }
}

/// Severity of a compiler diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// Informational context about a successful operation.
    Note,
    /// A recoverable condition the caller may want to show.
    Warning,
    /// A compilation failure.
    Error,
}

/// Stage that produced a diagnostic or error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompilationStage {
    /// Module construction, target parsing, or option validation.
    Input,
    /// dialect-mir verification, mem2reg, or annotated loop unrolling.
    MirPreparation,
    /// dialect-mir to LLVM-dialect conversion.
    Lowering,
    /// Validation that the PTX artifact is self-contained.
    Linking,
    /// LLVM-dialect to textual LLVM IR export.
    Export,
    /// LLVM `opt` execution.
    Optimization,
    /// LLVM `llc` execution or PTX file handling.
    Codegen,
    /// LLVM tool discovery and compatibility checks.
    Toolchain,
}

/// Structured compiler diagnostic retained for the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Diagnostic {
    /// Diagnostic severity.
    pub level: DiagnosticLevel,
    /// Compiler stage that produced the diagnostic.
    pub stage: CompilationStage,
    /// Human-readable detail suitable for logs or a UI.
    pub message: String,
}

impl Diagnostic {
    fn note(stage: CompilationStage, message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Note,
            stage,
            message: message.into(),
        }
    }

    fn warning(stage: CompilationStage, message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            stage,
            message: message.into(),
        }
    }
}

/// A cuda-oxide module paired permanently with the Pliron context that owns it.
///
/// Pliron pointers are arena keys and do not carry a context identity. The
/// compiler therefore retains the owner of its root module. Raw handles that a
/// callback copies out of [`CodegenModule::edit`] or [`CodegenModule::inspect`]
/// remain the caller's responsibility.
pub struct CodegenModule {
    context: Context,
    module: ModuleOp,
}

impl CodegenModule {
    /// Create a module and register every dialect accepted by experimental v1.
    pub fn new(name: &str) -> Result<Self, CompileError> {
        let name: Identifier = name
            .try_into()
            .map_err(|_| CompileError::InvalidModuleName {
                name: name.to_string(),
            })?;
        let mut context = Context::new();
        dialect_mir::register(&mut context);
        dialect_nvvm::register(&mut context);
        let module = ModuleOp::new(&mut context, name);
        Ok(Self { context, module })
    }

    /// Build or edit IR through the context that owns this compiler root.
    ///
    /// The callback can still copy raw Pliron handles out of this method.
    /// Callers must not resolve those handles in another context or after the
    /// referenced operation has been erased.
    pub fn edit<R>(&mut self, edit: impl FnOnce(&mut Context, &ModuleOp) -> R) -> R {
        edit(&mut self.context, &self.module)
    }

    /// Inspect IR through the context that owns this compiler root.
    ///
    /// Returning a raw Pliron handle does not attach a context identity to it;
    /// the caller remains responsible for its later use.
    pub fn inspect<R>(&self, inspect: impl FnOnce(&Context, &ModuleOp) -> R) -> R {
        inspect(&self.context, &self.module)
    }

    /// Mark one owned top-level `dialect-mir` function as a PTX kernel entry.
    ///
    /// `symbol` is resolved inside this module's own context. Device helper
    /// functions should not be marked. Compilation copies the marker to the
    /// lowered LLVM function and exports it as a PTX `.entry`.
    pub fn mark_kernel_entry(&mut self, symbol: &str) -> Result<(), CompileError> {
        let module = self.module.get_operation();
        module
            .try_deref(&self.context)
            .map_err(|error| CompileError::InvalidModule {
                message: format!("the owned module operation is no longer valid: {error:?}"),
            })?;
        if Operation::get_op::<ModuleOp>(module, &self.context).is_none() {
            return Err(CompileError::InvalidModule {
                message: "the owned module pointer no longer refers to a builtin.module"
                    .to_string(),
            });
        }
        let function = {
            let region_count = module.deref(&self.context).regions().count();
            if region_count != 1 {
                return Err(CompileError::InvalidModule {
                    message: format!(
                        "cannot mark `{symbol}` as a kernel: the module has {region_count} regions, expected exactly one"
                    ),
                });
            }
            let region = module.deref(&self.context).get_region(0);
            let blocks: Vec<_> = region.deref(&self.context).iter(&self.context).collect();
            if blocks.len() != 1 {
                return Err(CompileError::InvalidModule {
                    message: format!(
                        "cannot mark `{symbol}` as a kernel: the module has {} top-level blocks, expected exactly one",
                        blocks.len()
                    ),
                });
            }
            let block = blocks[0];
            let mut found = None;
            for operation in block.deref(&self.context).iter(&self.context) {
                let Some(candidate) =
                    Operation::get_op::<dialect_mir::ops::MirFuncOp>(operation, &self.context)
                else {
                    continue;
                };
                let operation_ref = candidate.get_operation().deref(&self.context);
                let Some(symbol_attr) = operation_ref
                    .attributes
                    .get::<IdentifierAttr>(&ATTR_KEY_SYM_NAME)
                else {
                    return Err(CompileError::InvalidModule {
                        message:
                            "cannot mark a kernel: a top-level dialect-mir function has no symbol"
                                .to_string(),
                    });
                };
                let candidate_symbol: Identifier = symbol_attr.clone().into();
                if candidate_symbol.to_string() != symbol {
                    continue;
                }
                if found.replace(operation).is_some() {
                    return Err(CompileError::InvalidModule {
                        message: format!(
                            "cannot mark `{symbol}` as a kernel: duplicate top-level dialect-mir functions"
                        ),
                    });
                }
            }
            found.ok_or_else(|| CompileError::InvalidModule {
                message: format!(
                    "cannot mark `{symbol}` as a kernel: no top-level dialect-mir function has that symbol"
                ),
            })?
        };

        let key: Identifier = "gpu_kernel"
            .try_into()
            .expect("gpu_kernel is a valid Pliron identifier");
        function
            .deref_mut(&self.context)
            .attributes
            .set(key, StringAttr::new("true".to_string()));
        Ok(())
    }
}

/// Explicit, reusable pair of LLVM tools.
#[derive(Clone, Debug)]
pub struct Toolchain {
    inner: LlvmToolchain,
    diagnostics: Vec<Diagnostic>,
}

impl Toolchain {
    /// Discover LLVM 21+ tools from the Rust sysroot and `PATH`.
    ///
    /// Discovery is explicit and does not read cuda-oxide environment knobs.
    /// It may return a toolchain without `opt`; that toolchain can compile only
    /// with [`Optimization::None`].
    pub fn discover() -> Result<Self, CompileError> {
        let opts = BackendOptions::default();
        let inner = LlvmToolchain::resolve(&opts).ok_or_else(|| CompileError::Toolchain {
            message: "no runnable LLVM 21+ `llc` was found in the Rust sysroot or PATH".to_string(),
        })?;
        validate_llvm_major("llc", inner.llc_major)?;
        let diagnostics = inner
            .diagnostics
            .iter()
            .cloned()
            .map(|message| Diagnostic::warning(CompilationStage::Toolchain, message))
            .collect();
        Ok(Self { inner, diagnostics })
    }

    /// Use explicit LLVM tools, verifying that they run and share one major.
    pub fn from_paths(llc: impl Into<PathBuf>, opt: Option<PathBuf>) -> Result<Self, CompileError> {
        let llc = llc.into();
        let llc_text = llc.to_string_lossy().into_owned();
        let llc_tool = probe_runnable(&llc_text).ok_or_else(|| CompileError::Toolchain {
            message: format!("`{llc_text} --version` did not run successfully"),
        })?;
        validate_llvm_major("llc", llc_tool.major)?;

        let opt = opt
            .map(|path| {
                let text = path.to_string_lossy().into_owned();
                probe_runnable(&text).ok_or_else(|| CompileError::Toolchain {
                    message: format!("`{text} --version` did not run successfully"),
                })
            })
            .transpose()?;

        if let Some(opt) = &opt {
            validate_llvm_major("opt", opt.major)?;
            if opt.major != llc_tool.major {
                return Err(CompileError::Toolchain {
                    message: format!(
                        "LLVM major mismatch: llc is {}, but opt is {}",
                        llc_tool.major.unwrap(),
                        opt.major.unwrap()
                    ),
                });
            }
        }

        Ok(Self {
            inner: LlvmToolchain {
                llc_path: llc_tool.path,
                llc_major: llc_tool.major,
                llc_from_env: false,
                opt: opt.map(|tool| OptTool {
                    path: tool.path,
                    major: tool.major,
                }),
                diagnostics: Vec::new(),
            },
            diagnostics: Vec::new(),
        })
    }

    /// Selected `llc` path.
    pub fn llc_path(&self) -> &Path {
        Path::new(&self.inner.llc_path)
    }

    /// Selected `opt` path, or `None` when optimization is unavailable.
    pub fn opt_path(&self) -> Option<&Path> {
        self.inner.opt.as_ref().map(|opt| Path::new(&opt.path))
    }

    /// Diagnostics recorded while selecting a matched tool pair.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

fn validate_llvm_major(tool: &str, major: Option<u32>) -> Result<(), CompileError> {
    match major {
        Some(major) if major >= 21 => Ok(()),
        Some(major) => Err(CompileError::Toolchain {
            message: format!("{tool} reports LLVM {major}; cuda-oxide requires LLVM 21 or newer"),
        }),
        None => Err(CompileError::Toolchain {
            message: format!("could not parse the LLVM major from `{tool} --version`"),
        }),
    }
}

/// Reusable standalone compiler.
#[derive(Clone, Debug)]
pub struct Compiler {
    toolchain: Toolchain,
}

impl Compiler {
    /// Create a compiler that reuses `toolchain` for every invocation.
    pub fn new(toolchain: Toolchain) -> Self {
        Self { toolchain }
    }

    /// Explicit convenience for callers that want default tool discovery.
    pub fn discover() -> Result<Self, CompileError> {
        Toolchain::discover().map(Self::new)
    }

    /// The explicit LLVM toolchain used by this compiler.
    pub fn toolchain(&self) -> &Toolchain {
        &self.toolchain
    }

    /// Compile a module without modifying its caller-visible IR.
    pub fn compile(
        &self,
        module: &mut CodegenModule,
        options: &CompileOptions,
    ) -> Result<Compilation, CompileError> {
        if options.debug_info == DebugInfo::Full && options.optimization != Optimization::None {
            return Err(CompileError::InvalidOptions {
                message: "full variable debug information requires Optimization::None".to_string(),
            });
        }
        if options.optimization == Optimization::O2 && self.toolchain.inner.opt.is_none() {
            return Err(CompileError::OptimizationUnavailable {
                message:
                    "Optimization::O2 requires an `opt` binary with the same LLVM major as llc"
                        .to_string(),
            });
        }

        let ctx = &mut module.context;
        let source_module = module.module.get_operation();
        source_module
            .try_deref(ctx)
            .map_err(|error| CompileError::InvalidModule {
                message: format!("the owned module operation is no longer valid: {error:?}"),
            })?;
        if Operation::get_op::<ModuleOp>(source_module, ctx).is_none() {
            return Err(CompileError::InvalidModule {
                message: "the owned module pointer no longer refers to a builtin.module"
                    .to_string(),
            });
        }
        let mut mapper = IrMapping::new();
        let mut rewriter = IRRewriter::<DummyListener>::default();
        let cloned = clone_operation(source_module, ctx, &mut rewriter, &mut mapper);
        let result = self.compile_clone(ctx, cloned, options);
        Operation::erase(cloned, ctx);
        result
    }

    fn compile_clone(
        &self,
        ctx: &mut Context,
        module: pliron::context::Ptr<Operation>,
        options: &CompileOptions,
    ) -> Result<Compilation, CompileError> {
        let debug_kind = match options.debug_info {
            DebugInfo::None => DebugKind::Off,
            DebugInfo::Full => DebugKind::Full,
        };
        let scratch = ScratchDirectory::new()?;
        let ll_path = scratch.path().join("module.ll");
        let ptx_path = scratch.path().join("module.ptx");
        let backend_options = BackendOptions {
            target_arch: Some(options.target.sm()),
            device_arch_hint: None,
            no_opt: options.optimization == Optimization::None,
            no_fma: !options.fma_contraction,
            verbose: false,
            llc_override: None,
            opt_override: None,
        };
        let request = ModulePipelineRequest::for_standalone_ptx(
            &backend_options,
            debug_kind,
            &self.toolchain.inner,
            OutputFiles {
                llvm_ir: &ll_path,
                ptx: &ptx_path,
                stale_before_export: &[],
            },
        );

        (|| {
            let generated =
                compile_translated_module(ctx, module, &request).map_err(CompileError::from)?;
            debug_assert_eq!(generated.artifact_kind, ModuleArtifactKind::Ptx);
            let ptx = std::fs::read(&ptx_path).map_err(|source| CompileError::Io {
                action: "read generated PTX",
                path: ptx_path.clone(),
                source,
            })?;
            let target = Target::parse(&generated.target)?;
            let mut diagnostics = self.toolchain.diagnostics.clone();
            diagnostics.push(Diagnostic::note(
                CompilationStage::Codegen,
                format!("generated PTX for {target}"),
            ));
            Ok(Compilation {
                ptx,
                target,
                diagnostics,
            })
        })()
    }
}

struct ScratchDirectory {
    path: PathBuf,
}

impl ScratchDirectory {
    fn new() -> Result<Self, CompileError> {
        let root = std::env::temp_dir();
        for _ in 0..32 {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0);
            let call = COMPILATION_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "cuda_oxide_codegen_{}_{}_{}",
                std::process::id(),
                unique,
                call
            ));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => {
                    return Err(CompileError::Io {
                        action: "create compilation scratch directory",
                        path,
                        source,
                    });
                }
            }
        }
        Err(CompileError::Io {
            action: "create a unique compilation scratch directory",
            path: root,
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "32 atomically-created names already existed",
            ),
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Successful PTX compilation.
#[derive(Clone, Debug)]
pub struct Compilation {
    ptx: Vec<u8>,
    target: Target,
    diagnostics: Vec<Diagnostic>,
}

impl Compilation {
    /// Generated PTX bytes.
    pub fn ptx(&self) -> &[u8] {
        &self.ptx
    }

    /// Consume the result and return its PTX bytes.
    pub fn into_ptx(self) -> Vec<u8> {
        self.ptx
    }

    /// Concrete CUDA target recorded in the PTX.
    pub fn target(&self) -> &Target {
        &self.target
    }

    /// Non-fatal diagnostics retained during compilation.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

/// Structured failure from the experimental compiler.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CompileError {
    /// The requested Pliron symbol name is invalid.
    #[error("invalid module name `{name}`")]
    InvalidModuleName {
        /// Rejected module name.
        name: String,
    },
    /// CUDA target text could not be parsed.
    #[error("invalid CUDA target `{target}`: {reason}")]
    InvalidTarget {
        /// Rejected target text.
        target: String,
        /// Parser explanation.
        reason: String,
    },
    /// The owned top-level module was erased or replaced through an edit.
    #[error("invalid codegen module: {message}")]
    InvalidModule {
        /// Validation detail.
        message: String,
    },
    /// Compile options conflict with one another.
    #[error("invalid compile options: {message}")]
    InvalidOptions {
        /// Validation detail.
        message: String,
    },
    /// LLVM tools were missing, too old, or from different releases.
    #[error("LLVM toolchain is unavailable or incompatible: {message}")]
    Toolchain {
        /// Tool discovery or compatibility detail.
        message: String,
    },
    /// Optimization was requested without a usable matching `opt`.
    #[error("optimization is unavailable: {message}")]
    OptimizationUnavailable {
        /// Missing-tool detail.
        message: String,
    },
    /// The requested LLVM optimization pass failed.
    #[error("LLVM optimization failed: {message}")]
    OptimizationFailed {
        /// Tool failure and captured stderr.
        message: String,
    },
    /// Input or transformed IR failed structural verification.
    #[error("input verification failed: {message}")]
    Verification {
        /// Verifier explanation.
        message: String,
        /// Printed failing operation, when Pliron identified one.
        operation: Option<String>,
    },
    /// MIR operations could not be lowered to the LLVM dialect.
    #[error("MIR to LLVM lowering failed: {message}")]
    Lowering {
        /// Lowering explanation.
        message: String,
    },
    /// The lowered LLVM-dialect module failed structural verification.
    #[error("lowered LLVM module verification failed: {message}")]
    LoweredVerification {
        /// Verifier explanation.
        message: String,
        /// Printed failing operation, when Pliron identified one.
        operation: Option<String>,
    },
    /// V1 found declarations that require a link step it does not provide.
    #[error("standalone PTX cannot resolve external symbols: {symbols:?}")]
    UnsupportedLinking {
        /// Sorted, deduplicated unresolved symbol names.
        symbols: Vec<String>,
    },
    /// LLVM text export failed.
    #[error("LLVM IR export failed: {message}")]
    Export {
        /// Exporter explanation.
        message: String,
    },
    /// `llc` failed or the generated PTX could not be read.
    #[error("PTX code generation failed: {message}")]
    Codegen {
        /// Tool failure and captured stderr.
        message: String,
    },
    /// Scratch artifact I/O failed.
    #[error("failed to {action} at {}: {source}", path.display())]
    Io {
        /// Operation attempted.
        action: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Operating-system error.
        #[source]
        source: std::io::Error,
    },
}

impl CompileError {
    /// Compiler stage that failed.
    pub fn stage(&self) -> CompilationStage {
        match self {
            Self::InvalidModuleName { .. }
            | Self::InvalidTarget { .. }
            | Self::InvalidModule { .. }
            | Self::InvalidOptions { .. } => CompilationStage::Input,
            Self::Toolchain { .. } => CompilationStage::Toolchain,
            Self::OptimizationUnavailable { .. } | Self::OptimizationFailed { .. } => {
                CompilationStage::Optimization
            }
            Self::Verification { .. } => CompilationStage::MirPreparation,
            Self::Lowering { .. } | Self::LoweredVerification { .. } => CompilationStage::Lowering,
            Self::UnsupportedLinking { .. } => CompilationStage::Linking,
            Self::Export { .. } => CompilationStage::Export,
            Self::Codegen { .. } | Self::Io { .. } => CompilationStage::Codegen,
        }
    }

    /// Convert this failure to a structured error diagnostic.
    pub fn diagnostic(&self) -> Diagnostic {
        Diagnostic {
            level: DiagnosticLevel::Error,
            stage: self.stage(),
            message: self.to_string(),
        }
    }
}

impl From<PipelineError> for CompileError {
    fn from(error: PipelineError) -> Self {
        match error {
            PipelineError::Verification {
                message, operation, ..
            } => Self::Verification { message, operation },
            PipelineError::Lowering(message) => Self::Lowering { message },
            PipelineError::LoweredVerification { message, operation } => {
                Self::LoweredVerification { message, operation }
            }
            PipelineError::UnsupportedLinking { symbols } => Self::UnsupportedLinking { symbols },
            PipelineError::Export(message) => Self::Export { message },
            PipelineError::PtxGeneration(message) => Self::Codegen { message },
            PipelineError::Optimization(message) => Self::OptimizationFailed { message },
            PipelineError::NoBody(message) | PipelineError::Translation(message) => {
                Self::Verification {
                    message,
                    operation: None,
                }
            }
        }
    }
}
