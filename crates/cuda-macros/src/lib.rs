/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod device_copy;
mod printf;
mod ptx_asm;

use proc_macro::TokenStream;

/// GPU printf macro for formatted output from GPU kernels.
///
/// This macro translates Rust-style format strings to C-style and calls
/// CUDA's `vprintf` function.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::gpu_printf;
///
/// #[kernel]
/// fn my_kernel() {
///     let tid = thread::index_1d().get();
///     gpu_printf!("Thread {}: Hello from GPU!\n", tid);
/// }
/// ```
///
/// # Format Specifiers
///
/// | Specifier | Description     | Example                                        |
/// |-----------|-----------------|------------------------------------------------|
/// | `{}`      | Default format  | `gpu_printf!("{}", 42)`                        |
/// | `{:x}`    | Hex (lower)     | `gpu_printf!("{:x}", 255)` → "ff"              |
/// | `{:X}`    | Hex (upper)     | `gpu_printf!("{:X}", 255)` → "FF"              |
/// | `{:#x}`   | Hex with prefix | `gpu_printf!("{:#x}", 255)` → "0xff"           |
/// | `{:o}`    | Octal           | `gpu_printf!("{:o}", 8)` → "10"                |
/// | `{:e}`    | Scientific      | `gpu_printf!("{:e}", 1000.0)` → "1.000000e+03" |
/// | `{:.N}`   | Precision       | `gpu_printf!("{:.2}", 3.14159)` → "3.14"       |
/// | `{:N}`    | Width           | `gpu_printf!("{:8}", 42)` → "      42"         |
/// | `{:0N}`   | Zero-pad        | `gpu_printf!("{:08}", 42)` → "00000042"        |
///
/// # Returns
///
/// The number of arguments (i32), or negative on error.
/// Note: CUDA vprintf returns arg count, not character count.
#[proc_macro]
pub fn gpu_printf(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as printf::GpuPrintfInput);
    printf::gpu_printf_impl(input).into()
}

/// Inline PTX assembly for CUDA device code.
///
/// The macro accepts the `%0` operand placeholders used by CUDA inline PTX, plus
/// `in` / `out` operands with PTX constraint strings. Supported constraints are
/// CUDA register constraints `"h"`, `"r"`, `"l"`, `"q"`, `"f"`, and `"d"`, plus
/// `"n"` for immediate integer constants:
///
/// ```ignore
/// let y: u32;
/// unsafe {
///     ptx_asm!(
///         "add.u32 %0, %1, %2;",
///         out("=r") y,
///         in("r") x,
///         in("r") z,
///     );
/// }
/// ```
///
/// Literal PTX registers that begin with `%` must be escaped as `%%`, matching
/// CUDA C++ inline PTX. Literal `$` labels can be written normally.
///
/// The initial surface supports zero or one `out`, up to 16 `in` operands,
/// `clobber("memory")`, `options(register_only)`, and the explicit
/// `options(register_only, may_diverge)` opt-in.
///
/// By default, snippets are treated as side-effecting and stay inside their
/// current control flow. Use `options(register_only)` only for snippets that
/// read explicit operands and write the explicit output. **Never** use
/// `may_diverge` for `.sync` instructions, collectives, or any snippet whose
/// participating lanes matter.
#[proc_macro]
pub fn ptx_asm(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as ptx_asm::PtxAsmInput);
    ptx_asm::ptx_asm_impl(input).into()
}

/// Derive `cuda_core::DeviceCopy` for a type whose fields are all themselves
/// `DeviceCopy`.
///
/// Re-exported from `cuda_core` next to the `DeviceCopy` trait so that
/// `use cuda_core::DeviceCopy;` brings both the trait and this derive into scope
/// (the serde `Serialize` trait+derive pattern).
#[proc_macro_derive(DeviceCopy)]
pub fn device_copy(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    let code = device_copy::impl_device_copy(&ast, quote!(::cuda_core::DeviceCopy));
    code.into()
}
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use reserved_oxide_symbols::{
    DEVICE_EXTERN_PREFIX, DEVICE_PREFIX, INSTANTIATE_PREFIX, KERNEL_PREFIX, KERNEL_SCOPE_LOCAL,
    RESERVED_ROOT, artifact_anchor_symbol, constant_symbol, kernel_symbol,
};
use syn::{
    Expr, ExprCall, ExprMethodCall, ExprPath, FnArg, ForeignItem, GenericArgument, GenericParam,
    Ident, Item, ItemFn, ItemForeignMod, ItemMod, LitStr, Pat, Path, PathArguments, Stmt, Token,
    Type, bracketed, parenthesized,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
};

/// Reject function names that start with the reserved cuda-oxide prefix
/// (`cuda_oxide_`).
///
/// User code must not define functions in the cuda-oxide internal naming
/// namespace. Two failure modes this guards against:
///
/// 1. **Cosmetic.** `#[kernel] fn cuda_oxide_kernel_foo()` would expand to
///    a doubly-nested name like
///    `fn cuda_oxide_kernel_<hash>_cuda_oxide_kernel_foo()`, producing
///    confusing symbol names in MIR dumps and stack traces.
/// 2. **Forward-compatibility.** Future refactors may extend the namespace;
///    rejecting it at the source level keeps the contract clean.
///
/// Returns `Some(compile_error)` to be returned from the macro entry point,
/// or `None` if the name is safe.
fn reject_reserved_name(name: &Ident) -> Option<TokenStream> {
    let name_str = name.to_string();
    if name_str.starts_with(RESERVED_ROOT) {
        let msg = format!(
            "function name `{name_str}` starts with the reserved cuda-oxide \
             prefix `{RESERVED_ROOT}`; rename your function — this namespace \
             is reserved for cuda-oxide internal symbol mangling \
             (see crates/reserved-oxide-symbols)"
        );
        Some(syn::Error::new(name.span(), msg).to_compile_error().into())
    } else {
        None
    }
}

/// Attribute arguments for #[kernel(...)]
/// Supports: #[kernel] or #[kernel(Type1, Type2, Type3)]
struct KernelArgs {
    /// Types to instantiate generic kernels for
    instantiate_types: Vec<Type>,
}

impl Parse for KernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(KernelArgs {
                instantiate_types: vec![],
            });
        }

        let types: Punctuated<Type, Token![,]> = Punctuated::parse_terminated(input)?;
        Ok(KernelArgs {
            instantiate_types: types.into_iter().collect(),
        })
    }
}

/// Generates a typed host-side loader and launch surface for the kernels in an
/// inline module.
///
/// The generated API loads the current crate's embedded artifact bundle and
/// exposes synchronous methods per `#[kernel]` function. When `cuda-host` is
/// built with its `async` feature, the macro also emits borrowed async and owned
/// async methods. Kernel parameter types are mapped to host-side launch types:
///
/// - `&[T]` -> `&cuda_core::DeviceBuffer<T>`
/// - `&mut [T]` -> `&mut cuda_core::DeviceBuffer<T>`
/// - `DisjointSlice<T>` -> `&mut cuda_core::DeviceBuffer<T>`
/// - `Copy` scalar/struct/closure/raw-pointer arguments keep their original
///   type and pass through `cuda_host::KernelScalar`
///
/// # Example
///
/// ```ignore
/// #[cuda_module]
/// mod kernels {
///     use super::*;
///
///     #[kernel]
///     pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///         // ...
///     }
/// }
///
/// let module = kernels::load(&ctx)?;
/// module.vecadd(&stream, LaunchConfig::for_num_elems(n), &a, &b, &mut c)?;
///
/// let module = kernels::load_async(0)?;
/// module.vecadd_async(LaunchConfig::for_num_elems(n), &a, &b, &mut c)?.sync()?;
///
/// let (a, b, c) = module
///     .vecadd_async_owned(LaunchConfig::for_num_elems(n), a, b, c)?
///     .await?;
/// ```
#[proc_macro_attribute]
pub fn cuda_module(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "cuda_module does not take arguments yet",
        )
        .to_compile_error()
        .into();
    }

    let input = parse_macro_input!(item as ItemMod);
    match expand_cuda_module(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

struct CudaModuleKernel {
    vis: syn::Visibility,
    cfg_attrs: Vec<syn::Attribute>,
    method_attrs: Vec<syn::Attribute>,
    unsafety: Option<Token![unsafe]>,
    fn_name: Ident,
    generics: syn::Generics,
    params: Vec<CudaModuleParam>,
    cluster_dim: Option<(u32, u32, u32)>,
    cooperative: bool,
    is_generic: bool,
}

struct CudaModuleParam {
    name: Ident,
    sync_host_ty: TokenStream2,
    async_host_ty: TokenStream2,
    marshal: CudaModuleParamMarshal,
}

enum CudaModuleParamMarshal {
    Scalar,
    ReadOnlyDeviceBuffer { elem_ty: TokenStream2 },
    WritableDeviceBuffer { elem_ty: TokenStream2 },
}

fn expand_cuda_module(module: ItemMod) -> syn::Result<TokenStream2> {
    let module_attrs = &module.attrs;
    let vis = &module.vis;
    let ident = &module.ident;
    let Some((_brace, items)) = &module.content else {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "cuda_module requires an inline module so kernel signatures are visible",
        ));
    };

    let kernels = collect_cuda_module_kernels(items)?;
    if kernels.is_empty() {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "cuda_module found no #[kernel] functions in this module",
        ));
    }
    let constants = collect_cuda_module_constants(items, ident)?;
    let module_items = cuda_module_items_with_constant_symbols(items, &constants);

    let non_generic_kernels = kernels.iter().filter(|kernel| !kernel.is_generic);
    let function_fields = non_generic_kernels.clone().map(|kernel| {
        let cfg_attrs = &kernel.cfg_attrs;
        let field = cuda_module_function_field(&kernel.fn_name);
        quote! {
            #(#cfg_attrs)*
            #field: ::cuda_core::CudaFunction,
        }
    });

    let function_initializers = non_generic_kernels.map(|kernel| {
        let cfg_attrs = &kernel.cfg_attrs;
        let field = cuda_module_function_field(&kernel.fn_name);
        let marker = cuda_kernel_marker_name(&kernel.fn_name);
        quote! {
            #(#cfg_attrs)*
            #field: module.load_function(<#marker as ::cuda_host::CudaKernel>::PTX_NAME)?,
        }
    });

    let artifact_anchor_statements = cuda_module_artifact_anchor_statements(&kernels)?;
    let has_generic = kernels.iter().any(|k| k.is_generic);
    let module_loader = if has_generic {
        // At least one kernel is generic: its PTX is emitted into the
        // consuming binary's bundle, not this crate's bundle. Merge all
        // PTX bundles from the executable so generic monomorphizations are
        // visible regardless of which crate compiled them.
        quote! {
            let _ = name; // merged load ignores the crate-name hint
            let module = ::cuda_host::load_all_ptx_bundles_merged(ctx)?;
        }
    } else {
        quote! {
            let module = ::cuda_host::load_embedded_module(ctx, name)?;
        }
    };
    let constant_fields = constants.iter().map(generate_cuda_module_constant_field);
    let constant_initializers = constants
        .iter()
        .map(generate_cuda_module_constant_initializer);
    let launch_methods = kernels.iter().map(generate_cuda_module_launch_method);
    let constant_resolver_methods = constants
        .iter()
        .map(generate_cuda_module_constant_resolver_method);
    let set_constant_methods = constants
        .iter()
        .map(generate_cuda_module_set_constant_method);
    let async_module_items = if cfg!(feature = "async") {
        quote! {
            pub fn load_async(
                device_id: usize,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::cuda_async::error::DeviceError> {
                load_async_named(device_id, env!("CARGO_PKG_NAME"))
            }

            pub fn load_async_named(
                device_id: usize,
                name: &str,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::cuda_async::error::DeviceError> {
                ::cuda_host::load_cuda_module_from_async_context(device_id, |ctx| load_named(ctx, name))
            }
        }
    } else {
        TokenStream2::new()
    };
    let async_launch_methods = if cfg!(feature = "async") {
        let async_launch_methods = kernels.iter().map(generate_cuda_module_async_launch_method);
        let owned_async_launch_methods = kernels
            .iter()
            .map(generate_cuda_module_owned_async_launch_method);
        quote! {
            #(#async_launch_methods)*
            #(#owned_async_launch_methods)*
        }
    } else {
        TokenStream2::new()
    };

    Ok(quote! {
        #(#module_attrs)*
        #vis mod #ident {
            #(#module_items)*

            #[derive(Clone, Debug)]
            #[allow(non_snake_case)]
            pub struct LoadedModule {
                __module: ::std::sync::Arc<::cuda_core::CudaModule>,
                __generic_functions: ::std::sync::Arc<
                    ::std::sync::Mutex<
                        ::std::collections::HashMap<&'static str, ::cuda_core::CudaFunction>
                    >
                >,
                #(#function_fields)*
                #(#constant_fields)*
            }

            pub fn load(
                ctx: &::std::sync::Arc<::cuda_core::CudaContext>,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::EmbeddedModuleError> {
                load_named(ctx, env!("CARGO_PKG_NAME"))
            }

            pub fn load_named(
                ctx: &::std::sync::Arc<::cuda_core::CudaContext>,
                name: &str,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::EmbeddedModuleError> {
                #artifact_anchor_statements
                #module_loader
                from_module(module).map_err(::cuda_host::EmbeddedModuleError::Driver)
            }

            pub fn from_module(
                module: ::std::sync::Arc<::cuda_core::CudaModule>,
            ) -> ::core::result::Result<LoadedModule, ::cuda_core::DriverError> {
                Ok(LoadedModule {
                    __module: module.clone(),
                    __generic_functions: ::std::sync::Arc::new(
                        ::std::sync::Mutex::new(::std::collections::HashMap::new())
                    ),
                    #(#function_initializers)*
                    #(#constant_initializers)*
                })
            }

            #async_module_items

            impl LoadedModule {
                pub fn as_cuda_module(&self) -> &::std::sync::Arc<::cuda_core::CudaModule> {
                    &self.__module
                }

                #(#launch_methods)*
                #(#constant_resolver_methods)*
                #(#set_constant_methods)*
                #async_launch_methods
            }
        }
    })
}

fn collect_cuda_module_kernels(items: &[Item]) -> syn::Result<Vec<CudaModuleKernel>> {
    let mut kernels = Vec::new();
    for item in items {
        let Item::Fn(item_fn) = item else {
            continue;
        };
        if !has_attr_named(&item_fn.attrs, "kernel") {
            continue;
        }
        let cluster_dim = cuda_module_cluster_dim(&item_fn.attrs)?;
        let cooperative = cuda_module_cooperative(&item_fn.attrs)?;
        let params = cuda_module_params(item_fn)?;
        let is_generic = item_fn
            .sig
            .generics
            .params
            .iter()
            .any(|param| matches!(param, GenericParam::Type(_)));
        kernels.push(CudaModuleKernel {
            vis: item_fn.vis.clone(),
            cfg_attrs: cuda_module_cfg_attrs(&item_fn.attrs),
            method_attrs: cuda_module_method_attrs(&item_fn.attrs),
            unsafety: item_fn.sig.unsafety,
            fn_name: item_fn.sig.ident.clone(),
            generics: item_fn.sig.generics.clone(),
            params,
            cluster_dim,
            cooperative,
            is_generic,
        });
    }
    Ok(kernels)
}

/// Generate the statements that pin this crate's embedded device artifact
/// into the final binary.
///
/// The codegen backend stores each crate's compiled device code (PTX,
/// cubin, NVVM IR, or LTOIR) in a `.oxart` data section of a small extra
/// object file. When the crate that holds the `#[cuda_module]` is a
/// *library*, that object becomes one member of the crate's `.rlib`
/// archive, and linkers only extract an archive member when it defines a
/// symbol that some already-linked object references. The backend defines
/// a global anchor symbol inside the artifact object for exactly this
/// purpose; here we emit the matching reference. Reading the anchor's
/// address through `black_box` inside `load_named()` means that any
/// program calling `load()` carries an undefined reference to the anchor,
/// which forces the linker to pull the artifact member out of the rlib.
/// Without this handshake the bundle was silently dropped and `load()`
/// failed at runtime with `ModuleNotFound` (issue #72).
///
/// Both sides derive the symbol name from `CARGO_PKG_NAME` and
/// `CARGO_PKG_VERSION`: this proc macro reads them while rustc compiles
/// the crate, and the codegen backend reads them inside the same rustc
/// process, so the names always agree under cargo.
///
/// The reference is only emitted when the module is guaranteed to produce
/// an artifact for this crate. Generic kernels are monomorphized (and
/// their PTX embedded) in the *consuming* crate, so a module with only
/// generic kernels yields no artifact here, and an anchor reference would
/// be an undefined-symbol link error. The same reasoning extends to
/// `#[cfg]`-gated kernels: the anchor is guarded by the disjunction of
/// the kernels' cfg conditions so it is only referenced when at least one
/// concrete kernel is actually compiled.
fn cuda_module_artifact_anchor_statements(
    kernels: &[CudaModuleKernel],
) -> syn::Result<TokenStream2> {
    let (Ok(package_name), Ok(package_version)) = (
        std::env::var("CARGO_PKG_NAME"),
        std::env::var("CARGO_PKG_VERSION"),
    ) else {
        // Not built by cargo (e.g. a raw rustc invocation): the backend
        // falls back to crate-name-based bundle naming and we cannot
        // reproduce it exactly, so skip the anchor rather than risk an
        // undefined symbol.
        return Ok(TokenStream2::new());
    };

    let non_generic: Vec<&CudaModuleKernel> =
        kernels.iter().filter(|kernel| !kernel.is_generic).collect();
    if non_generic.is_empty() {
        return Ok(TokenStream2::new());
    }

    let cfg_guard = if non_generic.iter().any(|kernel| kernel.cfg_attrs.is_empty()) {
        // At least one concrete kernel is compiled unconditionally, so the
        // artifact object always exists: no guard needed.
        None
    } else {
        // Every concrete kernel is cfg-gated. Reference the anchor only
        // when at least one of them is enabled. A kernel with several cfg
        // attributes requires all of them, hence all(...) per kernel
        // joined under any(...).
        let alternatives = non_generic
            .iter()
            .map(|kernel| {
                let predicates = kernel
                    .cfg_attrs
                    .iter()
                    .map(|attr| attr.parse_args::<TokenStream2>())
                    .collect::<syn::Result<Vec<_>>>()?;
                Ok(quote! { all( #(#predicates),* ) })
            })
            .collect::<syn::Result<Vec<_>>>()?;
        Some(quote! { #[cfg(any( #(#alternatives),* ))] })
    };

    let anchor = artifact_anchor_symbol(&package_name, &package_version);
    let anchor_name = LitStr::new(&anchor, proc_macro2::Span::call_site());
    Ok(quote! {
        // Keep-alive handshake with the codegen backend: see the macro
        // crate's `cuda_module_artifact_anchor_statements` for details.
        #cfg_guard
        let _artifact_anchor: *const ::core::primitive::u8 = {
            unsafe extern "C" {
                #[link_name = #anchor_name]
                static CUDA_OXIDE_BUNDLE_ANCHOR: ::core::primitive::u8;
            }
            ::std::hint::black_box(unsafe {
                ::core::ptr::addr_of!(CUDA_OXIDE_BUNDLE_ANCHOR)
            })
        };
    })
}

/// A `#[constant]` static collected from a `#[cuda_module]` body.
struct CudaModuleConstant {
    ident: Ident,
    ty: Box<Type>,
    cfg_attrs: Vec<syn::Attribute>,
    method_attrs: Vec<syn::Attribute>,
    symbol: String,
}

fn collect_cuda_module_constants(
    items: &[Item],
    module_ident: &Ident,
) -> syn::Result<Vec<CudaModuleConstant>> {
    let mut constants = Vec::new();
    for item in items {
        let Item::Static(item_static) = item else {
            continue;
        };
        if !has_attr_named(&item_static.attrs, "constant") {
            continue;
        }
        if extract_constant_inner_ty(&item_static.ty).is_none() {
            return Err(syn::Error::new_spanned(
                &item_static.ty,
                concat!(
                    "#[constant] static must have type `ConstantMemory<T>` ",
                    "(e.g. `static FOO: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;`). ",
                    "The wrapper prevents the compiler from constant-folding the initializer into kernel bodies.",
                ),
            ));
        }
        constants.push(CudaModuleConstant {
            ident: item_static.ident.clone(),
            ty: item_static.ty.clone(),
            cfg_attrs: cuda_module_cfg_attrs(&item_static.attrs),
            method_attrs: cuda_module_method_attrs(&item_static.attrs),
            symbol: cuda_module_constant_symbol(module_ident, &item_static.ident),
        });
    }
    Ok(constants)
}

fn cuda_module_constant_symbol(module_ident: &Ident, ident: &Ident) -> String {
    let start = ident.span().start();
    let base = format!(
        "{}_L{}C{}_{}",
        module_ident, start.line, start.column, ident
    );
    constant_symbol(&base)
}

fn cuda_module_items_with_constant_symbols(
    items: &[Item],
    constants: &[CudaModuleConstant],
) -> Vec<TokenStream2> {
    let mut constants = constants.iter();
    items
        .iter()
        .map(|item| {
            let Item::Static(item_static) = item else {
                return quote! { #item };
            };
            if !has_attr_named(&item_static.attrs, "constant") {
                return quote! { #item };
            }
            let constant = constants
                .next()
                .expect("constant collection and rewrite order drifted");

            let mut item_static = item_static.clone();
            let symbol = LitStr::new(&constant.symbol, constant.ident.span());
            item_static.attrs = item_static
                .attrs
                .into_iter()
                .map(|attr| {
                    if attr_path_ends_with(&attr, "constant") {
                        let path = attr.path().clone();
                        parse_quote!(#[#path(export_name = #symbol)])
                    } else {
                        attr
                    }
                })
                .collect();
            quote! { #item_static }
        })
        .collect()
}

fn cuda_module_constant_field_ident(ident: &Ident) -> Ident {
    format_ident!("__{}", ident)
}

fn cuda_module_constant_resolver_ident(ident: &Ident) -> Ident {
    format_ident!("__resolve_{}", ident)
}

/// Extract `T` from a `ConstantMemory<T>` type path. Returns `None` for anything
/// that's not a path ending in `ConstantMemory<...>` with one generic argument.
fn extract_constant_inner_ty(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let last = type_path.path.segments.last()?;
    if last.ident != "ConstantMemory" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    args.args.iter().find_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

/// Like [`extract_constant_inner_ty`] but for sites that have already been
/// gated by `#[constant]`'s type-path validation, so extraction failure is
/// a compiler-internal invariant violation, not a user error.
fn constant_inner_ty(ty: &Type) -> &Type {
    extract_constant_inner_ty(ty).unwrap_or_else(|| {
        panic!(
            "#[cuda_module] collected a #[constant] static whose type is not ConstantMemory<T>; \
             this should have been rejected by the #[constant] attribute"
        )
    })
}

fn generate_cuda_module_constant_field(constant: &CudaModuleConstant) -> TokenStream2 {
    let CudaModuleConstant {
        ident, cfg_attrs, ..
    } = constant;
    let field = cuda_module_constant_field_ident(ident);
    quote! {
        #(#cfg_attrs)*
        #field: ::std::sync::Arc<::std::sync::Mutex<::core::option::Option<::cuda_core::ConstantHandle>>>,
    }
}

fn generate_cuda_module_constant_initializer(constant: &CudaModuleConstant) -> TokenStream2 {
    let CudaModuleConstant {
        ident, cfg_attrs, ..
    } = constant;
    let field = cuda_module_constant_field_ident(ident);
    quote! {
        #(#cfg_attrs)*
        #field: ::std::sync::Arc::new(::std::sync::Mutex::new(::core::option::Option::None)),
    }
}

fn generate_cuda_module_constant_resolver_method(constant: &CudaModuleConstant) -> TokenStream2 {
    let CudaModuleConstant {
        ident,
        ty,
        cfg_attrs,
        symbol,
        ..
    } = constant;
    let field = cuda_module_constant_field_ident(ident);
    let resolver = cuda_module_constant_resolver_ident(ident);
    let symbol_lit = LitStr::new(symbol, ident.span());
    let inner_ty = constant_inner_ty(ty);
    let mismatch_msg = format!(
        "host/device size mismatch for #[constant] static `{ident}`: \
         device size {{}} bytes, host expected {{}} bytes (PTX symbol `{symbol}`)"
    );
    quote! {
        #(#cfg_attrs)*
        #[allow(non_snake_case)]
        fn #resolver(&self) -> ::core::result::Result<::cuda_core::ConstantHandle, ::cuda_core::DriverError> {
            let mut slot = self
                .#field
                .lock()
                .expect("cuda constant handle cache mutex poisoned");
            if let ::core::option::Option::Some(handle) = *slot {
                return ::core::result::Result::Ok(handle);
            }

            let (dptr, size) = self.__module.get_global(#symbol_lit)?;
            assert_eq!(
                size,
                ::core::mem::size_of::<#inner_ty>(),
                #mismatch_msg,
                size,
                ::core::mem::size_of::<#inner_ty>(),
            );
            // SAFETY: `dptr` was just resolved by `cuModuleGetGlobal` for a
            // module that the LoadedModule keeps alive, and the size matches
            // `size_of::<#inner_ty>()` (asserted above).
            let handle = unsafe { ::cuda_core::ConstantHandle::from_raw(dptr) };
            *slot = ::core::option::Option::Some(handle);
            ::core::result::Result::Ok(handle)
        }
    }
}

/// Generate stream-ordered `set_<name>` and one-shot `set_<name>_blocking`
/// methods on `LoadedModule`. The async setter stages owned host bytes so
/// temporaries remain valid until CUDA reaches the stream callback.
fn generate_cuda_module_set_constant_method(constant: &CudaModuleConstant) -> TokenStream2 {
    let CudaModuleConstant {
        ident,
        ty,
        cfg_attrs,
        method_attrs,
        ..
    } = constant;
    let method_suffix = ident.to_string().to_ascii_lowercase();
    let method_name = format_ident!("set_{}", method_suffix);
    let blocking_name = format_ident!("set_{}_blocking", method_suffix);
    let resolver = cuda_module_constant_resolver_ident(ident);
    let inner_ty = constant_inner_ty(ty);

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #[allow(non_snake_case)]
        pub fn #method_name(
            &self,
            stream: &::cuda_core::CudaStream,
            value: &#inner_ty,
        ) -> ::core::result::Result<(), ::cuda_core::DriverError> {
            let handle = self.#resolver()?;
            let num_bytes = ::core::mem::size_of::<#inner_ty>();
            let mut bytes = ::std::boxed::Box::<[u8]>::new_uninit_slice(num_bytes);
            unsafe {
                ::core::ptr::copy_nonoverlapping(
                    value as *const #inner_ty as *const u8,
                    bytes.as_mut_ptr() as *mut u8,
                    num_bytes,
                );
            }
            handle.write_async_staged(stream, bytes)
        }

        #(#cfg_attrs)*
        #(#method_attrs)*
        #[allow(non_snake_case)]
        pub fn #blocking_name(
            &self,
            value: &#inner_ty,
        ) -> ::core::result::Result<(), ::cuda_core::DriverError> {
            let handle = self.#resolver()?;
            // SAFETY: handle was size-checked against `#inner_ty` by the lazy
            // resolver; `value` is a valid host pointer for
            // `size_of::<#inner_ty>()`.
            unsafe {
                handle.write_blocking(
                    &self.__module,
                    value as *const #inner_ty as *const u8,
                    ::core::mem::size_of::<#inner_ty>(),
                )
            }
        }
    }
}

fn cuda_module_method_attrs(attrs: &[syn::Attribute]) -> Vec<syn::Attribute> {
    attrs
        .iter()
        .filter(|attr| attr_path_ends_with(attr, "doc"))
        .cloned()
        .collect()
}

fn cuda_module_cfg_attrs(attrs: &[syn::Attribute]) -> Vec<syn::Attribute> {
    attrs
        .iter()
        .filter(|attr| attr_path_ends_with(attr, "cfg"))
        .cloned()
        .collect()
}

fn has_attr_named(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| attr_path_ends_with(attr, name))
}

fn attr_path_ends_with(attr: &syn::Attribute, name: &str) -> bool {
    attr.path()
        .segments
        .last()
        .map(|segment| segment.ident == name)
        .unwrap_or(false)
}

fn cuda_module_cluster_dim(attrs: &[syn::Attribute]) -> syn::Result<Option<(u32, u32, u32)>> {
    for attr in attrs {
        if attr_path_ends_with(attr, "cluster_launch") {
            let args = attr.parse_args::<ClusterArgs>()?;
            return Ok(Some((args.x, args.y, args.z)));
        }
    }
    Ok(None)
}

fn cuda_module_cooperative(attrs: &[syn::Attribute]) -> syn::Result<bool> {
    for attr in attrs {
        if attr_path_ends_with(attr, "cooperative_launch") {
            if !matches!(attr.meta, syn::Meta::Path(_)) {
                return Err(syn::Error::new_spanned(
                    attr,
                    "cooperative_launch takes no arguments: use a bare #[cooperative_launch]",
                ));
            }
            return Ok(true);
        }
    }
    Ok(false)
}

fn cuda_module_params(item_fn: &ItemFn) -> syn::Result<Vec<CudaModuleParam>> {
    item_fn
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(receiver) => Err(syn::Error::new_spanned(
                receiver,
                "cuda_module kernels cannot take self parameters",
            )),
            FnArg::Typed(pat_type) => cuda_module_param_from_typed(pat_type),
        })
        .collect()
}

fn cuda_module_param_from_typed(pat_type: &syn::PatType) -> syn::Result<CudaModuleParam> {
    let Pat::Ident(pat_ident) = &*pat_type.pat else {
        return Err(syn::Error::new_spanned(
            &pat_type.pat,
            "cuda_module only supports simple identifier kernel parameters",
        ));
    };
    let name = pat_ident.ident.clone();
    let (sync_host_ty, async_host_ty, marshal) = cuda_module_host_type(&pat_type.ty)?;
    Ok(CudaModuleParam {
        name,
        sync_host_ty,
        async_host_ty,
        marshal,
    })
}

fn cuda_module_host_type(
    ty: &Type,
) -> syn::Result<(TokenStream2, TokenStream2, CudaModuleParamMarshal)> {
    if let Some((elem_ty, mutable)) = cuda_module_slice_elem(ty) {
        let sync_host_ty = if mutable {
            quote! { &mut ::cuda_core::DeviceBuffer<#elem_ty> }
        } else {
            quote! { &::cuda_core::DeviceBuffer<#elem_ty> }
        };
        let (async_host_ty, marshal) = if mutable {
            (
                quote! { &'__cuda_module_async mut impl ::cuda_host::KernelSliceArgMut<Elem = #elem_ty> },
                CudaModuleParamMarshal::WritableDeviceBuffer {
                    elem_ty: quote! { #elem_ty },
                },
            )
        } else {
            (
                quote! { &'__cuda_module_async impl ::cuda_host::KernelSliceArg<Elem = #elem_ty> },
                CudaModuleParamMarshal::ReadOnlyDeviceBuffer {
                    elem_ty: quote! { #elem_ty },
                },
            )
        };
        return Ok((sync_host_ty, async_host_ty, marshal));
    }

    if let Some(elem_ty) = cuda_module_disjoint_slice_elem(ty) {
        return Ok((
            quote! { &mut ::cuda_core::DeviceBuffer<#elem_ty> },
            quote! { &'__cuda_module_async mut impl ::cuda_host::KernelSliceArgMut<Elem = #elem_ty> },
            CudaModuleParamMarshal::WritableDeviceBuffer {
                elem_ty: quote! { #elem_ty },
            },
        ));
    }

    if matches!(ty, Type::Reference(_)) {
        return Err(syn::Error::new_spanned(
            ty,
            "cuda_module only supports slice references; use &[T], &mut [T], DisjointSlice<T>, a raw pointer, or a by-value KernelScalar",
        ));
    }

    Ok((
        quote! { #ty },
        quote! { #ty },
        CudaModuleParamMarshal::Scalar,
    ))
}

fn cuda_module_slice_elem(ty: &Type) -> Option<(TokenStream2, bool)> {
    let Type::Reference(type_ref) = ty else {
        return None;
    };
    let Type::Slice(slice) = &*type_ref.elem else {
        return None;
    };
    let elem = &slice.elem;
    Some((quote! { #elem }, type_ref.mutability.is_some()))
}

fn cuda_module_disjoint_slice_elem(ty: &Type) -> Option<TokenStream2> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "DisjointSlice" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| {
        if let GenericArgument::Type(ty) = arg {
            Some(quote! { #ty })
        } else {
            None
        }
    })
}

fn generate_cuda_module_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let unsafety = &kernel.unsafety;
    let fn_name = &kernel.fn_name;
    let generics = cuda_module_launch_generics(kernel);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params = kernel.params.iter().map(|param| {
        let name = &param.name;
        let host_ty = &param.sync_host_ty;
        quote! { #name: #host_ty }
    });
    let arg_marshalling = kernel
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| cuda_module_arg_marshalling(index, param));
    let function_binding = cuda_module_function_binding(kernel);
    let launch_call = cuda_module_launch_call(kernel);

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis #unsafety fn #fn_name #impl_generics (
            &self,
            stream: &::cuda_core::CudaStream,
            config: ::cuda_core::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<(), ::cuda_core::DriverError>
        #where_clause
        {
            #function_binding
            let mut __args: ::std::vec::Vec<*mut ::std::ffi::c_void> = ::std::vec::Vec::new();
            #(#arg_marshalling)*
            #launch_call
        }
    }
}

fn generate_cuda_module_async_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let unsafety = &kernel.unsafety;
    let fn_name = format_ident!("{}_async", kernel.fn_name);
    let generics = cuda_module_async_launch_generics(kernel);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params = kernel.params.iter().map(|param| {
        let name = &param.name;
        let host_ty = &param.async_host_ty;
        quote! { #name: #host_ty }
    });
    let arg_marshalling = kernel.params.iter().map(cuda_module_async_arg_marshalling);
    let function_binding = cuda_module_function_binding(kernel);
    let cluster_dim = kernel.cluster_dim.map(|(x, y, z)| quote! { (#x, #y, #z) });
    let set_cluster_dim = cluster_dim.map(|cluster_dim| {
        quote! {
            ::cuda_host::set_async_kernel_cluster_dim(&mut __launch, #cluster_dim);
        }
    });
    let set_cooperative = kernel.cooperative.then(|| {
        quote! {
            ::cuda_host::set_async_kernel_cooperative(&mut __launch, true);
        }
    });

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis #unsafety fn #fn_name #impl_generics (
            &self,
            config: ::cuda_core::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<::cuda_host::AsyncKernelLaunch<'__cuda_module_async>, ::cuda_core::DriverError>
        #where_clause
        {
            #function_binding
            let mut __launch = ::cuda_host::new_async_kernel_launch(__func.clone(), config);
            #set_cluster_dim
            #set_cooperative
            #(#arg_marshalling)*
            Ok(__launch)
        }
    }
}

fn generate_cuda_module_owned_async_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let unsafety = &kernel.unsafety;
    let fn_name = format_ident!("{}_async_owned", kernel.fn_name);
    let resources = cuda_module_owned_resource_params(kernel);
    let generics = cuda_module_owned_async_launch_generics(kernel, &resources);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params = kernel.params.iter().enumerate().map(|(index, param)| {
        let name = &param.name;
        match &param.marshal {
            CudaModuleParamMarshal::Scalar => {
                let host_ty = &param.async_host_ty;
                quote! { #name: #host_ty }
            }
            CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
                let resource_ty = cuda_module_owned_resource_type(index);
                quote! { #name: #resource_ty }
            }
            CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
                let resource_ty = cuda_module_owned_resource_type(index);
                quote! { mut #name: #resource_ty }
            }
        }
    });
    let arg_marshalling = kernel
        .params
        .iter()
        .map(cuda_module_owned_async_arg_marshalling);
    let function_binding = cuda_module_function_binding(kernel);
    let cluster_dim = kernel.cluster_dim.map(|(x, y, z)| quote! { (#x, #y, #z) });
    let set_cluster_dim = cluster_dim.map(|cluster_dim| {
        quote! {
            ::cuda_host::set_async_kernel_cluster_dim(&mut __launch, #cluster_dim);
        }
    });
    let set_cooperative = kernel.cooperative.then(|| {
        quote! {
            ::cuda_host::set_async_kernel_cooperative(&mut __launch, true);
        }
    });
    let resources_ty = cuda_module_owned_resources_ty(&resources);
    let resource_names = resources.iter().map(|(_, name, _, _)| name);
    let resources_expr = if resources.is_empty() {
        quote! { () }
    } else {
        quote! { (#(#resource_names),*) }
    };

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis #unsafety fn #fn_name #impl_generics (
            &self,
            config: ::cuda_core::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<::cuda_host::OwnedAsyncKernelLaunch<#resources_ty>, ::cuda_core::DriverError>
        #where_clause
        {
            #function_binding
            let mut __launch: ::cuda_host::AsyncKernelLaunch<'static> =
                ::cuda_host::new_async_kernel_launch(__func.clone(), config);
            #set_cluster_dim
            #set_cooperative
            #(#arg_marshalling)*
            Ok(::cuda_host::new_owned_async_kernel_launch(__launch, #resources_expr))
        }
    }
}

fn cuda_module_launch_generics(kernel: &CudaModuleKernel) -> syn::Generics {
    let mut generics = kernel.generics.clone();
    for param in &kernel.params {
        if matches!(param.marshal, CudaModuleParamMarshal::Scalar) {
            let host_ty = &param.sync_host_ty;
            generics
                .make_where_clause()
                .predicates
                .push(syn::parse_quote! { #host_ty: ::cuda_host::KernelScalar });
        }
    }
    generics
}

fn cuda_module_owned_async_launch_generics(
    kernel: &CudaModuleKernel,
    resources: &[(usize, Ident, TokenStream2, bool)],
) -> syn::Generics {
    let mut generics = kernel.generics.clone();
    for (index, _, elem_ty, writable) in resources {
        let resource_ty = cuda_module_owned_resource_type(*index);
        generics.params.push(syn::parse_quote! { #resource_ty });
        let predicate: syn::WherePredicate = if *writable {
            syn::parse_quote! {
                #resource_ty: ::cuda_host::KernelSliceArgMut<Elem = #elem_ty> + Send + 'static
            }
        } else {
            syn::parse_quote! {
                #resource_ty: ::cuda_host::KernelSliceArg<Elem = #elem_ty> + Send + 'static
            }
        };
        generics.make_where_clause().predicates.push(predicate);
    }
    for param in &kernel.params {
        if matches!(param.marshal, CudaModuleParamMarshal::Scalar) {
            let host_ty = &param.async_host_ty;
            generics
                .make_where_clause()
                .predicates
                .push(syn::parse_quote! { #host_ty: ::cuda_host::KernelScalar + 'static });
        }
    }
    generics
}

fn cuda_module_async_launch_generics(kernel: &CudaModuleKernel) -> syn::Generics {
    let mut generics = kernel.generics.clone();
    generics
        .params
        .insert(0, syn::parse_quote! { '__cuda_module_async });
    for param in &kernel.params {
        if matches!(param.marshal, CudaModuleParamMarshal::Scalar) {
            let host_ty = &param.async_host_ty;
            generics.make_where_clause().predicates.push(
                syn::parse_quote! { #host_ty: ::cuda_host::KernelScalar + '__cuda_module_async },
            );
        }
    }
    generics
}

fn cuda_module_owned_resource_params(
    kernel: &CudaModuleKernel,
) -> Vec<(usize, Ident, TokenStream2, bool)> {
    kernel
        .params
        .iter()
        .enumerate()
        .filter_map(|(index, param)| match &param.marshal {
            CudaModuleParamMarshal::Scalar => None,
            CudaModuleParamMarshal::ReadOnlyDeviceBuffer { elem_ty } => {
                Some((index, param.name.clone(), elem_ty.clone(), false))
            }
            CudaModuleParamMarshal::WritableDeviceBuffer { elem_ty } => {
                Some((index, param.name.clone(), elem_ty.clone(), true))
            }
        })
        .collect()
}

fn cuda_module_owned_resource_type(index: usize) -> Ident {
    format_ident!("__CudaModuleArg{index}")
}

fn cuda_module_owned_resources_ty(
    resources: &[(usize, Ident, TokenStream2, bool)],
) -> TokenStream2 {
    if resources.is_empty() {
        quote! { () }
    } else {
        let resource_tys = resources
            .iter()
            .map(|(index, _, _, _)| cuda_module_owned_resource_type(*index));
        quote! { (#(#resource_tys),*) }
    }
}

fn cuda_module_arg_marshalling(index: usize, param: &CudaModuleParam) -> TokenStream2 {
    let name = &param.name;
    let value_name = format_ident!("__arg_{index}");
    match param.marshal {
        CudaModuleParamMarshal::Scalar => {
            quote! {
                let mut #value_name = #name;
                ::cuda_host::push_kernel_scalar(&mut __args, &mut #value_name);
            }
        }
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
            let ptr_name = format_ident!("__arg_{index}_ptr");
            let len_name = format_ident!("__arg_{index}_len");
            quote! {
                let (mut #ptr_name, mut #len_name) =
                    ::cuda_host::read_only_device_buffer_arg(#name);
                ::cuda_host::push_kernel_device_slice(
                    &mut __args,
                    &mut #ptr_name,
                    &mut #len_name,
                );
            }
        }
        CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
            let ptr_name = format_ident!("__arg_{index}_ptr");
            let len_name = format_ident!("__arg_{index}_len");
            quote! {
                let (mut #ptr_name, mut #len_name) =
                    ::cuda_host::writable_device_buffer_arg(#name);
                ::cuda_host::push_kernel_device_slice(
                    &mut __args,
                    &mut #ptr_name,
                    &mut #len_name,
                );
            }
        }
    }
}

fn cuda_module_owned_async_arg_marshalling(param: &CudaModuleParam) -> TokenStream2 {
    let name = &param.name;
    match param.marshal {
        CudaModuleParamMarshal::Scalar => {
            quote! {
                ::cuda_host::push_async_kernel_scalar(&mut __launch, #name);
            }
        }
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_read_only_device_slice(&mut __launch, &#name);
            }
        }
        CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_writable_device_slice(&mut __launch, &mut #name);
            }
        }
    }
}

fn cuda_module_async_arg_marshalling(param: &CudaModuleParam) -> TokenStream2 {
    let name = &param.name;
    match param.marshal {
        CudaModuleParamMarshal::Scalar => {
            quote! {
                ::cuda_host::push_async_kernel_scalar(&mut __launch, #name);
            }
        }
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_read_only_device_slice(&mut __launch, #name);
            }
        }
        CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_writable_device_slice(&mut __launch, #name);
            }
        }
    }
}

fn cuda_module_function_binding(kernel: &CudaModuleKernel) -> TokenStream2 {
    if kernel.is_generic {
        let fn_name = &kernel.fn_name;
        let marker = cuda_kernel_marker_name(fn_name);
        let type_params = cuda_module_type_param_names(&kernel.generics);
        let kernel_entry = format_ident!("{}", kernel_symbol(&fn_name.to_string()));
        let turbofish = if type_params.is_empty() {
            quote! {}
        } else {
            quote! { ::<#(#type_params),*> }
        };
        let marker_args = if type_params.is_empty() {
            quote! {}
        } else {
            quote! { <#(#type_params),*> }
        };
        quote! {
            let __kernel_ptr = #kernel_entry #turbofish as *const ();
            unsafe {
                let mut __force_mono: *const () = ::core::ptr::null();
                ::core::ptr::write_volatile(&mut __force_mono, __kernel_ptr);
                let _ = ::core::ptr::read_volatile(&__force_mono);
            }
            let __ptx_name =
                <#marker #marker_args as ::cuda_host::GenericCudaKernel>::ptx_name();
            let __func_storage = {
                let mut __cache = self
                    .__generic_functions
                    .lock()
                    .expect("cuda_module generic function cache poisoned");
                if let Some(__func) = __cache.get(__ptx_name) {
                    __func.clone()
                } else {
                    let __func = self.__module.load_function(__ptx_name)?;
                    __cache.insert(__ptx_name, __func.clone());
                    __func
                }
            };
            let __func = &__func_storage;
        }
    } else {
        let field = cuda_module_function_field(&kernel.fn_name);
        quote! {
            let __func = &self.#field;
        }
    }
}

fn cuda_module_launch_call(kernel: &CudaModuleKernel) -> TokenStream2 {
    let cluster_dim = kernel.cluster_dim.map(|(x, y, z)| quote! { (#x, #y, #z) });
    match (cluster_dim, kernel.cooperative) {
        (Some(cluster_dim), true) => quote! {
            unsafe {
                ::cuda_core::launch_kernel_ex_cooperative_on_stream(
                    __func,
                    config.grid_dim,
                    config.block_dim,
                    config.shared_mem_bytes,
                    #cluster_dim,
                    stream,
                    &mut __args,
                )
            }
        },
        (Some(cluster_dim), false) => quote! {
            unsafe {
                ::cuda_core::launch_kernel_ex_on_stream(
                    __func,
                    config.grid_dim,
                    config.block_dim,
                    config.shared_mem_bytes,
                    #cluster_dim,
                    stream,
                    &mut __args,
                )
            }
        },
        (None, true) => quote! {
            unsafe {
                ::cuda_core::launch_kernel_cooperative_on_stream(
                    __func,
                    config.grid_dim,
                    config.block_dim,
                    config.shared_mem_bytes,
                    stream,
                    &mut __args,
                )
            }
        },
        (None, false) => quote! {
            unsafe {
                ::cuda_core::launch_kernel_on_stream(
                    __func,
                    config.grid_dim,
                    config.block_dim,
                    config.shared_mem_bytes,
                    stream,
                    &mut __args,
                )
            }
        },
    }
}

fn cuda_module_type_param_names(generics: &syn::Generics) -> Vec<Ident> {
    generics
        .params
        .iter()
        .filter_map(|param| {
            if let GenericParam::Type(type_param) = param {
                Some(type_param.ident.clone())
            } else {
                None
            }
        })
        .collect()
}

fn cuda_module_function_field(fn_name: &Ident) -> Ident {
    format_ident!("__{}_function", fn_name)
}

fn cuda_kernel_marker_name(fn_name: &Ident) -> Ident {
    format_ident!("__{}_CudaKernel", fn_name)
}

/// Marks a function as a CUDA kernel.
///
/// This attribute:
/// 1. Adds `#[no_mangle]` to preserve the function name in the binary
/// 2. Marks the function for detection by the `rustc-codegen-cuda` backend
///
/// # Generic Kernels
///
/// For generic kernels (like `template<class F> __global__` in CUDA C++),
/// specify the types to instantiate:
///
/// ```ignore
/// #[kernel(Scale, Fma, Square)]
/// pub fn map<F: GpuFn>(f: F, input: &[i32], output: DisjointSlice<i32>) {
///     // ...
/// }
/// ```
///
/// This generates three PTX entry points: `map_Scale`, `map_Fma`, `map_Square`.
/// Each is a monomorphized version of the generic kernel.
///
/// # Example (non-generic)
///
/// ```ignore
/// #[kernel]
/// pub fn simple_kernel(data: &mut [i32]) {
///     // ...
/// }
/// ```
///
/// # Loop unrolling
///
/// Put `#[unroll]` on a loop with a compile-time-known trip count to request full
/// unrolling. Use `#[unroll(N)]`, where `N >= 2`, to request `N` iterations of
/// work per trip; a remainder loop handles any leftovers.
///
/// ```ignore
/// #[kernel]
/// pub fn example(n: u32) {
///     let mut i = 0;
///     #[unroll]
///     while i < 4 { work(i); i += 1; }
///
///     let mut j = 0;
///     #[unroll(4)]
///     while j < n { work(j); j += 1; }
/// }
/// ```
///
/// The pass currently recognizes explicit counted `while` loops. Range-based
/// `for` loops are not yet recognized.
///
/// Only the annotated loop is unrolled. Inner loops are copied intact unless
/// they carry their own annotation. Several `continue` paths (multiple
/// back-edges) are supported.
///
/// Full `#[unroll]` preserves `break` paths and multiple exit targets. Partial
/// `#[unroll(N)]` currently requires the loop condition to be the only exit. If
/// the loop has a `break` or another extra exit, the compiler warns and skips
/// partial unrolling for that loop.
///
/// Partial unrolling also requires a positive counter step, a `<` or `<=` test,
/// and a limit that does not change inside the loop. One annotation may create
/// at most 1,024 body copies, 8,192 cloned basic blocks, and 65,536 cloned
/// operations. Unsupported or larger requests warn and are not unrolled.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as KernelArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    if let Some(err) = reject_reserved_name(&input.sig.ident) {
        return err;
    }

    // Consume any `#[unroll]` / `#[unroll(N)]` attributes written directly on
    // loops inside the kernel body. We strip the attribute from the loop
    // expression (so rustc never sees an expression attribute, keeping us off
    // nightly `stmt_expr_attributes`) and inject an
    // `__unroll_config::<FACTOR>()` marker into the loop body. If the visitor
    // hits a malformed attribute it records the error so we can surface it as a
    // compile error below.
    if let Err(err) = rewrite_loop_unroll_attrs(&mut input) {
        return err.to_compile_error().into();
    }

    // Check if function has type parameters
    let has_generics = input
        .sig
        .generics
        .params
        .iter()
        .any(|p| matches!(p, GenericParam::Type(_)));

    if has_generics && args.instantiate_types.is_empty() {
        // Generic kernel without explicit types - allow it!
        // Instantiation will happen from call sites (nvcc-style)
        return generate_generic_kernel_no_instantiation(input);
    }

    if !has_generics && !args.instantiate_types.is_empty() {
        // Non-generic kernel with instantiation types - error
        return syn::Error::new_spanned(
            &input.sig.ident,
            "Instantiation types only apply to generic kernels",
        )
        .to_compile_error()
        .into();
    }

    if has_generics {
        // Generate wrapper kernels for each instantiation type
        generate_generic_kernel(input, args.instantiate_types)
    } else {
        // Simple non-generic kernel
        generate_simple_kernel(input)
    }
}

struct ConstantArgs {
    export_name: Option<LitStr>,
}

impl Parse for ConstantArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self { export_name: None });
        }

        let key: Ident = input.parse()?;
        if key != "export_name" {
            return Err(syn::Error::new_spanned(
                key,
                "#[constant] does not take public arguments",
            ));
        }
        input.parse::<Token![=]>()?;
        let export_name = input.parse()?;
        if !input.is_empty() {
            input.parse::<Token![,]>()?;
            if !input.is_empty() {
                return Err(input.error("unexpected tokens after #[constant] export_name"));
            }
        }
        Ok(Self {
            export_name: Some(export_name),
        })
    }
}

/// Mark a module-scope `static` as a CUDA constant-memory global.
///
/// The static must be typed `ConstantMemory<T>` (see
/// [`cuda_device::ConstantMemory`](../../cuda_device/struct.ConstantMemory.html)). The
/// `ConstantMemory<T>` wrapper has `UnsafeCell<T>` semantics on the device,
/// preventing the compiler from constant-folding the initializer and making
/// the host's `cuMemcpyHtoD` updates observable from kernels.
///
/// The macro adds a reserved `#[unsafe(export_name = "cuda_oxide_const_246e25db_...")]`
/// so the PTX symbol carries a name the host can resolve via
/// `cuModuleGetGlobal`. When used inside `#[cuda_module]`, the generated name
/// includes module/source-location context to avoid collisions between constants
/// that share the same Rust identifier. The host-side `#[cuda_module]` expansion
/// separately generates per-constant setter methods on the loaded module:
///
/// - `module.set_<name>(&stream, &value)` — stream-ordered async write
///   (recommended; orders correctly against surrounding kernel launches).
/// - `module.set_<name>_blocking(&value)` — synchronous `cuMemcpyHtoD`
///   for one-shot initialization where no stream is in scope.
///
/// # Restrictions
///
/// - The static must be typed `ConstantMemory<T>`.
/// - The initializer must be `ConstantMemory::UNINIT` (or any other all-zeros
///   value). Honoring arbitrary non-zero initializers is not yet
///   implemented; populate from the host before any kernel reads the value.
/// - The attribute must appear inside a `#[cuda_module]`. Placed elsewhere
///   it silently produces an unreachable symbol (no setter is generated).
/// - The identifier must not start with the reserved cuda-oxide prefix.
///
/// # Example
///
/// ```ignore
/// #[cuda_module]
/// mod kernels {
///     #[constant]
///     static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;
///
///     #[kernel]
///     pub fn compute(mut out: DisjointSlice<f32>) {
///         let c = COEFFS.get();
///         let i = thread::index_1d().get();
///         if let Some(e) = out.get_mut(thread::index_1d()) {
///             *e = c[0] * (i as f32) + c[1];
///         }
///     }
/// }
///
/// // Host
/// module.set_coeffs(&stream, &[1.0, 2.0, 3.0, 4.0])?;
/// ```
#[proc_macro_attribute]
pub fn constant(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ConstantArgs);
    let input = parse_macro_input!(item as syn::ItemStatic);

    if let Some(err) = reject_reserved_name(&input.ident) {
        return err;
    }

    if extract_constant_inner_ty(&input.ty).is_none() {
        return syn::Error::new_spanned(
            &input.ty,
            "#[constant] static must have type `ConstantMemory<T>` \
             (e.g. `static FOO: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;`). \
             The wrapper prevents the compiler from constant-folding the \
             initializer into kernel bodies.",
        )
        .to_compile_error()
        .into();
    }

    let export_name = args.export_name.unwrap_or_else(|| {
        LitStr::new(
            &constant_symbol(&input.ident.to_string()),
            input.ident.span(),
        )
    });

    quote! {
        #[unsafe(export_name = #export_name)]
        #input
    }
    .into()
}

/// Find the generic type parameter that has a Fn/FnMut/FnOnce bound (the closure type).
/// Returns the type parameter name if found.
fn find_closure_generic(generics: &syn::Generics) -> Option<syn::Ident> {
    for param in &generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            for bound in &type_param.bounds {
                if is_fn_trait_bound(bound) {
                    return Some(type_param.ident.clone());
                }
            }
        }
    }

    if let Some(where_clause) = &generics.where_clause {
        for predicate in &where_clause.predicates {
            let syn::WherePredicate::Type(predicate_type) = predicate else {
                continue;
            };
            if !predicate_type.bounds.iter().any(is_fn_trait_bound) {
                continue;
            }
            let Type::Path(type_path) = &predicate_type.bounded_ty else {
                continue;
            };
            if type_path.qself.is_none()
                && type_path.path.segments.len() == 1
                && let Some(segment) = type_path.path.segments.first()
            {
                return Some(segment.ident.clone());
            }
        }
    }

    None
}

fn is_fn_trait_bound(bound: &syn::TypeParamBound) -> bool {
    let syn::TypeParamBound::Trait(trait_bound) = bound else {
        return false;
    };
    trait_bound.path.segments.last().is_some_and(|segment| {
        matches!(
            segment.ident.to_string().as_str(),
            "Fn" | "FnMut" | "FnOnce"
        )
    })
}

/// Find which function parameter uses the closure type.
/// Returns the index and info of the closure parameter.
fn find_closure_param<'a>(
    args_info: &'a [(&'a Ident, &'a Type)],
    closure_type_name: &syn::Ident,
) -> Option<(usize, &'a (&'a Ident, &'a Type))> {
    for (idx, (_name, ty)) in args_info.iter().enumerate() {
        // Check if the type is a simple path matching our closure generic
        if let Type::Path(type_path) = *ty
            && type_path.qself.is_none()
            && let Some(segment) = type_path.path.segments.first()
            && type_path.path.segments.len() == 1
            && segment.ident == *closure_type_name
        {
            return Some((idx, &args_info[idx]));
        }
    }
    None
}

/// Strip `mut` from function argument patterns.
///
/// The wrapper function just forwards arguments, so it doesn't need `mut`.
/// Keeping `mut` causes "variable does not need to be mutable" warnings.
fn strip_mut_from_inputs(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
) -> Vec<FnArg> {
    inputs
        .iter()
        .map(|arg| {
            match arg {
                FnArg::Typed(pat_type) => {
                    let mut new_pat_type = pat_type.clone();
                    if let Pat::Ident(pat_ident) = &*pat_type.pat
                        && pat_ident.mutability.is_some()
                    {
                        // Create new PatIdent without mut
                        let new_pat_ident = syn::PatIdent {
                            attrs: pat_ident.attrs.clone(),
                            by_ref: pat_ident.by_ref,
                            mutability: None, // Strip mut
                            ident: pat_ident.ident.clone(),
                            subpat: pat_ident.subpat.clone(),
                        };
                        new_pat_type.pat = Box::new(Pat::Ident(new_pat_ident));
                    }
                    FnArg::Typed(new_pat_type)
                }
                other => other.clone(),
            }
        })
        .collect()
}

/// True when `path`'s *last* segment is `name`.
///
/// We deliberately match on the tail only, so all of these resolve to the
/// same intrinsic:
///
/// ```ignore
/// index_1d()
/// thread::index_1d()
/// cuda_device::thread::index_1d()
/// ::cuda_device::thread::index_1d()
/// ```
///
/// And imports/aliases work too:
///
/// ```ignore
/// use cuda_device::thread::index_1d;          // bare ident → matches
/// use cuda_device::thread::index_1d as foo;   // aliased    → won't match (path tail is `foo`)
/// ```
///
/// The aliased form is intentionally not rewritten — if the user picked a
/// new name, they get the bare-stub-panic behaviour, not silent capture.
///
/// Caveat: if the user defines a *local* `fn index_1d` (or any other
/// reserved name) and calls it from inside `#[kernel]` / `#[device]`,
/// that call gets rewritten too. See the `Reserved names` section in
/// `ThreadIndex`'s doc-block — the convention is to pick a different
/// name (e.g. `compute_index_1d`) for any helper you want to keep.
fn is_thread_index_path(path: &Path, name: &str) -> bool {
    path.segments.last().is_some_and(|seg| seg.ident == name)
}

/// Build the rewritten path that points the user's call at the
/// `__internal::<name>` shim, preserving whatever prefix the user wrote.
///
/// The motivation is unused-import hygiene. If the user wrote
/// `use cuda_device::thread;` and called `thread::index_1d()`, replacing
/// the whole call with an absolute path makes rustc see the `thread`
/// import as unused. Instead, we splice `__internal` in front of the
/// last segment and keep everything before it intact:
///
/// ```ignore
/// thread::index_1d()                   →  thread::__internal::index_1d(&scope)
/// cuda_device::thread::index_1d()      →  cuda_device::thread::__internal::index_1d(&scope)
/// ::cuda_device::thread::index_1d()    →  ::cuda_device::thread::__internal::index_1d(&scope)
/// ```
///
/// Bare-ident calls are the only shape that can't carry a prefix, so for
/// those we fall back to the absolute path (the user wasn't naming
/// anything to import; see the bare-ident case in `is_thread_index_path`'s
/// doc-comment for why we still rewrite those):
///
/// ```ignore
/// index_1d()                           →  ::cuda_device::thread::__internal::index_1d(&scope)
/// ```
fn internal_thread_path(user_path: &Path, name: &str, arguments: syn::PathArguments) -> Path {
    let ident = format_ident!("{}", name);

    if user_path.segments.len() == 1 {
        let mut absolute: Path = parse_quote! { ::cuda_device::thread::__internal::#ident };
        if let Some(last) = absolute.segments.last_mut() {
            last.arguments = arguments;
        }
        return absolute;
    }

    let leading_colon = user_path.leading_colon;
    let prefix_segments: Vec<&syn::PathSegment> = user_path
        .segments
        .iter()
        .take(user_path.segments.len() - 1)
        .collect();
    let mut rewritten: Path =
        parse_quote! { #leading_colon #(#prefix_segments)::* :: __internal :: #ident };
    if let Some(last) = rewritten.segments.last_mut() {
        last.arguments = arguments;
    }
    rewritten
}

/// One scoped intrinsic the rewriter knows about.
///
/// Adding a new `thread::*` function that needs the `'kernel` scope is a
/// one-line entry here, plus the matching public stub and `__internal::*`
/// impl in `cuda-device`.
struct ScopedIntrinsic {
    /// The unqualified function name (last segment of the path we match).
    name: &'static str,
    /// If true, copy the call-site's turbofish onto the rewritten path
    /// (e.g. `index_2d::<S>` → `__internal::index_2d::<S>`).
    preserve_turbofish: bool,
    /// If true, forward the original call arguments after the scope ref
    /// (e.g. `index_2d_runtime(s)` → `__internal::index_2d_runtime(&scope, s)`).
    forward_args: bool,
}

const SCOPED_INTRINSICS: &[ScopedIntrinsic] = &[
    ScopedIntrinsic {
        name: "index_1d",
        preserve_turbofish: false,
        forward_args: false,
    },
    ScopedIntrinsic {
        name: "index_2d",
        preserve_turbofish: true,
        forward_args: false,
    },
    ScopedIntrinsic {
        name: "index_2d_runtime",
        preserve_turbofish: false,
        forward_args: true,
    },
];

/// Method names whose zero-arg call sites get the kernel scope spliced in
/// as a leading `&scope` argument.
///
/// These are matched on the *method name only* (not the receiver type, which
/// the macro can't see anyway). The scope is only injected when the user
/// wrote a zero-arg call like `slice.get_mut_indexed()`; if they passed
/// arguments themselves, we leave the call alone and let typeck decide.
///
/// Same caveat as `SCOPED_INTRINSICS`: a local method on an unrelated type
/// with the same name and a zero-arg form will get the scope appended,
/// which will cause a typeck error ("expected 0 arguments, got 1"). Pick
/// a different name (e.g. `pop_indexed`) for any helper you want to keep.
const SCOPED_METHODS: &[&str] = &["get_mut_indexed"];

fn is_scoped_method(method: &Ident) -> bool {
    SCOPED_METHODS.iter().any(|name| method == name)
}

struct ThreadIndexCallRewriter {
    scope_ident: Ident,
    rewrote_index_call: bool,
}

impl VisitMut for ThreadIndexCallRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        visit_mut::visit_expr_mut(self, expr);

        match expr {
            Expr::Call(ExprCall { func, args, .. }) => {
                let Expr::Path(ExprPath { path, .. }) = &**func else {
                    return;
                };
                let Some(intrinsic) = SCOPED_INTRINSICS
                    .iter()
                    .find(|i| is_thread_index_path(path, i.name))
                else {
                    return;
                };

                let last_args = path
                    .segments
                    .last()
                    .map(|seg| seg.arguments.clone())
                    .unwrap_or(syn::PathArguments::None);
                let path_args = if intrinsic.preserve_turbofish {
                    last_args
                } else {
                    syn::PathArguments::None
                };
                let internal_path = internal_thread_path(path, intrinsic.name, path_args);
                let scope_ident = &self.scope_ident;

                *expr = if intrinsic.forward_args {
                    parse_quote! { #internal_path(&#scope_ident, #args) }
                } else {
                    parse_quote! { #internal_path(&#scope_ident) }
                };
                self.rewrote_index_call = true;
            }
            Expr::MethodCall(ExprMethodCall { method, args, .. }) => {
                if !is_scoped_method(method) || !args.is_empty() {
                    return;
                }
                let scope_ident = &self.scope_ident;
                args.push(parse_quote! { &#scope_ident });
                self.rewrote_index_call = true;
            }
            _ => {}
        }
    }
}

fn inject_thread_index_scope(input: &mut ItemFn) {
    let scope_ident = format_ident!("{}", KERNEL_SCOPE_LOCAL);
    let mut rewriter = ThreadIndexCallRewriter {
        scope_ident: scope_ident.clone(),
        rewrote_index_call: false,
    };
    rewriter.visit_block_mut(&mut input.block);

    if rewriter.rewrote_index_call {
        let scope_stmt: Stmt = parse_quote! {
            let #scope_ident = unsafe { ::cuda_device::thread::__internal::make_kernel_scope() };
        };
        input.block.stmts.insert(0, scope_stmt);
    }
}

/// Generate a generic kernel that will be instantiated from call sites (nvcc-style)
fn generate_generic_kernel_no_instantiation(mut input: ItemFn) -> TokenStream {
    inject_thread_index_scope(&mut input);

    let fn_name = &input.sig.ident;
    let vis = &input.vis;
    let generics = &input.sig.generics;
    let where_clause = &input.sig.generics.where_clause;
    let inputs = &input.sig.inputs;
    let output = &input.sig.output;
    let block = &input.block;

    let kernel_name = format_ident!("{}{}", KERNEL_PREFIX, fn_name);
    let instantiate_name = format_ident!("{}{}", INSTANTIATE_PREFIX, fn_name);

    // For the wrapper function, strip `mut` from parameters since it just forwards them
    let wrapper_inputs = strip_mut_from_inputs(inputs);

    // Extract argument names and info for forwarding
    let args_info: Vec<_> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg
                && let Pat::Ident(pat_ident) = &*pat_type.pat
            {
                return Some((&pat_ident.ident, &*pat_type.ty));
            }
            None
        })
        .collect();

    let arg_names: Vec<_> = args_info.iter().map(|(name, _)| *name).collect();

    // Find the closure generic type (looks for Fn/FnMut/FnOnce bounds)
    let closure_generic = find_closure_generic(generics);

    // Extract generic type parameter names (T, F, etc.) for use in function pointer cast
    let generic_param_names: Vec<&syn::Ident> = generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Type(type_param) = p {
                Some(&type_param.ident)
            } else {
                None
            }
        })
        .collect();

    let marker_name = format_ident!("__{}_CudaKernel", fn_name);
    let instantiate_helper = if let Some(closure_type_name) = closure_generic {
        if let Some((_closure_idx, (_closure_name, closure_type))) =
            find_closure_param(&args_info, &closure_type_name)
        {
            let arg_types: Vec<TokenStream2> =
                args_info.iter().map(|(_, ty)| quote! { #ty }).collect();

            quote! {
                /// Auto-generated helper to force kernel monomorphization.
                ///
                /// Takes the closure by *reference* so its anonymous type
                /// is bound to the generic parameter `F` at the call site
                /// without moving the closure — the caller still needs the
                /// closure value to push as the kernel argument right
                /// after. Then forces rustc to emit a CGU entry for the
                /// concrete `#kernel_name::<...>` instantiation. Returns
                /// the PTX export name produced by the kernel's
                /// `GenericCudaKernel::ptx_name()` impl, which is the
                /// single source of truth for the on-wire name on the
                /// host side.
                ///
                /// Bound is intentionally not `'static`: closures that
                /// borrow non-`'static` data (e.g. capture `&[T]`) still
                /// monomorphize cleanly. The caller is responsible for
                /// keeping that borrow alive across the asynchronous
                /// launch — `cuda_host::type_id_u128` does not enforce
                /// this.
                #[doc(hidden)]
                #[inline(never)]
                #vis fn #instantiate_name #generics (_f: &#closure_type) -> &'static str #where_clause {
                    let __kernel_ptr = #kernel_name::<#(#generic_param_names),*> as fn(#(#arg_types),*) as *const ();
                    unsafe {
                        let mut __force_mono: *const () = core::ptr::null();
                        core::ptr::write_volatile(&mut __force_mono, __kernel_ptr);
                        let _ = core::ptr::read_volatile(&__force_mono);
                    }
                    <#marker_name::<#(#generic_param_names),*> as cuda_host::GenericCudaKernel>::ptx_name()
                }
            }
        } else {
            quote! {}
        }
    } else {
        quote! {}
    };

    // Generate the GenericCudaKernel trait implementation for unified compilation
    let generic_cuda_kernel_impl =
        generate_generic_cuda_kernel_impl(fn_name, generics, where_clause);

    let expanded = quote! {
        // Original generic kernel implementation
        #[inline(always)]
        #vis fn #fn_name #generics (#inputs) #output #where_clause
        #block

        // Entry point for collector - NOT inlined so we can detect it
        // When called with concrete types, this instantiates the kernel
        // Note: wrapper_inputs has `mut` stripped since we just forward args
        #[inline(never)]
        #vis fn #kernel_name #generics (#(#wrapper_inputs),*) #output #where_clause {
            #fn_name(#(#arg_names),*)
        }

        #instantiate_helper

        #generic_cuda_kernel_impl
    };

    TokenStream::from(expanded)
}

/// Generate a dummy binding for a given type.
/// Used by instantiate! helper to create zero-valued arguments.
///
/// The generated values are never actually executed - they exist only to force
/// rustc to monomorphize the kernel with the correct types.
fn _generate_dummy_binding(name: &Ident, ty: &Type) -> TokenStream2 {
    match ty {
        // Special case: &[T] or &mut [T] → empty slice literal
        // (slices don't implement Default and can't be safely zeroed)
        Type::Reference(type_ref) if matches!(&*type_ref.elem, Type::Slice(_)) => {
            if let Type::Slice(slice) = &*type_ref.elem {
                let elem_ty = &slice.elem;
                if type_ref.mutability.is_some() {
                    quote! { let #name: &mut [#elem_ty] = &mut []; }
                } else {
                    quote! { let #name: &[#elem_ty] = &[]; }
                }
            } else {
                unreachable!()
            }
        }

        // Everything else: use mem::zeroed()
        // Safe because this code never actually runs - it only exists to
        // force monomorphization of the kernel with the correct types.
        _ => {
            quote! { let #name: #ty = unsafe { core::mem::zeroed() }; }
        }
    }
}

/// Generate a simple non-generic kernel
fn generate_simple_kernel(mut input: ItemFn) -> TokenStream {
    inject_thread_index_scope(&mut input);

    let fn_name = input.sig.ident.clone();
    let new_name = format_ident!("{}{}", KERNEL_PREFIX, fn_name);

    // Clone the original function for the CudaKernel impl
    let original_fn = input.clone();
    input.sig.ident = new_name;

    // PTX entry name is the unprefixed user name; the collector strips
    // KERNEL_PREFIX when generating PTX.
    let ptx_entry_name = fn_name.to_string();

    // Generate the CudaKernel trait implementation (host-side only)
    // This provides the PTX name for cuda_launch! to look up
    let cuda_kernel_impl = generate_cuda_kernel_impl(&fn_name, &ptx_entry_name, &original_fn);

    let expanded = quote! {
        #[unsafe(no_mangle)]
        #input

        #cuda_kernel_impl
    };

    TokenStream::from(expanded)
}

/// Generate the GenericCudaKernel trait implementation for a generic kernel.
///
/// For generic kernels like `fn scale<T>()`, emits:
///
/// ```ignore
/// pub struct __scale_CudaKernel<T>(PhantomData<T>);
/// impl<T> GenericCudaKernel for __scale_CudaKernel<T> {
///     fn ptx_name() -> &'static str {
///         // "scale_TID_<hex32>" — one 32-char hex chunk for the
///         // 1-tuple `(T,)`. For an N-generic kernel we hash the
///         // N-tuple `(T0, T1, ...)` so the name length is constant
///         // regardless of arity.
///     }
/// }
/// ```
///
/// The body computes the same string the backend writes into the PTX:
/// `<base>_TID_<hex32>`, where `<hex32>` is
/// `cuda_host::type_id_u128::<(T0, T1, ...,)>()` rendered as 32
/// lowercase hex chars. The backend's `compute_kernel_export_name`
/// computes the same hash via `Ty::new_tup(tcx, &[T0, T1, ...])` +
/// `tcx.type_id_hash(...)`, so the two strings match byte-for-byte.
///
/// Bound on the impl is `where_clause` verbatim — typically `Copy` on
/// each value-passed generic. We deliberately do not add `'static`:
/// `type_id_u128` has bound `T: ?Sized`, so closure types that borrow
/// non-`'static` data still satisfy the marker's bounds and can be
/// launched through the typed `module.<kernel>(...)` path. Keeping the
/// borrow alive across `stream.synchronize()` remains the caller's
/// responsibility, exactly as it was under the previous `type_name`
/// scheme.
fn generate_generic_cuda_kernel_impl(
    fn_name: &Ident,
    generics: &syn::Generics,
    where_clause: &Option<syn::WhereClause>,
) -> TokenStream2 {
    let marker_name = format_ident!("__{}_CudaKernel", fn_name);
    let base_name = fn_name.to_string();

    let type_params: Vec<_> = generics.params.iter().collect();
    let type_param_names: Vec<_> = generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Type(tp) = p {
                Some(&tp.ident)
            } else {
                None
            }
        })
        .collect();

    let ptx_name_body = if type_param_names.is_empty() {
        quote! {
            fn ptx_name() -> &'static str {
                #base_name
            }
        }
    } else {
        // Trailing comma in the tuple type expression keeps the
        // arity-1 case `(T,)` a real 1-tuple — without it,
        // `(T)` would just be a parenthesized type and the hash
        // would differ from the backend's `Ty::new_tup(tcx, &[T])`.
        quote! {
            fn ptx_name() -> &'static str {
                let __hash = ::cuda_host::type_id_u128::<( #(#type_param_names,)* )>();
                let name = format!("{}_TID_{:032x}", #base_name, __hash);
                Box::leak(name.into_boxed_str())
            }
        }
    };

    quote! {
        /// Marker type for a generic kernel; implements `GenericCudaKernel`.
        /// The type parameters mirror the kernel's generic parameters.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        pub struct #marker_name<#(#type_params),*>(
            core::marker::PhantomData<(#(#type_param_names),*)>
        ) #where_clause;

        impl<#(#type_params),*> cuda_host::GenericCudaKernel for #marker_name<#(#type_param_names),*>
        #where_clause
        {
            #ptx_name_body
        }
    }
}

/// Generate the CudaKernel trait implementation for a kernel function.
///
/// This generates a marker struct that implements `CudaKernel`, allowing
/// `cuda_launch!` to look up the PTX entry point name at compile time.
fn generate_cuda_kernel_impl(fn_name: &Ident, ptx_name: &str, _func: &ItemFn) -> TokenStream2 {
    // Create a marker struct for this kernel
    // We use a struct because Rust doesn't allow trait impls on function pointers easily
    let marker_name = format_ident!("__{}_CudaKernel", fn_name);

    quote! {
        /// Marker type for the kernel, implements CudaKernel trait.
        /// This enables cuda_launch! to look up the PTX entry point name.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        pub struct #marker_name;

        impl cuda_host::CudaKernel for #marker_name {
            const PTX_NAME: &'static str = #ptx_name;
        }
    }
}

/// Generate wrapper kernels for a generic kernel
fn generate_generic_kernel(mut input: ItemFn, instantiate_types: Vec<Type>) -> TokenStream {
    inject_thread_index_scope(&mut input);

    let fn_name = &input.sig.ident;
    let vis = &input.vis;
    let generics = &input.sig.generics;

    // Extract the type parameter name (assume single type param for now)
    let type_param = generics
        .params
        .iter()
        .find_map(|p| {
            if let GenericParam::Type(tp) = p {
                Some(&tp.ident)
            } else {
                None
            }
        })
        .expect("Expected type parameter");

    // Extract function arguments (excluding self)
    let args: Vec<_> = input.sig.inputs.iter().collect();

    // Build the argument pattern and types for wrappers
    let arg_names: Vec<TokenStream2> = args
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg
                && let Pat::Ident(pat_ident) = &*pat_type.pat
            {
                return Some(quote! { #pat_ident });
            }
            None
        })
        .collect();

    // For each instantiation type, generate a wrapper that substitutes the type
    let wrappers: Vec<TokenStream2> = instantiate_types
        .iter()
        .map(|inst_type| {
            // Get a clean name for the type (for the kernel name suffix)
            let type_name = get_type_name(inst_type);
            let wrapper_name = format_ident!("{}{}_{}", KERNEL_PREFIX, fn_name, type_name);

            // Export name (what appears in PTX)
            let export_name_str = format!("{}_{}", fn_name, type_name);

            // Generate wrapper function args with substituted types
            let wrapper_args: Vec<TokenStream2> = args
                .iter()
                .map(|arg| {
                    if let FnArg::Typed(pat_type) = arg {
                        let pat = &pat_type.pat;
                        let ty = &pat_type.ty;
                        // Substitute type parameter with concrete type
                        let subst_ty = substitute_type(ty, type_param, inst_type);
                        quote! { #pat: #subst_ty }
                    } else {
                        quote! { #arg }
                    }
                })
                .collect();

            quote! {
                #[unsafe(no_mangle)]
                #[unsafe(export_name = #export_name_str)]
                #vis fn #wrapper_name(#(#wrapper_args),*) {
                    #fn_name::<#inst_type>(#(#arg_names),*);
                }
            }
        })
        .collect();

    // Keep the original generic function (without #[no_mangle] - it's not an entry point)
    // and add all the wrapper kernels
    let expanded = quote! {
        #[inline(always)]
        #input

        #(#wrappers)*
    };

    TokenStream::from(expanded)
}

/// Get a clean name from a type for use in function names
fn get_type_name(ty: &Type) -> String {
    match ty {
        Type::Path(type_path) => {
            // Get the last segment (e.g., "Scale" from "crate::Scale")
            type_path
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_else(|| "Unknown".to_string())
        }
        _ => "Unknown".to_string(),
    }
}

/// Substitute a type parameter with a concrete type in a type expression
fn substitute_type(ty: &Type, param: &syn::Ident, replacement: &Type) -> TokenStream2 {
    match ty {
        Type::Path(type_path) => {
            // Check if this is just the type parameter
            if type_path.path.is_ident(param) {
                return quote! { #replacement };
            }
            quote! { #ty }
        }
        Type::Reference(type_ref) => {
            let elem = substitute_type(&type_ref.elem, param, replacement);
            let lifetime = &type_ref.lifetime;
            let mutability = &type_ref.mutability;
            quote! { &#lifetime #mutability #elem }
        }
        _ => quote! { #ty },
    }
}

/// Specifies launch bounds for a kernel (max threads per block, min blocks per SM).
///
/// This attribute sets kernel launch bounds at compile time by emitting `.maxntid`
/// and `.minnctapersm` PTX directives. This helps the CUDA compiler optimize
/// register allocation and occupancy.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::{kernel, launch_bounds, DisjointSlice};
///
/// #[kernel]
/// #[launch_bounds(256)]              // Max 256 threads per block
/// pub fn simple_kernel(output: DisjointSlice<f32>) { ... }
///
/// #[kernel]
/// #[launch_bounds(256, 2)]           // Max 256 threads, min 2 blocks per SM
/// pub fn optimized_kernel(output: DisjointSlice<f32>) { ... }
/// ```
///
/// # Parameters
///
/// - First parameter (required): Maximum threads per block
/// - Second parameter (optional): Minimum blocks per SM for occupancy hints
///
/// # Requirements
///
/// - Must be used WITH `#[kernel]` (not standalone)
/// - The `#[launch_bounds]` attribute must come AFTER `#[kernel]`
///
/// # Performance Impact
///
/// Launch bounds help the compiler:
/// - Allocate registers more efficiently
/// - Optimize occupancy (threads per SM)
/// - Make better scheduling decisions
///
/// # PTX Output
///
/// ```ptx
/// .entry my_kernel .maxntid 256 .minnctapersm 2 { ... }
/// ```
#[proc_macro_attribute]
pub fn launch_bounds(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args: LaunchBoundsArgs = parse_macro_input!(attr as LaunchBoundsArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let max_threads = args.max_threads;
    let min_blocks = args.min_blocks;

    // Inject the launch bounds config marker at the start of the function body
    let marker_call: syn::Stmt = syn::parse_quote! {
        cuda_device::thread::__launch_bounds_config::<#max_threads, #min_blocks>();
    };

    // Prepend the marker to the function body
    input.block.stmts.insert(0, marker_call);

    quote! {
        #input
    }
    .into()
}

/// Arguments for `#[launch_bounds(...)]` attribute.
struct LaunchBoundsArgs {
    max_threads: u32,
    min_blocks: u32,
}

impl Parse for LaunchBoundsArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args: Punctuated<syn::LitInt, Token![,]> = Punctuated::parse_terminated(input)?;
        let values: Vec<u32> = args
            .iter()
            .map(|lit| lit.base10_parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;

        match values.len() {
            1 => Ok(LaunchBoundsArgs {
                max_threads: values[0],
                min_blocks: 0, // Unspecified
            }),
            2 => Ok(LaunchBoundsArgs {
                max_threads: values[0],
                min_blocks: values[1],
            }),
            _ => Err(syn::Error::new_spanned(
                args.first().unwrap(),
                "launch_bounds expects 1 or 2 parameters: #[launch_bounds(max_threads)] or #[launch_bounds(max_threads, min_blocks)]",
            )),
        }
    }
}

/// Arguments for the `#[unroll]` / `#[unroll(N)]` attribute.
///
/// Bare `#[unroll]` parses to factor `0` (full unroll); `#[unroll(N)]` requires
/// `N >= 2`.
struct UnrollArgs {
    factor: u32,
}

impl Parse for UnrollArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Bare `#[unroll]` => full unroll (factor 0).
        if input.is_empty() {
            return Ok(UnrollArgs { factor: 0 });
        }

        let args: Punctuated<syn::LitInt, Token![,]> = Punctuated::parse_terminated(input)?;
        let values: Vec<u32> = args
            .iter()
            .map(|lit| lit.base10_parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;

        match values.len() {
            1 if values[0] >= 2 => Ok(UnrollArgs { factor: values[0] }),
            1 => Err(syn::Error::new_spanned(
                args.first().unwrap(),
                "partial unroll factor must be at least 2; use #[unroll] for full unrolling",
            )),
            _ => Err(syn::Error::new_spanned(
                args.first().unwrap(),
                "unroll expects no argument (full unroll) or one factor: #[unroll] or #[unroll(N)]",
            )),
        }
    }
}

/// `VisitMut` pass that consumes `#[unroll]` / `#[unroll(N)]` attributes written
/// directly on loops inside a `#[kernel]` (or `#[device]`) function body.
///
/// This mirrors how CubeCL's `#[cube]` macro handles per-loop `#[unroll]`: the
/// enclosing function macro owns the whole body AST, so it can strip the inner
/// attribute and lower it into a marker BEFORE rustc ever sees it. That keeps us
/// off the nightly `stmt_expr_attributes` feature (rustc otherwise rejects
/// attributes on expressions).
///
/// For every `for` / `while` / `loop` expression carrying an outer `#[unroll]`
/// attribute, the visitor:
///
/// 1. removes the `unroll` attribute from the loop expr's `attrs`, and
/// 2. inserts `cuda_device::thread::__unroll_config::<FACTOR>();` as the FIRST
///    statement of that loop's body block.
///
/// `FACTOR` follows [`UnrollArgs`]: bare `#[unroll]` => `0` (full unroll),
/// `#[unroll(N)]` => `N`. The importer reads the marker from the block it lands
/// in, so the request applies to that loop only.
///
/// The visitor recurses through nested blocks/loops/ifs via the default
/// `visit_mut` traversal, so an annotated loop anywhere in the function is
/// handled. Loops without an `#[unroll]` attribute are left untouched, so a
/// kernel with no per-loop annotations expands byte-identically to before.
#[derive(Default)]
struct LoopUnrollAttrVisitor {
    /// First parse error encountered (e.g. a malformed `#[unroll(...)]`). The
    /// caller surfaces this as a compile error.
    error: Option<syn::Error>,
}

impl LoopUnrollAttrVisitor {
    /// If `attrs` contains an outer `#[unroll]` / `#[unroll(N)]`, remove it and
    /// return the parsed factor. Returns `None` when no `unroll` attribute is
    /// present (leaving `attrs` untouched). Records a parse error and returns
    /// `None` if the attribute is malformed.
    fn take_unroll_factor(&mut self, attrs: &mut Vec<syn::Attribute>) -> Option<u32> {
        let idx = attrs
            .iter()
            .position(|attr| attr.path().is_ident("unroll"))?;
        let attr = attrs.remove(idx);

        // `#[unroll]` (bare) is `Meta::Path`; `#[unroll(N)]` is `Meta::List`.
        let factor = match &attr.meta {
            syn::Meta::Path(_) => 0u32,
            syn::Meta::List(list) => match list.parse_args::<UnrollArgs>() {
                Ok(parsed) => parsed.factor,
                Err(err) => {
                    if self.error.is_none() {
                        self.error = Some(err);
                    }
                    return None;
                }
            },
            syn::Meta::NameValue(_) => {
                if self.error.is_none() {
                    self.error = Some(syn::Error::new_spanned(
                        &attr,
                        "unroll expects no argument (full unroll) or one factor: #[unroll] or #[unroll(N)]",
                    ));
                }
                return None;
            }
        };
        Some(factor)
    }

    /// Build the `__unroll_config::<FACTOR>()` marker statement.
    fn marker_stmt(factor: u32) -> Stmt {
        parse_quote! {
            cuda_device::thread::__unroll_config::<#factor>();
        }
    }
}

impl VisitMut for LoopUnrollAttrVisitor {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // Pull the unroll factor (if any) off this loop expr and inject the
        // marker into its body. We act only on loop expressions; everything
        // else falls through to the default recursion below.
        match expr {
            Expr::ForLoop(for_loop) => {
                if let Some(factor) = self.take_unroll_factor(&mut for_loop.attrs) {
                    for_loop.body.stmts.insert(0, Self::marker_stmt(factor));
                }
            }
            Expr::While(while_loop) => {
                if let Some(factor) = self.take_unroll_factor(&mut while_loop.attrs) {
                    while_loop.body.stmts.insert(0, Self::marker_stmt(factor));
                }
            }
            Expr::Loop(loop_expr) => {
                if let Some(factor) = self.take_unroll_factor(&mut loop_expr.attrs) {
                    loop_expr.body.stmts.insert(0, Self::marker_stmt(factor));
                }
            }
            _ => {}
        }

        // Recurse into nested expressions/blocks so annotated loops nested
        // inside other loops, `if`s, or blocks are also handled.
        visit_mut::visit_expr_mut(self, expr);
    }
}

/// Consume per-loop unroll annotations in a kernel or device function and
/// replace them with the marker calls understood by the MIR importer.
fn rewrite_loop_unroll_attrs(input: &mut ItemFn) -> syn::Result<()> {
    let mut visitor = LoopUnrollAttrVisitor::default();
    visitor.visit_block_mut(&mut input.block);
    match visitor.error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Specifies compile-time cluster dimensions for a kernel.
///
/// This attribute sets the thread block cluster size at compile time by emitting
/// the `.reqnctapercluster` PTX directive. When used, the kernel will automatically
/// launch with the specified cluster configuration.
///
/// Note: Named `cluster_launch` (not `cluster`) to avoid conflict with `cuda_device::cluster` module.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::{kernel, cluster, cluster_launch, DisjointSlice};
///
/// #[kernel]
/// #[cluster_launch(4, 1, 1)]  // 4 blocks per cluster in X dimension
/// pub fn my_cluster_kernel(output: DisjointSlice<u32>) {
///     let rank = cluster::block_rank();
///     // ...
/// }
/// ```
///
/// # Cluster Dimensions
///
/// - `#[cluster_launch(n)]` - 1D cluster with n blocks
/// - `#[cluster_launch(x, y)]` - 2D cluster with x*y blocks
/// - `#[cluster_launch(x, y, z)]` - 3D cluster with x*y*z blocks
///
/// Maximum cluster size is typically 16 blocks (hardware dependent).
///
/// # Requirements
///
/// - Must be used WITH `#[kernel]` (not standalone)
/// - Requires sm_90+ (Hopper) or newer GPU
/// - The `#[cluster_launch]` attribute must come AFTER `#[kernel]`
///
/// # How It Works
///
/// The macro injects `cuda_device::cluster::__cluster_config::<X, Y, Z>()` at the
/// start of the kernel. The compiler:
/// 1. Detects this marker during MIR translation
/// 2. Extracts the const generic parameters (X, Y, Z)
/// 3. Emits `!nvvm.annotations` metadata with cluster dimensions
/// 4. LLVM NVPTX backend generates `.reqnctapercluster X, Y, Z` in PTX
///
/// # PTX Output
///
/// ```ptx
/// .entry my_cluster_kernel .reqnctapercluster 4, 1, 1 { ... }
/// ```
///
/// # Compile-Time vs Runtime Clusters
///
/// | Method | Pros | Cons |
/// |--------|------|------|
/// | `#[cluster_launch(x,y,z)]` (compile-time) | Simple, no special launch API | Fixed at compile time |
/// | `cuLaunchKernelEx` (runtime) | Dynamic cluster sizes | Requires FFI, complex setup |
#[proc_macro_attribute]
pub fn cluster_launch(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args: ClusterArgs = parse_macro_input!(attr as ClusterArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let x = args.x;
    let y = args.y;
    let z = args.z;

    // Inject the cluster config marker at the start of the function body
    let marker_call: syn::Stmt = syn::parse_quote! {
        cuda_device::cluster::__cluster_config::<#x, #y, #z>();
    };

    // Prepend the marker to the function body
    input.block.stmts.insert(0, marker_call);

    quote! {
        #input
    }
    .into()
}

/// Arguments for `#[cluster_launch(...)]` attribute.
struct ClusterArgs {
    x: u32,
    y: u32,
    z: u32,
}

impl Parse for ClusterArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args: Punctuated<syn::LitInt, Token![,]> = Punctuated::parse_terminated(input)?;
        let values: Vec<u32> = args
            .iter()
            .map(|lit| lit.base10_parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;

        match values.len() {
            1 => Ok(ClusterArgs {
                x: values[0],
                y: 1,
                z: 1,
            }),
            2 => Ok(ClusterArgs {
                x: values[0],
                y: values[1],
                z: 1,
            }),
            3 => Ok(ClusterArgs {
                x: values[0],
                y: values[1],
                z: values[2],
            }),
            _ => Err(syn::Error::new_spanned(
                args.first().unwrap(),
                "cluster expects 1, 2, or 3 dimensions: #[cluster(x)], #[cluster(x, y)], or #[cluster(x, y, z)]",
            )),
        }
    }
}

/// Marks a kernel for cooperative launch (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`).
///
/// A cooperative launch guarantees that every block in the grid is
/// co-resident on the device, which is the precondition for grid-wide
/// barriers: without it, `cuda_device::grid::sync()` deadlocks (or reads a
/// null grid-workspace pointer) because blocks that have not been scheduled
/// yet can never reach the barrier.
///
/// Unlike `#[cluster_launch]`, this attribute changes nothing in the
/// generated PTX. Cooperative-ness is purely a launch-time property: the
/// `#[cuda_module]` macro reads this marker and routes every generated
/// launch method through `cuLaunchKernelEx` with the cooperative attribute
/// set, instead of plain `cuLaunchKernel`.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::{cooperative_launch, grid, kernel, DisjointSlice};
///
/// #[kernel]
/// #[cooperative_launch]
/// pub fn my_grid_sync_kernel(mut out: DisjointSlice<u32>) {
///     // ... per-block work ...
///     grid::sync();
///     // ... grid-wide post-barrier work ...
/// }
/// ```
///
/// # Requirements
///
/// - Must be used WITH `#[kernel]` (not standalone), on a kernel inside a
///   `#[cuda_module]` module
/// - The `#[cooperative_launch]` attribute must come AFTER `#[kernel]`
/// - The device must support cooperative launch
///   (`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`)
/// - The grid must fit on the device in one wave, otherwise the driver
///   rejects the launch with `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`
///
/// May be combined with `#[cluster_launch(x, y, z)]`; both launch
/// attributes are then passed to `cuLaunchKernelEx` in the same call.
///
/// Outside `#[cuda_module]`, the legacy (caller-unsafe) `cuda_launch!`
/// macro offers the same behaviour through its `cooperative: true` field.
#[proc_macro_attribute]
pub fn cooperative_launch(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "cooperative_launch takes no arguments: use a bare #[cooperative_launch]",
        )
        .to_compile_error()
        .into();
    }

    // Launch-time only: the marker is consumed by #[cuda_module]; the kernel
    // body and PTX are unchanged. Parse as a function so misuse on other
    // items is rejected with a clear error.
    let input = parse_macro_input!(item as ItemFn);
    quote! {
        #input
    }
    .into()
}

/// Marks a function as a CUDA device function.
///
/// Device functions run on the GPU and can be called from kernels or other device functions,
/// but cannot be called from host code.
///
/// This attribute:
/// 1. Adds `#[no_mangle]` to preserve the function name in the binary
/// 2. Renames the function into the reserved `cuda_oxide_device_<hash>_` namespace
///    for detection by the codegen backend (the prefix lives in
///    `crates/reserved-oxide-symbols/`)
/// 3. Marks the function for extraction by the `rustc-codegen-cuda` backend
///
/// Device functions can:
/// - Return values (unlike kernels which must return `()`)
/// - Be called from kernels and other device functions
/// - Use generics (each monomorphization becomes a separate device function)
/// - Use per-loop `#[unroll]` and `#[unroll(N)]` annotations
///
/// # Loop unrolling
///
/// Loop annotations work the same way in device function definitions as they do
/// in kernels. Use an explicit counted `while` loop; range-based `for` loops are
/// not yet recognized. Partial factors must be `N >= 2`. Multiple `continue`
/// paths are supported; full unrolling preserves `break` and multiple exit
/// targets. Partial unrolling requires a positive counter step, a `<` or `<=`
/// test, an unchanging limit, and no exit besides the normal header test.
///
/// One annotation may create at most 1,024 body copies, 8,192 cloned basic
/// blocks, and 65,536 cloned operations. Unsupported or larger requests warn
/// and are not unrolled.
///
/// # Example: Device Function Definition
///
/// ```ignore
/// use cuda_device::device;
///
/// #[device]
/// pub fn helper(x: f32, y: f32) -> f32 {
///     x * x + y * y
/// }
///
/// #[kernel]
/// pub fn my_kernel(data: *mut f32) {
///     let result = helper(1.0, 2.0);
///     unsafe { *data = result; }
/// }
/// ```
///
/// # Example: External Device Function Declaration (FFI)
///
/// ```ignore
/// use cuda_device::{device, convergent};
///
/// // Declare external device functions from LTOIR (e.g., CCCL)
/// #[device]
/// extern "C" {
///     #[convergent]
///     fn cub_block_reduce_sum_f32(input: f32, temp: *mut u8) -> f32;
///
///     fn fast_math_helper(x: f32) -> f32;
/// }
///
/// #[kernel]
/// pub fn my_kernel(data: *mut f32) {
///     let result = unsafe { cub_block_reduce_sum_f32(*data, temp_ptr) };
/// }
/// ```
#[proc_macro_attribute]
pub fn device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Try parsing as a function definition first
    if let Ok(input) = syn::parse::<ItemFn>(item.clone()) {
        return generate_device_function(input);
    }

    // Try parsing as an extern block
    if let Ok(input) = syn::parse::<ItemForeignMod>(item.clone()) {
        return generate_device_extern_block(input);
    }

    // Neither worked - emit error
    syn::Error::new_spanned(
        proc_macro2::TokenStream::from(item),
        "#[device] can only be applied to functions or extern blocks",
    )
    .to_compile_error()
    .into()
}

/// Generate a device function definition.
///
/// Renames the function into the reserved `cuda_oxide_device_<hash>_` namespace
/// for collector detection, and generates a thin wrapper with the original name
/// so user code can call `my_func()` rather than the mangled internal symbol.
///
/// Handles both non-generic and generic device functions:
/// - **Non-generic**: `#[no_mangle]` on the prefixed function, `#[inline(always)]` wrapper.
/// - **Generic**: No `#[no_mangle]` (generics use mangled symbols), `#[inline(never)]` on
///   the prefixed function (so monomorphizations appear in CGUs for the collector),
///   `#[inline(always)]` wrapper with generics + turbofish forwarding.
///
/// This mirrors the pattern used by `#[kernel]` for generic kernels
/// (see `generate_generic_kernel_no_instantiation`).
fn generate_device_function(mut input: ItemFn) -> TokenStream {
    if let Some(err) = reject_reserved_name(&input.sig.ident) {
        return err;
    }
    if let Err(err) = rewrite_loop_unroll_attrs(&mut input) {
        return err.to_compile_error().into();
    }
    inject_thread_index_scope(&mut input);

    let fn_name = input.sig.ident.clone();
    let vis = input.vis.clone();
    let new_name = format_ident!("{}{}", DEVICE_PREFIX, fn_name);

    // Check if the function has type parameters
    let has_generics = input
        .sig
        .generics
        .params
        .iter()
        .any(|p| matches!(p, GenericParam::Type(_)));

    // Extract parameter names for forwarding
    let params: Vec<_> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg
                && let Pat::Ident(pat_ident) = &*pat_type.pat
            {
                return Some(pat_ident.ident.clone());
            }
            None
        })
        .collect();

    let return_type = &input.sig.output;
    let generics = &input.sig.generics;
    let where_clause = &input.sig.generics.where_clause;

    // Strip `mut` from wrapper parameters since the wrapper just forwards args.
    // In Rust, `mut` on a by-value parameter is purely local binding mutability —
    // it's not part of the function's type signature and callers don't need `mut`
    // to pass a value. The original (renamed) function keeps `mut` for its body,
    // but the wrapper only forwards the value and never mutates it locally.
    let wrapper_inputs = strip_mut_from_inputs(&input.sig.inputs);

    // Rename the original function with the prefix
    input.sig.ident = new_name.clone();

    if has_generics {
        // Generic device function: mirrors the generic kernel pattern.
        //
        // - No #[no_mangle] — generic functions use mangled symbol names per
        //   monomorphization (e.g., `cuda_oxide_device_<hash>_add::<f32>` gets a
        //   unique mangled name). #[no_mangle] requires a single concrete symbol.
        //
        // - #[inline(never)] on the prefixed function — ensures each monomorphization
        //   appears as a distinct CGU item so the collector can find it. If it were
        //   inlined, the function would disappear from the CGU.
        //
        // - The wrapper forwards type parameters via turbofish:
        //   `cuda_oxide_device_<hash>_add::<T>(a, b)`.

        // Extract type parameter names for turbofish forwarding (T, U, etc.)
        let type_param_names: Vec<&syn::Ident> = generics
            .params
            .iter()
            .filter_map(|p| {
                if let GenericParam::Type(type_param) = p {
                    Some(&type_param.ident)
                } else {
                    None
                }
            })
            .collect();

        let expanded = quote! {
            #[inline(never)]
            #input

            /// Wrapper for the generic device function with the original name.
            #[inline(always)]
            #vis fn #fn_name #generics (#(#wrapper_inputs),*) #return_type #where_clause {
                #new_name::<#(#type_param_names),*>(#(#params),*)
            }
        };

        TokenStream::from(expanded)
    } else {
        // Non-generic device function: simple case.
        let expanded = quote! {
            #[unsafe(no_mangle)]
            #input

            /// Wrapper for the device function with the original name.
            #[inline(always)]
            #vis fn #fn_name #generics (#(#wrapper_inputs),*) #return_type #where_clause {
                #new_name(#(#params),*)
            }
        };

        TokenStream::from(expanded)
    }
}

/// Generate device extern block declarations (for FFI with external LTOIR).
///
/// For each function in the extern block:
/// 1. Rename it into the reserved `cuda_oxide_device_extern_<hash>_` namespace
///    (for collector detection)
/// 2. Generate a wrapper function with the original name (for user code)
///
/// User code calls `foo()` while the collector sees the hash-suffixed reserved
/// form. The `#[link_name]` attribute restores the original name in the binary
/// so external LTOIR resolves correctly.
fn generate_device_extern_block(mut input: ItemForeignMod) -> TokenStream {
    // Device extern declarations use CUDA's C calling convention. Reject
    // other Rust ABIs because their argument and return conventions may differ.
    if input
        .abi
        .name
        .as_ref()
        .is_some_and(|name| name.value() != "C")
    {
        return syn::Error::new_spanned(
            &input.abi,
            "#[device] extern blocks must use `extern \"C\"`",
        )
        .to_compile_error()
        .into();
    }

    let mut wrappers = Vec::new();

    // Process each item in the extern block
    for item in &mut input.items {
        if let ForeignItem::Fn(foreign_fn) = item {
            if let Some(err) = reject_reserved_name(&foreign_fn.sig.ident) {
                return err;
            }

            // Save original info for wrapper generation
            let original_name = foreign_fn.sig.ident.clone();
            let original_attrs = foreign_fn.attrs.clone();
            let original_sig = foreign_fn.sig.clone();

            let new_name = format_ident!("{}{}", DEVICE_EXTERN_PREFIX, original_name);
            foreign_fn.sig.ident = new_name.clone();

            // Store original name as link_name for the linker
            let original_name_str = original_name.to_string();
            foreign_fn.attrs.push(syn::parse_quote! {
                #[doc(hidden)]
            });
            foreign_fn.attrs.push(syn::parse_quote! {
                #[link_name = #original_name_str]
            });

            // Generate wrapper function with the original name. User code
            // calls `foo()`; the wrapper forwards to the reserved internal
            // symbol the macro just produced.
            let params: Vec<_> = original_sig
                .inputs
                .iter()
                .filter_map(|arg| {
                    if let syn::FnArg::Typed(pat_type) = arg
                        && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
                    {
                        return Some(pat_ident.ident.clone());
                    }
                    None
                })
                .collect();

            let return_type = &original_sig.output;
            let inputs = &original_sig.inputs;

            // Keep user's attributes (like #[convergent]) on the wrapper
            let wrapper = quote! {
                #(#original_attrs)*
                #[inline(always)]
                #[allow(non_snake_case)]
                pub unsafe fn #original_name(#inputs) #return_type {
                    #new_name(#(#params),*)
                }
            };
            wrappers.push(wrapper);
        }
    }

    let expanded = quote! {
        #input

        #(#wrappers)*
    };

    TokenStream::from(expanded)
}

// ============================================================================
// NVVM Attributes for Device FFI
// ============================================================================

/// Marks a device function as convergent.
///
/// Convergent functions must be called by all threads in a warp/block together.
/// This prevents the optimizer from moving calls across control flow boundaries.
///
/// # When to Use
///
/// - Synchronization primitives (`__syncthreads`, barriers)
/// - Warp-collective operations (`__shfl_*`, warp vote, warp reduce)
/// - Block-collective operations (CUB block reduce/scan)
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[convergent]
///     fn cub_block_reduce_sum(input: f32, temp: *mut u8) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @cub_block_reduce_sum(float, ptr) #0
/// attributes #0 = { convergent nounwind }
/// ```
#[proc_macro_attribute]
pub fn convergent(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This is a marker attribute - just pass through the item unchanged.
    // The collector will read this attribute and apply the LLVM convergent attribute.
    item
}

/// Marks a device function as pure (no side effects).
///
/// Pure functions only depend on their inputs and have no side effects.
/// This enables aggressive optimizations like CSE and dead code elimination.
///
/// # When to Use
///
/// - Math functions that don't access memory
/// - Functions that compute results purely from input arguments
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[pure]
///     fn fast_rsqrt(x: f32) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @fast_rsqrt(float) #0
/// attributes #0 = { nounwind readnone }
/// ```
#[proc_macro_attribute]
pub fn pure(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Marker attribute - collector will read and apply LLVM readnone attribute
    item
}

/// Marks a device function as read-only.
///
/// Read-only functions may read memory but never write to it.
/// This enables optimizations like load hoisting and caching.
///
/// # When to Use
///
/// - Lookup table functions
/// - Functions that only read from input arrays
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[readonly]
///     fn lookup_table(table: *const f32, idx: i32) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @lookup_table(ptr, i32) #0
/// attributes #0 = { nounwind readonly }
/// ```
#[proc_macro_attribute]
pub fn readonly(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Marker attribute - collector will read and apply LLVM readonly attribute
    item
}

// ============================================================================
// cuda_launch! Macro (unified compilation)
// ============================================================================

/// Try to extract closure from an expression.
///
/// Closure marshalling no longer needs per-capture extraction: the
/// backend emits a single byval `.param` for the whole closure struct,
/// and the host pushes one scalar. The closure literal is still parsed
/// out of the launch args so the `instantiate_name` helper has a
/// concrete `&F` to bind the kernel's generic closure type to.
fn as_closure_expr(expr: &syn::Expr) -> Option<&syn::ExprClosure> {
    match expr {
        syn::Expr::Closure(closure) => Some(closure),
        syn::Expr::Group(group) => as_closure_expr(&group.expr),
        syn::Expr::Paren(paren) => as_closure_expr(&paren.expr),
        _ => None,
    }
}

/// Argument type for cuda_launch! - same as LaunchArg but renamed for clarity
enum CudaLaunchArg {
    /// Direct expression - passed via .arg()
    Direct(syn::Expr),
    /// Slice with explicit length - passed as ptr + len
    SliceWithLen(syn::Expr),
    /// Mutable slice with explicit length - passed as ptr + len
    SliceMutWithLen(syn::Expr),
    /// Closure expression. The closure value is pushed as a single byval
    /// scalar argument; the backend emits a matching single .param entry
    /// for aggregate kernel parameters. No per-capture decomposition.
    Closure { closure_expr: syn::ExprClosure },
}

impl Parse for CudaLaunchArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Check for tagged arguments
        if input.peek(Ident) {
            let ident: Ident = input.fork().parse()?;
            match ident.to_string().as_str() {
                "slice" => {
                    input.parse::<Ident>()?;
                    let content;
                    parenthesized!(content in input);
                    let expr: syn::Expr = content.parse()?;
                    return Ok(CudaLaunchArg::SliceWithLen(expr));
                }
                "slice_mut" => {
                    input.parse::<Ident>()?;
                    let content;
                    parenthesized!(content in input);
                    let expr: syn::Expr = content.parse()?;
                    return Ok(CudaLaunchArg::SliceMutWithLen(expr));
                }
                // "move" keyword starts a move closure
                "move" => {
                    // Parse the full closure expression (move |args| body)
                    let expr: syn::Expr = input.parse()?;
                    if let Some(closure) = as_closure_expr(&expr) {
                        return Ok(CudaLaunchArg::Closure {
                            closure_expr: closure.clone(),
                        });
                    }
                    // Not a closure, treat as direct expression
                    return Ok(CudaLaunchArg::Direct(expr));
                }
                _ => {}
            }
        }

        // Check for closure starting with `|` (non-move closure)
        if input.peek(Token![|]) {
            let expr: syn::Expr = input.parse()?;
            if let Some(closure) = as_closure_expr(&expr) {
                return Ok(CudaLaunchArg::Closure {
                    closure_expr: closure.clone(),
                });
            }
            // Shouldn't happen, but fallback to direct
            return Ok(CudaLaunchArg::Direct(expr));
        }

        // Default: direct expression
        let expr: syn::Expr = input.parse()?;

        // Check if the parsed expression happens to be a closure
        if let Some(closure) = as_closure_expr(&expr) {
            return Ok(CudaLaunchArg::Closure {
                closure_expr: closure.clone(),
            });
        }

        Ok(CudaLaunchArg::Direct(expr))
    }
}

/// Input for cuda_launch! macro
struct CudaLaunchInput {
    /// Kernel path - can be simple name or path with generics: `scale` or `scale::<f32>`
    kernel: syn::Path,
    stream: syn::Expr,
    module: syn::Expr,
    config: syn::Expr,
    args: Vec<CudaLaunchArg>,
    /// Optional cluster dimensions (x, y, z) for thread block cluster launches.
    /// When present, uses `cuLaunchKernelEx` via `launch_cluster()` instead of `cuLaunchKernel`.
    cluster_dim: Option<syn::Expr>,
    /// Optional cooperative-launch flag. When `true`, the kernel is launched
    /// via `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 1`,
    /// which is required for `cuda_device::grid::sync()` to work.
    cooperative: Option<syn::Expr>,
}

impl CudaLaunchInput {
    /// Extract the base kernel name (without generics) and generic arguments
    fn kernel_parts(&self) -> (Ident, Option<&syn::PathArguments>) {
        let last_segment = self
            .kernel
            .segments
            .last()
            .expect("kernel path must have segments");
        let base_name = last_segment.ident.clone();
        let generics = match &last_segment.arguments {
            syn::PathArguments::None => None,
            args => Some(args),
        };
        (base_name, generics)
    }

    /// Check if this is a generic kernel (has type parameters)
    fn is_generic(&self) -> bool {
        self.kernel
            .segments
            .last()
            .map(|seg| !matches!(seg.arguments, syn::PathArguments::None))
            .unwrap_or(false)
    }
}

impl Parse for CudaLaunchInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut kernel = None;
        let mut stream = None;
        let mut module = None;
        let mut config = None;
        let mut args = Vec::new();
        let mut cluster_dim = None;
        let mut cooperative = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "kernel" => kernel = Some(input.parse()?),
                "stream" => stream = Some(input.parse()?),
                "module" => module = Some(input.parse()?),
                "config" => config = Some(input.parse()?),
                "cluster_dim" => cluster_dim = Some(input.parse()?),
                "cooperative" => cooperative = Some(input.parse()?),
                "args" => {
                    let content;
                    bracketed!(content in input);
                    if !content.is_empty() {
                        let parsed: Punctuated<CudaLaunchArg, Token![,]> =
                            Punctuated::parse_terminated(&content)?;
                        args = parsed.into_iter().collect();
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown field: {}. Expected: kernel, stream, module, config, cluster_dim, cooperative, args",
                            key
                        ),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        if cluster_dim.is_some() && cooperative.is_some() {
            return Err(syn::Error::new(
                input.span(),
                "cuda_launch!: `cluster_dim` and `cooperative` are mutually exclusive — \
                 cooperative cluster launches are not yet supported by this macro",
            ));
        }

        Ok(CudaLaunchInput {
            kernel: kernel.ok_or_else(|| syn::Error::new(input.span(), "missing 'kernel'"))?,
            stream: stream.ok_or_else(|| syn::Error::new(input.span(), "missing 'stream'"))?,
            module: module.ok_or_else(|| syn::Error::new(input.span(), "missing 'module'"))?,
            config: config.ok_or_else(|| syn::Error::new(input.span(), "missing 'config'"))?,
            args,
            cluster_dim,
            cooperative,
        })
    }
}

/// Launch a CUDA kernel synchronously on a given stream. **Unsafe**: the
/// expansion calls the unsafe `cuda_core` launch functions without wrapping
/// them, so every use must appear inside an `unsafe { }` block.
///
/// Uses the `CudaKernel` trait (generated by `#[kernel]`) to look up the PTX
/// entry point name. Arguments are marshaled into a `Vec<*mut c_void>` and
/// passed directly to `cuda_core::launch_kernel` (`cuLaunchKernel`).
///
/// # Safety
///
/// This macro cannot check the kernel's signature. It hands the driver a raw
/// array of argument pointers and trusts you completely. By wrapping the
/// macro in `unsafe { }`, the caller promises:
///
/// - the argument **count and order** match the kernel's actual parameter
///   list (with each `slice(..)` / `slice_mut(..)` counting as two
///   parameters: pointer then length);
/// - each argument's **type, size, and alignment** match the corresponding
///   kernel parameter;
/// - every pointer argument is **device-accessible** (a valid device
///   allocation, or host memory reachable via HMM/unified memory) and stays
///   alive until the kernel finishes.
///
/// A mismatch is undefined behavior, not a runtime error: too few or
/// mistyped arguments make the driver read past the end of the args array,
/// and a bad pointer makes the device dereference junk.
///
/// For kernels embedded in your own crate, prefer `#[cuda_module]`: it
/// reads the kernel signatures at compile time and generates typed launch
/// methods, so none of the above can go wrong. This macro's remaining niche
/// is modules loaded at **runtime by name** (e.g. external PTX files),
/// where no compile-time signature exists to check.
///
/// # Usage
///
/// ```ignore
/// // SAFETY: argument count, order, and types match `vecadd`'s signature;
/// // a_dev, b_dev, c_dev are live device buffers.
/// unsafe {
///     cuda_launch! {
///         kernel: vecadd,
///         stream: stream,
///         module: module,
///         config: LaunchConfig::for_num_elems(n as u32),
///         args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
///     }
/// }
/// ```
///
/// # Fields
///
/// | Field         | Type              | Description                                   |
/// |---------------|-------------------|-----------------------------------------------|
/// | `kernel`      | path              | `#[kernel]` function name (may be generic)    |
/// | `stream`      | `Arc<CudaStream>` | Stream to launch on                           |
/// | `module`      | `Arc<CudaModule>` | Loaded PTX module containing the kernel       |
/// | `config`      | `LaunchConfig`    | Grid/block dimensions, shared memory          |
/// | `cluster_dim` | `(u32,u32,u32)`   | *(optional)* Cluster dims for `cuLaunchKernelEx` |
/// | `cooperative` | `bool`            | *(optional)* Set `true` to launch via `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` (required for `grid::sync()`) |
/// | `args`        | `[arg, ...]`      | Kernel arguments (see below)                  |
///
/// `cluster_dim` and `cooperative` are mutually exclusive at this layer.
///
/// # Argument forms
///
/// - `expr` -- scalar or pointer passed directly
/// - `slice(buf)` -- immutable device buffer; pushes `(cu_deviceptr, len)` as two args
/// - `slice_mut(buf)` -- mutable device buffer; same as `slice` but borrows `&mut`
/// - `move |captures| body` -- closure whose captures are marshaled individually
/// - `|captures| body` -- non-move closure; captures passed as raw pointers (HMM)
///
/// # Returns
///
/// `Result<(), cuda_core::DriverError>` -- the launch is asynchronous, so
/// a successful return only means the launch was enqueued.  Call
/// `stream.synchronize()` to wait for completion.
#[proc_macro]
pub fn cuda_launch(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as CudaLaunchInput);

    let _kernel_path = &input.kernel;
    let stream = &input.stream;
    let module = &input.module;
    let config = &input.config;
    let cluster_dim = &input.cluster_dim;
    let cooperative = &input.cooperative;

    // Get base kernel name and generic arguments
    let (kernel_base, generics) = input.kernel_parts();

    let kernel_entry = format_ident!("{}{}", KERNEL_PREFIX, kernel_base);

    // Build the marker type name for CudaKernel lookup
    let marker_name = format_ident!("__{}_CudaKernel", kernel_base);

    // Check if any argument is a closure (for special handling)
    let has_closure = input
        .args
        .iter()
        .any(|arg| matches!(arg, CudaLaunchArg::Closure { .. }));

    // Extract closure info if present (for monomorphization). Only the
    // first closure is treated as the type-inference anchor; the macro
    // currently supports at most one closure parameter per kernel.
    let closure_info: Option<&syn::ExprClosure> = input.args.iter().find_map(|arg| {
        if let CudaLaunchArg::Closure { closure_expr } = arg {
            Some(closure_expr)
        } else {
            None
        }
    });

    // Generate argument marshaling code.
    //
    // Each argument becomes a stack-local variable whose address is pushed
    // into a `Vec<*mut c_void>`. This directly matches what cuLaunchKernel
    // expects: an array of pointers-to-argument-values. No trait dispatch
    // (PushKernelArg) or heap allocation per arg.
    let arg_code: Vec<TokenStream2> = input
        .args
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let val_name = format_ident!("__arg_{}", i);
            match arg {
                CudaLaunchArg::Direct(expr) => {
                    quote! {
                        let mut #val_name = #expr;
                        __args.push(&mut #val_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::SliceWithLen(expr) => {
                    let ptr_name = format_ident!("__arg_{}_ptr", i);
                    let len_name = format_ident!("__arg_{}_len", i);
                    quote! {
                        let #val_name = &#expr;
                        let mut #ptr_name = #val_name.cu_deviceptr();
                        let mut #len_name = #val_name.len() as u64;
                        __args.push(&mut #ptr_name as *mut _ as *mut std::ffi::c_void);
                        __args.push(&mut #len_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::SliceMutWithLen(expr) => {
                    let ptr_name = format_ident!("__arg_{}_ptr", i);
                    let len_name = format_ident!("__arg_{}_len", i);
                    quote! {
                        let #val_name = &mut #expr;
                        let mut #ptr_name = #val_name.cu_deviceptr();
                        let mut #len_name = #val_name.len() as u64;
                        __args.push(&mut #ptr_name as *mut _ as *mut std::ffi::c_void);
                        __args.push(&mut #len_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::Closure { .. } => {
                    // Push the whole closure as a single byval scalar. The
                    // backend emits a single byval kernel parameter for
                    // aggregate (struct / closure) entry-point args, so
                    // pushing `__closure` once matches what the device-side
                    // `.param` declaration expects.
                    //
                    // Routed through `push_kernel_scalar` so ZST closures
                    // (zero captures) are dropped from the host packet —
                    // matching the backend, which drops their `.param`
                    // declaration too. Move closures push by value;
                    // non-move closures push the closure struct (which
                    // contains host references the GPU dereferences via
                    // HMM).
                    let _ = i;
                    quote! {
                        ::cuda_host::push_kernel_scalar(&mut __args, &mut __closure);
                    }
                }
            }
        })
        .collect();

    // Build the instantiate helper name (for closures)
    let instantiate_name = format_ident!("{}{}", INSTANTIATE_PREFIX, kernel_base);

    // Generate the launch call — regular, cluster, or cooperative.
    //
    // All paths use the stream-aware cuda_core helpers. Those helpers bind the
    // stream's owning CUDA context to the calling thread and then delegate to
    // the raw cuLaunchKernel/cuLaunchKernelEx wrappers.
    let launch_call = if let Some(cdim) = cluster_dim {
        quote! {
            {
                let __cfg = #config;
                cuda_core::launch_kernel_ex_on_stream(
                    &__func,
                    __cfg.grid_dim,
                    __cfg.block_dim,
                    __cfg.shared_mem_bytes,
                    #cdim,
                    (#stream).as_ref(),
                    &mut __args,
                )
            }
        }
    } else if let Some(coop) = cooperative {
        quote! {
            {
                let __cfg = #config;
                let __cooperative: bool = #coop;
                if __cooperative {
                    cuda_core::launch_kernel_cooperative_on_stream(
                        &__func,
                        __cfg.grid_dim,
                        __cfg.block_dim,
                        __cfg.shared_mem_bytes,
                        (#stream).as_ref(),
                        &mut __args,
                    )
                } else {
                    cuda_core::launch_kernel_on_stream(
                        &__func,
                        __cfg.grid_dim,
                        __cfg.block_dim,
                        __cfg.shared_mem_bytes,
                        (#stream).as_ref(),
                        &mut __args,
                    )
                }
            }
        }
    } else {
        quote! {
            {
                let __cfg = #config;
                cuda_core::launch_kernel_on_stream(
                    &__func,
                    __cfg.grid_dim,
                    __cfg.block_dim,
                    __cfg.shared_mem_bytes,
                    (#stream).as_ref(),
                    &mut __args,
                )
            }
        }
    };

    let expanded = if has_closure {
        let closure_expr = closure_info.expect("has_closure but no closure_info");

        // The on-wire PTX name comes from the kernel's
        // GenericCudaKernel::ptx_name() impl (via the instantiate helper).
        // The helper takes `&F` so we can keep ownership of `__closure`
        // and push it as the byval kernel argument right after — the
        // backend's kernel-boundary ABI emits a single .param for the
        // whole closure struct, matching this single push.
        let _ = closure_expr.span();

        quote! {
            {
                let mut __closure = #closure_expr;
                let __ptx_name: &'static str = #instantiate_name(&__closure);
                let __func = #module.load_function(__ptx_name).unwrap_or_else(|err| {
                    panic!(
                        "Failed to load kernel `{}` (expected PTX entry `{}`): {:?}",
                        stringify!(#kernel_base),
                        __ptx_name,
                        err,
                    )
                });

                let mut __args: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                #launch_call
            }
        }
    } else if input.is_generic() {
        quote! {
            {
                let __kernel_ptr = #kernel_entry #generics as *const ();
                // Caller-unsafe on purpose: the volatile write/read pair that
                // forces monomorphization is covered by the same `unsafe { }`
                // block the caller must already supply for the launch itself.
                let mut __force_mono: *const () = core::ptr::null();
                core::ptr::write_volatile(&mut __force_mono, __kernel_ptr);
                let _ = core::ptr::read_volatile(&__force_mono);

                let __ptx_name = <#marker_name #generics as cuda_host::GenericCudaKernel>::ptx_name();
                let __func = #module.load_function(__ptx_name).unwrap_or_else(|err| {
                    panic!(
                        "Failed to load kernel `{}` (expected PTX entry `{}`): {:?}",
                        stringify!(#kernel_base),
                        __ptx_name,
                        err,
                    )
                });

                let mut __args: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                #launch_call
            }
        }
    } else {
        quote! {
            {
                const __PTX_NAME: &str = <#marker_name as cuda_host::CudaKernel>::PTX_NAME;
                let __func = #module.load_function(__PTX_NAME).unwrap_or_else(|err| {
                    panic!(
                        "Failed to load kernel `{}` (expected PTX entry `{}`): {:?}",
                        stringify!(#kernel_base),
                        __PTX_NAME,
                        err,
                    )
                });

                let mut __args: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                #launch_call
            }
        }
    };

    TokenStream::from(expanded)
}

// ============================================================================
// cuda_launch_async! Macro (async path via cuda-async)
// ============================================================================

/// Parsed input for the [`cuda_launch_async!`] macro.
///
/// Unlike [`CudaLaunchInput`], this struct has no `stream` field. The stream
/// is assigned later by the [`SchedulingPolicy`] when the returned
/// [`AsyncKernelLaunch`] is `.sync()`'d or `.await`'d.
struct CudaLaunchAsyncInput {
    /// Path to the `#[kernel]` function, possibly with generic arguments.
    kernel: syn::Path,
    /// Expression resolving to an `Arc<CudaModule>` that contains the compiled PTX.
    module: syn::Expr,
    /// Expression resolving to a [`LaunchConfig`] (grid/block dims, shared mem).
    config: syn::Expr,
    /// Kernel arguments: `slice(x)`, `slice_mut(x)`, direct values, or closures.
    args: Vec<CudaLaunchArg>,
}

impl CudaLaunchAsyncInput {
    /// Splits the kernel path into its base identifier and optional generic arguments.
    /// For `vecadd::<f32>` returns `("vecadd", Some(<f32>))`.
    fn kernel_parts(&self) -> (Ident, Option<&syn::PathArguments>) {
        let last_segment = self
            .kernel
            .segments
            .last()
            .expect("kernel path must have segments");
        let base_name = last_segment.ident.clone();
        let generics = match &last_segment.arguments {
            syn::PathArguments::None => None,
            args => Some(args),
        };
        (base_name, generics)
    }

    /// Returns `true` if the kernel path has explicit generic type arguments.
    fn is_generic(&self) -> bool {
        self.kernel
            .segments
            .last()
            .map(|seg| !matches!(seg.arguments, syn::PathArguments::None))
            .unwrap_or(false)
    }
}

/// Parses the `cuda_launch_async! { kernel: ..., module: ..., config: ..., args: [...] }` syntax.
///
/// Fields can appear in any order. The `args` field uses bracket syntax with the same
/// argument forms as `cuda_launch!`: `slice(x)`, `slice_mut(x)`, direct values, or closures.
impl Parse for CudaLaunchAsyncInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut kernel = None;
        let mut module = None;
        let mut config = None;
        let mut args = Vec::new();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "kernel" => kernel = Some(input.parse()?),
                "module" => module = Some(input.parse()?),
                "config" => config = Some(input.parse()?),
                "args" => {
                    let content;
                    bracketed!(content in input);
                    if !content.is_empty() {
                        let parsed: Punctuated<CudaLaunchArg, Token![,]> =
                            Punctuated::parse_terminated(&content)?;
                        args = parsed.into_iter().collect();
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown field: {}. Expected: kernel, module, config, args",
                            key
                        ),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        Ok(CudaLaunchAsyncInput {
            kernel: kernel.ok_or_else(|| syn::Error::new(input.span(), "missing 'kernel'"))?,
            module: module.ok_or_else(|| syn::Error::new(input.span(), "missing 'module'"))?,
            config: config.ok_or_else(|| syn::Error::new(input.span(), "missing 'config'"))?,
            args,
        })
    }
}

/// Launch a CUDA kernel asynchronously, returning a lazy `AsyncKernelLaunch`.
///
/// Unlike [`cuda_launch!`], this macro does **not** take a `stream:` parameter. The
/// CUDA stream is assigned later by the active `SchedulingPolicy` when the returned
/// operation is `.sync()`'d or `.await`'d. This enables lazy composition: multiple
/// launches can be chained with `.and_then()`, run in parallel with `zip!()`, or
/// awaited individually.
///
/// # Fields
///
/// | Field    | Type                | Description                                |
/// |----------|---------------------|--------------------------------------------|
/// | `kernel` | path                | `#[kernel]` function name (may be generic) |
/// | `module` | `Arc<CudaModule>`   | Loaded PTX module containing the kernel    |
/// | `config` | `LaunchConfig`      | Grid/block dimensions, shared memory       |
/// | `args`   | `[arg, ...]`        | Kernel arguments (see below)               |
///
/// # Argument forms
///
/// - `slice(x)` -- immutable device slice; pushes `(ptr, len)` as two kernel args
/// - `slice_mut(x)` -- mutable device slice; same as `slice` but takes `&mut`
/// - `expr` -- scalar or device pointer passed directly
/// - `|captures| body` -- closure environment passed by value
///
/// # Returns
///
/// An `AsyncKernelLaunch` implementing `DeviceOperation`. No GPU work is enqueued
/// until the caller schedules it.
///
/// # Usage
///
/// ```ignore
/// use cuda_host::cuda_launch_async;
/// use cuda_core::LaunchConfig;
///
/// let op = cuda_launch_async! {
///     kernel: vecadd,
///     module: module,
///     config: LaunchConfig::for_num_elems(N as u32),
///     args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
/// };
///
/// // Synchronous (blocks calling thread):
/// op.sync()?;
///
/// // Or asynchronous (suspends the async task):
/// // op.await?;
///
/// // Or compose before executing:
/// // let chained = op.and_then(|()| another_op);
/// // chained.sync()?;
/// ```
#[proc_macro]
pub fn cuda_launch_async(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as CudaLaunchAsyncInput);

    let module = &input.module;
    let config = &input.config;
    let (kernel_base, generics) = input.kernel_parts();
    let marker_name = format_ident!("__{}_CudaKernel", kernel_base);
    let instantiate_name = format_ident!("{}{}", INSTANTIATE_PREFIX, kernel_base);
    let has_closure = input
        .args
        .iter()
        .any(|arg| matches!(arg, CudaLaunchArg::Closure { .. }));
    let closure_expr = input.args.iter().find_map(|arg| {
        if let CudaLaunchArg::Closure { closure_expr } = arg {
            Some(closure_expr)
        } else {
            None
        }
    });

    let arg_code: Vec<TokenStream2> = input
        .args
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let tmp_name = format_ident!("__arg_{}", i);
            match arg {
                CudaLaunchArg::Direct(expr) => {
                    quote! {
                        __launch.push_arg(Box::new(#expr));
                    }
                }
                CudaLaunchArg::SliceWithLen(expr) => {
                    let len_name = format_ident!("__arg_{}_len", i);
                    quote! {
                        let #tmp_name = &#expr;
                        __launch.push_arg(Box::new(#tmp_name.cu_deviceptr()));
                        let #len_name = #tmp_name.len() as u64;
                        __launch.push_arg(Box::new(#len_name));
                    }
                }
                CudaLaunchArg::SliceMutWithLen(expr) => {
                    let len_name = format_ident!("__arg_{}_len", i);
                    quote! {
                        let #tmp_name = &mut #expr;
                        __launch.push_arg(Box::new(#tmp_name.cu_deviceptr()));
                        let #len_name = #tmp_name.len() as u64;
                        __launch.push_arg(Box::new(#len_name));
                    }
                }
                CudaLaunchArg::Closure { .. } => {
                    // Push the whole closure as one byval scalar so the
                    // host packet matches the single aggregate `.param`
                    // at the kernel boundary. ZST closures are omitted
                    // to keep later packet slots aligned.
                    quote! {
                        if ::core::mem::size_of_val(&__closure) != 0 {
                            __launch.push_scalar_arg(__closure);
                        }
                    }
                }
            }
        })
        .collect();

    let expanded = if has_closure {
        let closure_expr = closure_expr.expect("has_closure but no closure expression");
        quote! {
            {
                let __closure = #closure_expr;
                let __ptx_name: &'static str = #instantiate_name(&__closure);
                let __func = #module.load_function(__ptx_name).unwrap_or_else(|err| {
                    panic!(
                        "Failed to load kernel `{}` (expected PTX entry `{}`): {:?}",
                        stringify!(#kernel_base),
                        __ptx_name,
                        err,
                    )
                });
                let mut __launch = cuda_async::launch::AsyncKernelLaunch::new(
                    std::sync::Arc::new(__func),
                );
                #(#arg_code)*
                __launch.set_launch_config(#config);
                __launch
            }
        }
    } else if input.is_generic() {
        let kernel_entry = format_ident!("{}{}", KERNEL_PREFIX, kernel_base);
        quote! {
            {
                let __kernel_ptr = #kernel_entry #generics as *const ();
                unsafe {
                    let mut __force_mono: *const () = core::ptr::null();
                    core::ptr::write_volatile(&mut __force_mono, __kernel_ptr);
                    let _ = core::ptr::read_volatile(&__force_mono);
                }
                let __ptx_name = <#marker_name #generics as cuda_host::GenericCudaKernel>::ptx_name();
                let __func = #module.load_function(__ptx_name).unwrap_or_else(|err| {
                    panic!(
                        "Failed to load kernel `{}` (expected PTX entry `{}`): {:?}",
                        stringify!(#kernel_base),
                        __ptx_name,
                        err,
                    )
                });
                let mut __launch = cuda_async::launch::AsyncKernelLaunch::new(
                    std::sync::Arc::new(__func),
                );
                #(#arg_code)*
                __launch.set_launch_config(#config);
                __launch
            }
        }
    } else {
        quote! {
            {
                const __PTX_NAME: &str =
                    <#marker_name as cuda_host::CudaKernel>::PTX_NAME;
                let __func = #module.load_function(__PTX_NAME).unwrap_or_else(|err| {
                    panic!(
                        "Failed to load kernel `{}` (expected PTX entry `{}`): {:?}",
                        stringify!(#kernel_base),
                        __PTX_NAME,
                        err,
                    )
                });
                let mut __launch = cuda_async::launch::AsyncKernelLaunch::new(
                    std::sync::Arc::new(__func),
                );
                #(#arg_code)*
                __launch.set_launch_config(#config);
                __launch
            }
        }
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expands a `#[cuda_module]` body and returns the generated tokens as a
    /// whitespace-free string, so tests can assert on call paths without
    /// caring how `quote!` spaces out `::` separators.
    fn expand_to_compact_string(module: ItemMod) -> String {
        expand_cuda_module(module)
            .expect("cuda_module expansion failed")
            .to_string()
            .replace(' ', "")
    }

    #[test]
    fn cooperative_kernel_launches_through_cooperative_driver_call() {
        let module: ItemMod = parse_quote! {
            mod kernels {
                #[kernel]
                #[cooperative_launch]
                pub fn grid_sync_kernel(mut out: DisjointSlice<u32>) {}
            }
        };
        let expanded = expand_to_compact_string(module);

        // The sync launch method must route through the cooperative driver
        // entry point (cuLaunchKernelEx + CU_LAUNCH_ATTRIBUTE_COOPERATIVE)
        // instead of plain cuLaunchKernel.
        assert!(
            expanded.contains("launch_kernel_cooperative_on_stream"),
            "expected cooperative launch call in generated tokens:\n{expanded}"
        );
        assert!(
            !expanded.contains("launch_kernel_on_stream"),
            "plain launch call should be replaced by the cooperative one:\n{expanded}"
        );
    }

    #[test]
    fn plain_kernel_keeps_plain_driver_call() {
        let module: ItemMod = parse_quote! {
            mod kernels {
                #[kernel]
                pub fn plain_kernel(mut out: DisjointSlice<u32>) {}
            }
        };
        let expanded = expand_to_compact_string(module);

        assert!(
            expanded.contains("launch_kernel_on_stream"),
            "expected plain launch call in generated tokens:\n{expanded}"
        );
        assert!(
            !expanded.contains("launch_kernel_cooperative_on_stream"),
            "cooperative call must not appear without #[cooperative_launch]:\n{expanded}"
        );
    }

    #[test]
    fn cooperative_plus_cluster_kernel_uses_combined_driver_call() {
        let module: ItemMod = parse_quote! {
            mod kernels {
                #[kernel]
                #[cluster_launch(2, 1, 1)]
                #[cooperative_launch]
                pub fn clustered_grid_sync_kernel(mut out: DisjointSlice<u32>) {}
            }
        };
        let expanded = expand_to_compact_string(module);

        // cuLaunchKernelEx accepts both attributes in one attrs array, so the
        // combination is allowed and routes through the combined helper.
        assert!(
            expanded.contains("launch_kernel_ex_cooperative_on_stream"),
            "expected combined cluster+cooperative launch call:\n{expanded}"
        );
        assert!(
            !expanded.contains("launch_kernel_ex_on_stream"),
            "cluster-only call should be replaced by the combined one:\n{expanded}"
        );
    }

    #[cfg(feature = "async")]
    #[test]
    fn cooperative_kernel_sets_async_builder_knob() {
        let module: ItemMod = parse_quote! {
            mod kernels {
                #[kernel]
                #[cooperative_launch]
                pub fn grid_sync_kernel(mut out: DisjointSlice<u32>) {}
            }
        };
        let expanded = expand_to_compact_string(module);

        // Both the borrowed-async and owned-async builder methods set the
        // cooperative knob, exactly like set_async_kernel_cluster_dim is set
        // for #[cluster_launch].
        assert_eq!(
            expanded.matches("set_async_kernel_cooperative").count(),
            2,
            "expected the cooperative knob in both async builder methods:\n{expanded}"
        );
    }

    #[test]
    fn cooperative_launch_with_arguments_is_rejected() {
        let module: ItemMod = parse_quote! {
            mod kernels {
                #[kernel]
                #[cooperative_launch(4)]
                pub fn grid_sync_kernel(mut out: DisjointSlice<u32>) {}
            }
        };
        let error = expand_cuda_module(module).expect_err("expected expansion to fail");
        assert!(
            error
                .to_string()
                .contains("cooperative_launch takes no arguments"),
            "unexpected error message: {error}"
        );
    }

    /// Runs the per-loop `#[unroll]` visitor over `func`'s body and returns the
    /// rewritten function as a whitespace-free string, panicking on any
    /// recorded parse error.
    fn run_loop_unroll_visitor(mut func: ItemFn) -> String {
        rewrite_loop_unroll_attrs(&mut func)
            .unwrap_or_else(|err| panic!("unexpected unroll-attr error: {err}"));
        quote!(#func).to_string().replace(' ', "")
    }

    #[test]
    fn bare_unroll_on_while_injects_factor_zero_marker() {
        let func: ItemFn = parse_quote! {
            fn k() {
                let mut i = 0u32;
                #[unroll]
                while i < 4 { i += 1; }
            }
        };
        let out = run_loop_unroll_visitor(func);
        // Bare #[unroll] => full unroll (factor 0).
        assert!(
            out.contains("cuda_device::thread::__unroll_config::<0u32>()"),
            "expected factor-0 marker:\n{out}"
        );
        // The expression attribute must be stripped so rustc never sees it.
        assert!(
            !out.contains("#[unroll]"),
            "the #[unroll] attribute should be removed:\n{out}"
        );
    }

    #[test]
    fn unroll_n_on_for_loop_injects_factor_n_marker() {
        let func: ItemFn = parse_quote! {
            fn k() {
                #[unroll(4)]
                for i in 0..8 { let _ = i; }
            }
        };
        let out = run_loop_unroll_visitor(func);
        assert!(
            out.contains("cuda_device::thread::__unroll_config::<4u32>()"),
            "expected factor-4 marker:\n{out}"
        );
        assert!(
            !out.contains("#[unroll"),
            "attribute should be removed:\n{out}"
        );
    }

    #[test]
    fn unroll_on_bare_loop_injects_marker() {
        let func: ItemFn = parse_quote! {
            fn k() {
                #[unroll(2)]
                loop { break; }
            }
        };
        let out = run_loop_unroll_visitor(func);
        assert!(
            out.contains("cuda_device::thread::__unroll_config::<2u32>()"),
            "expected factor-2 marker on `loop`:\n{out}"
        );
    }

    #[test]
    fn nested_annotated_loop_is_handled() {
        let func: ItemFn = parse_quote! {
            fn k() {
                for outer in 0..2 {
                    if outer == 0 {
                        #[unroll(3)]
                        for inner in 0..4 { let _ = inner; }
                    }
                }
            }
        };
        let out = run_loop_unroll_visitor(func);
        assert!(
            out.contains("cuda_device::thread::__unroll_config::<3u32>()"),
            "expected marker injected into nested loop:\n{out}"
        );
    }

    #[test]
    fn loop_without_unroll_attr_is_unchanged() {
        // A function whose loops carry no #[unroll] must be byte-identical
        // before and after the visitor runs (no marker, no edits).
        let src: ItemFn = parse_quote! {
            fn k() {
                let mut i = 0u32;
                while i < 4 { i += 1; }
                for j in 0..2 { let _ = j; }
            }
        };
        let before = quote!(#src).to_string();
        let after = run_loop_unroll_visitor(src);
        assert_eq!(
            before.replace(' ', ""),
            after,
            "loops without #[unroll] must not be modified"
        );
        assert!(
            !after.contains("__unroll_config"),
            "no marker should be injected without #[unroll]:\n{after}"
        );
    }

    #[test]
    fn malformed_unroll_attr_records_error() {
        // `#[unroll(1, 2)]` has too many args; the visitor must record an error.
        let mut func: ItemFn = parse_quote! {
            fn k() {
                #[unroll(1, 2)]
                for i in 0..4 { let _ = i; }
            }
        };
        let mut visitor = LoopUnrollAttrVisitor::default();
        visitor.visit_block_mut(&mut func.block);
        assert!(
            visitor.error.is_some(),
            "malformed #[unroll(1, 2)] should record a parse error"
        );
    }

    #[test]
    fn partial_unroll_factor_must_be_at_least_two() {
        assert!(syn::parse_str::<UnrollArgs>("0").is_err());
        assert!(syn::parse_str::<UnrollArgs>("1").is_err());
        assert_eq!(syn::parse_str::<UnrollArgs>("2").unwrap().factor, 2);
    }
}
