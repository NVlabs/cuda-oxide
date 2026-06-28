/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{
    CudaModuleKernel, CudaModuleParamMarshal, attr_path_ends_with, collect_cuda_module_kernels,
    expand_cuda_module_with_extra, has_attr_named,
};
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, format_ident, quote};
use std::collections::{BTreeMap, BTreeSet};
use syn::visit_mut::{self, VisitMut};
use syn::{
    Expr, ExprMethodCall, FnArg, Ident, Item, ItemFn, ItemMod, Pat, Stmt, Type, parse_quote,
};

struct CudaProgram {
    module: ItemMod,
    graphs: Vec<CudaProgramGraph>,
}

struct CudaProgramGraph {
    vis: syn::Visibility,
    name: Ident,
    graph_name: Ident,
    struct_name: Ident,
    params: Vec<CudaProgramParam>,
    steps: Vec<CudaProgramStep>,
    resources: Vec<CudaProgramResource>,
    dependencies: Vec<CudaProgramDependency>,
    needs_lifetime: bool,
}

struct CudaProgramParam {
    name: Ident,
    ty: Type,
}

struct CudaProgramStep {
    kernel: Ident,
    args: Vec<CudaProgramStepArg>,
    config: Expr,
    unsafe_kernel: bool,
}

struct CudaProgramStepArg {
    expr: Expr,
    resource: Option<String>,
    role: CudaProgramArgumentRole,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaProgramArgumentRole {
    Read,
    Write,
    Scalar,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaProgramResourceRole {
    Input,
    Output,
    Scratch,
    Scalar,
}

struct CudaProgramResource {
    name: String,
    type_name: String,
    role: CudaProgramResourceRole,
}

struct CudaProgramDependency {
    from: usize,
    to: usize,
    resource: String,
}

pub(super) fn expand_cuda_program(module: ItemMod) -> syn::Result<TokenStream2> {
    let CudaProgram { module, graphs } = parse_cuda_program(module)?;
    let extra_items = generate_cuda_program_items(&graphs);
    expand_cuda_module_with_extra(module, extra_items)
}

fn parse_cuda_program(mut module: ItemMod) -> syn::Result<CudaProgram> {
    let Some((_brace, items)) = &mut module.content else {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "cuda_program requires an inline module so kernels and programs are visible",
        ));
    };

    let mut module_items = Vec::with_capacity(items.len());
    let mut program_fns = Vec::new();
    for item in items.drain(..) {
        match item {
            Item::Fn(mut item_fn) if is_cuda_program_graph_fn(&item_fn) => {
                item_fn.attrs.retain(|attr| {
                    !attr_path_ends_with(attr, "program") && !attr_path_ends_with(attr, "pipeline")
                });
                program_fns.push(item_fn);
            }
            item => module_items.push(item),
        }
    }
    *items = module_items;

    let kernels = collect_cuda_module_kernels(items)?;
    let graphs = program_fns
        .iter()
        .map(|item_fn| parse_cuda_program_graph(item_fn, &kernels))
        .collect::<syn::Result<Vec<_>>>()?;
    if graphs.is_empty() {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "cuda_program found no #[program] or #[pipeline] functions in this module",
        ));
    }

    Ok(CudaProgram { module, graphs })
}

fn is_cuda_program_graph_fn(item_fn: &ItemFn) -> bool {
    has_attr_named(&item_fn.attrs, "program") || has_attr_named(&item_fn.attrs, "pipeline")
}

fn parse_cuda_program_graph(
    item_fn: &ItemFn,
    kernels: &[CudaModuleKernel],
) -> syn::Result<CudaProgramGraph> {
    if item_fn.sig.unsafety.is_some() {
        return Err(syn::Error::new_spanned(
            item_fn.sig.unsafety,
            "cuda_program graph declarations cannot be unsafe; unsafe kernels remain marked on the kernel functions",
        ));
    }
    if item_fn.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            item_fn.sig.asyncness,
            "cuda_program graph declarations cannot be async",
        ));
    }
    if !matches!(item_fn.sig.output, syn::ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.output,
            "cuda_program graph declarations do not support return types yet",
        ));
    }
    if !item_fn.sig.generics.params.is_empty() || item_fn.sig.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.generics,
            "cuda_program graph declarations do not support generics yet",
        ));
    }

    let mut params = Vec::new();
    let mut needs_lifetime = false;
    for arg in &item_fn.sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "cuda_program graph declarations cannot take self parameters",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "cuda_program graph declarations only support simple identifier parameters",
            ));
        };
        needs_lifetime |= type_contains_reference(&pat_type.ty);
        params.push(CudaProgramParam {
            name: pat_ident.ident.clone(),
            ty: (*pat_type.ty).clone(),
        });
    }

    let steps = item_fn
        .block
        .stmts
        .iter()
        .map(|stmt| parse_cuda_program_step(stmt, kernels))
        .collect::<syn::Result<Vec<_>>>()?;
    if steps.is_empty() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.ident,
            "cuda_program graph declarations must contain at least one kernel operation",
        ));
    }

    let name = item_fn.sig.ident.clone();
    let graph_name = format_ident!("{}_graph", name);
    let struct_name = format_ident!("{}Graph", pascal_case_ident(&name));
    let resources = infer_cuda_program_resources(&params, &steps);
    let dependencies = infer_cuda_program_dependencies(&steps);

    Ok(CudaProgramGraph {
        vis: item_fn.vis.clone(),
        name,
        graph_name,
        struct_name,
        params,
        steps,
        resources,
        dependencies,
        needs_lifetime,
    })
}

fn parse_cuda_program_step(
    stmt: &Stmt,
    kernels: &[CudaModuleKernel],
) -> syn::Result<CudaProgramStep> {
    let Stmt::Expr(expr, _) = stmt else {
        return Err(syn::Error::new_spanned(
            stmt,
            "cuda_program graph bodies only support kernel operation statements",
        ));
    };
    let Expr::MethodCall(method_call) = expr else {
        return Err(syn::Error::new_spanned(
            expr,
            "cuda_program operation must end with `.grid_len(n)` or `.launch_config(config)`",
        ));
    };

    let config = cuda_program_step_config(method_call)?;
    let Expr::Call(call) = &*method_call.receiver else {
        return Err(syn::Error::new_spanned(
            &method_call.receiver,
            "cuda_program operation receiver must be a kernel call",
        ));
    };
    let Expr::Path(path) = &*call.func else {
        return Err(syn::Error::new_spanned(
            &call.func,
            "cuda_program operation must call a kernel by simple name",
        ));
    };
    let Some(kernel) = path.path.get_ident().cloned() else {
        return Err(syn::Error::new_spanned(
            path,
            "cuda_program operation must call a kernel by simple name",
        ));
    };
    let Some(kernel_info) = kernels.iter().find(|candidate| candidate.fn_name == kernel) else {
        return Err(syn::Error::new_spanned(
            &kernel,
            format!("cuda_program operation `{kernel}` does not match a #[kernel] in this module"),
        ));
    };
    if call.args.len() != kernel_info.params.len() {
        return Err(syn::Error::new_spanned(
            call,
            format!(
                "cuda_program operation `{kernel}` passes {} arguments, but the kernel takes {}",
                call.args.len(),
                kernel_info.params.len()
            ),
        ));
    }
    let args = call
        .args
        .iter()
        .zip(kernel_info.params.iter())
        .map(|(expr, param)| CudaProgramStepArg {
            expr: expr.clone(),
            resource: cuda_program_resource_name(expr),
            role: cuda_program_argument_role(&param.marshal),
        })
        .collect();

    Ok(CudaProgramStep {
        kernel,
        args,
        config,
        unsafe_kernel: kernel_info.unsafety.is_some(),
    })
}

fn cuda_program_argument_role(marshal: &CudaModuleParamMarshal) -> CudaProgramArgumentRole {
    match marshal {
        CudaModuleParamMarshal::Scalar => CudaProgramArgumentRole::Scalar,
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => CudaProgramArgumentRole::Read,
        CudaModuleParamMarshal::WritableDeviceBuffer { .. } => CudaProgramArgumentRole::Write,
    }
}

fn cuda_program_resource_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(path) => path.path.get_ident().map(ToString::to_string),
        Expr::Reference(reference) => cuda_program_resource_name(&reference.expr),
        _ => None,
    }
}

fn infer_cuda_program_resources(
    params: &[CudaProgramParam],
    steps: &[CudaProgramStep],
) -> Vec<CudaProgramResource> {
    let mut reads = BTreeSet::new();
    let mut writes = BTreeSet::new();
    let mut scalars = BTreeSet::new();

    for step in steps {
        for arg in &step.args {
            let Some(resource) = &arg.resource else {
                continue;
            };
            match arg.role {
                CudaProgramArgumentRole::Read => {
                    reads.insert(resource.clone());
                }
                CudaProgramArgumentRole::Write => {
                    writes.insert(resource.clone());
                }
                CudaProgramArgumentRole::Scalar => {
                    scalars.insert(resource.clone());
                }
            }
        }
    }

    params
        .iter()
        .map(|param| {
            let name = param.name.to_string();
            let role = if reads.contains(&name) && writes.contains(&name) {
                CudaProgramResourceRole::Scratch
            } else if writes.contains(&name) {
                CudaProgramResourceRole::Output
            } else if reads.contains(&name) {
                CudaProgramResourceRole::Input
            } else if scalars.contains(&name) || !type_contains_reference(&param.ty) {
                CudaProgramResourceRole::Scalar
            } else {
                CudaProgramResourceRole::Input
            };
            CudaProgramResource {
                name,
                type_name: param.ty.to_token_stream().to_string(),
                role,
            }
        })
        .collect()
}

fn infer_cuda_program_dependencies(steps: &[CudaProgramStep]) -> Vec<CudaProgramDependency> {
    let mut last_writer: BTreeMap<String, usize> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut dependencies = Vec::new();

    for (to, step) in steps.iter().enumerate() {
        for arg in &step.args {
            let Some(resource) = &arg.resource else {
                continue;
            };
            match arg.role {
                CudaProgramArgumentRole::Read => {
                    if let Some(&from) = last_writer.get(resource)
                        && seen.insert((from, to, resource.clone()))
                    {
                        dependencies.push(CudaProgramDependency {
                            from,
                            to,
                            resource: resource.clone(),
                        });
                    }
                }
                CudaProgramArgumentRole::Write => {
                    if let Some(&from) = last_writer.get(resource)
                        && seen.insert((from, to, resource.clone()))
                    {
                        dependencies.push(CudaProgramDependency {
                            from,
                            to,
                            resource: resource.clone(),
                        });
                    }
                    last_writer.insert(resource.clone(), to);
                }
                CudaProgramArgumentRole::Scalar => {}
            }
        }
    }

    dependencies
}

fn cuda_program_step_config(method_call: &ExprMethodCall) -> syn::Result<Expr> {
    let method = method_call.method.to_string();
    if method == "grid_len" {
        if method_call.args.len() != 1 {
            return Err(syn::Error::new_spanned(
                &method_call.args,
                "grid_len takes exactly one element-count argument",
            ));
        }
        let n = method_call.args.first().expect("checked len");
        return Ok(parse_quote! {
            ::cuda_core::LaunchConfig::for_num_elems(
                ::core::convert::TryInto::<u32>::try_into(#n)
                    .expect("cuda_program grid_len does not fit in u32")
            )
        });
    }
    if method == "launch_config" || method == "config" {
        if method_call.args.len() != 1 {
            return Err(syn::Error::new_spanned(
                &method_call.args,
                "launch_config takes exactly one LaunchConfig argument",
            ));
        }
        let config = method_call.args.first().expect("checked len");
        return Ok(parse_quote! { #config });
    }

    Err(syn::Error::new_spanned(
        &method_call.method,
        "unsupported cuda_program operation method; use `.grid_len(n)` or `.launch_config(config)`",
    ))
}

fn generate_cuda_program_items(graphs: &[CudaProgramGraph]) -> TokenStream2 {
    let graph_items = graphs.iter().map(generate_cuda_program_graph);
    quote! {
        #(#graph_items)*
    }
}

fn generate_cuda_program_graph(graph: &CudaProgramGraph) -> TokenStream2 {
    let CudaProgramGraph {
        vis,
        name,
        graph_name,
        struct_name,
        params,
        steps,
        resources,
        dependencies,
        needs_lifetime,
    } = graph;
    let lifetime = syn::Lifetime::new("'__cuda_program", proc_macro2::Span::call_site());
    let lifetime_def = needs_lifetime
        .then(|| quote! { <#lifetime> })
        .unwrap_or_default();
    let lifetime_use = needs_lifetime
        .then(|| quote! { <#lifetime> })
        .unwrap_or_default();
    let plan_lifetime = if *needs_lifetime {
        quote! { #lifetime }
    } else {
        quote! { 'static }
    };
    let field_defs = params.iter().map(|param| {
        let param_name = &param.name;
        let ty = if *needs_lifetime {
            type_with_named_lifetime(&param.ty, &lifetime)
        } else {
            param.ty.clone()
        };
        quote! { #param_name: #ty }
    });
    let builder_params = params.iter().map(|param| {
        let param_name = &param.name;
        let ty = if *needs_lifetime {
            type_with_named_lifetime(&param.ty, &lifetime)
        } else {
            param.ty.clone()
        };
        quote! { #param_name: #ty }
    });
    let field_names = params.iter().map(|param| &param.name);
    let destructure_names = params.iter().map(|param| &param.name);
    let step_launches = steps.iter().map(generate_cuda_program_step_launch);
    let operations = steps.iter().map(|step| step.kernel.to_string());
    let upper_name = name.to_string().to_uppercase();
    let operations_const = format_ident!("__{upper_name}_PROGRAM_OPERATIONS");
    let resource_const = format_ident!("__{upper_name}_PROGRAM_RESOURCES");
    let operation_const = format_ident!("__{upper_name}_PROGRAM_OPERATION_NODES");
    let dependency_const = format_ident!("__{upper_name}_PROGRAM_DEPENDENCIES");
    let argument_consts = steps
        .iter()
        .enumerate()
        .map(|(index, step)| generate_cuda_program_argument_const(name, index, step))
        .collect::<Vec<_>>();
    let argument_const_idents = (0..steps.len())
        .map(|index| format_ident!("__{upper_name}_PROGRAM_OP_{index}_ARGS"))
        .collect::<Vec<_>>();
    let resource_entries = resources
        .iter()
        .map(generate_cuda_program_resource_metadata);
    let operation_entries = steps
        .iter()
        .enumerate()
        .zip(argument_const_idents.iter())
        .map(|((index, step), arg_const)| {
            let operation = step.kernel.to_string();
            quote! {
                ::cuda_host::ProgramOperationMetadata {
                    index: #index,
                    name: #operation,
                    arguments: #arg_const,
                }
            }
        });
    let dependency_entries = dependencies
        .iter()
        .map(generate_cuda_program_dependency_metadata);

    quote! {
        const #operations_const: &[&str] = &[#(#operations),*];
        #(#argument_consts)*
        const #resource_const: &[::cuda_host::ProgramResourceMetadata] = &[
            #(#resource_entries),*
        ];
        const #operation_const: &[::cuda_host::ProgramOperationMetadata] = &[
            #(#operation_entries),*
        ];
        const #dependency_const: &[::cuda_host::ProgramDependencyMetadata] = &[
            #(#dependency_entries),*
        ];

        #vis struct #struct_name #lifetime_def {
            #(#field_defs,)*
        }

        #vis fn #graph_name #lifetime_def(
            #(#builder_params),*
        ) -> #struct_name #lifetime_use {
            #struct_name {
                #(#field_names),*
            }
        }

        impl #lifetime_def #struct_name #lifetime_use {
            pub const METADATA: ::cuda_host::ProgramGraphMetadata =
                ::cuda_host::ProgramGraphMetadata {
                    name: stringify!(#name),
                    operations: #operations_const,
                    resources: #resource_const,
                    operation_nodes: #operation_const,
                    dependencies: #dependency_const,
                };

            pub fn metadata(&self) -> ::cuda_host::ProgramGraphMetadata {
                Self::METADATA
            }

            pub fn resources(&self) -> &'static [::cuda_host::ProgramResourceMetadata] {
                Self::METADATA.resources
            }

            pub fn operations(&self) -> &'static [::cuda_host::ProgramOperationMetadata] {
                Self::METADATA.operation_nodes
            }

            pub fn dependencies(&self) -> &'static [::cuda_host::ProgramDependencyMetadata] {
                Self::METADATA.dependencies
            }

            pub fn bind(
                self,
                module: &LoadedModule,
                lowering: ::cuda_host::ProgramLowering,
            ) -> ::core::result::Result<::cuda_host::BoundProgram<#plan_lifetime>, ::cuda_core::DriverError> {
                let #struct_name { #(#destructure_names),* } = self;
                let __module = module.clone();
                Ok(::cuda_host::BoundProgram::new(Self::METADATA, lowering, move |stream| {
                    match lowering {
                        ::cuda_host::ProgramLowering::SequentialLaunches => {
                            #(#step_launches)*
                            Ok(())
                        }
                    }
                }))
            }

            pub fn lower(
                self,
                module: &LoadedModule,
                lowering: ::cuda_host::ProgramLowering,
            ) -> ::core::result::Result<::cuda_host::BoundProgram<#plan_lifetime>, ::cuda_core::DriverError> {
                self.bind(module, lowering)
            }
        }
    }
}

fn generate_cuda_program_step_launch(step: &CudaProgramStep) -> TokenStream2 {
    let kernel = &step.kernel;
    let config = &step.config;
    let args = step.args.iter().map(|arg| &arg.expr);
    if step.unsafe_kernel {
        quote! {
            unsafe {
                __module.#kernel(stream, #config, #(#args),*)?;
            }
        }
    } else {
        quote! {
            __module.#kernel(stream, #config, #(#args),*)?;
        }
    }
}

fn generate_cuda_program_argument_const(
    graph_name: &Ident,
    index: usize,
    step: &CudaProgramStep,
) -> TokenStream2 {
    let arg_const = format_ident!(
        "__{}_PROGRAM_OP_{index}_ARGS",
        graph_name.to_string().to_uppercase()
    );
    let args = step
        .args
        .iter()
        .map(generate_cuda_program_argument_metadata);
    quote! {
        const #arg_const: &[::cuda_host::ProgramArgumentMetadata] = &[
            #(#args),*
        ];
    }
}

fn generate_cuda_program_argument_metadata(arg: &CudaProgramStepArg) -> TokenStream2 {
    let expression = arg.expr.to_token_stream().to_string();
    let resource = match &arg.resource {
        Some(resource) => quote! { ::core::option::Option::Some(#resource) },
        None => quote! { ::core::option::Option::None },
    };
    let role = cuda_program_argument_role_tokens(arg.role);
    quote! {
        ::cuda_host::ProgramArgumentMetadata {
            expression: #expression,
            resource: #resource,
            role: #role,
        }
    }
}

fn generate_cuda_program_resource_metadata(resource: &CudaProgramResource) -> TokenStream2 {
    let name = &resource.name;
    let type_name = &resource.type_name;
    let role = cuda_program_resource_role_tokens(resource.role);
    quote! {
        ::cuda_host::ProgramResourceMetadata {
            name: #name,
            type_name: #type_name,
            role: #role,
        }
    }
}

fn generate_cuda_program_dependency_metadata(dependency: &CudaProgramDependency) -> TokenStream2 {
    let from = dependency.from;
    let to = dependency.to;
    let resource = &dependency.resource;
    quote! {
        ::cuda_host::ProgramDependencyMetadata {
            from: #from,
            to: #to,
            resource: #resource,
        }
    }
}

fn cuda_program_argument_role_tokens(role: CudaProgramArgumentRole) -> TokenStream2 {
    match role {
        CudaProgramArgumentRole::Read => quote! { ::cuda_host::ProgramArgumentRole::Read },
        CudaProgramArgumentRole::Write => quote! { ::cuda_host::ProgramArgumentRole::Write },
        CudaProgramArgumentRole::Scalar => quote! { ::cuda_host::ProgramArgumentRole::Scalar },
    }
}

fn cuda_program_resource_role_tokens(role: CudaProgramResourceRole) -> TokenStream2 {
    match role {
        CudaProgramResourceRole::Input => quote! { ::cuda_host::ProgramResourceRole::Input },
        CudaProgramResourceRole::Output => quote! { ::cuda_host::ProgramResourceRole::Output },
        CudaProgramResourceRole::Scratch => quote! { ::cuda_host::ProgramResourceRole::Scratch },
        CudaProgramResourceRole::Scalar => quote! { ::cuda_host::ProgramResourceRole::Scalar },
    }
}

fn pascal_case_ident(ident: &Ident) -> Ident {
    let mut out = String::new();
    for part in ident.to_string().split('_').filter(|part| !part.is_empty()) {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    if out.is_empty() {
        out.push_str("Program");
    }
    format_ident!("{}", out)
}

fn type_contains_reference(ty: &Type) -> bool {
    struct ContainsReference(bool);

    impl<'ast> syn::visit::Visit<'ast> for ContainsReference {
        fn visit_type_reference(&mut self, _node: &'ast syn::TypeReference) {
            self.0 = true;
        }
    }

    let mut visitor = ContainsReference(false);
    syn::visit::Visit::visit_type(&mut visitor, ty);
    visitor.0
}

fn type_with_named_lifetime(ty: &Type, lifetime: &syn::Lifetime) -> Type {
    struct RewriteElidedLifetime<'a> {
        lifetime: &'a syn::Lifetime,
    }

    impl VisitMut for RewriteElidedLifetime<'_> {
        fn visit_type_reference_mut(&mut self, node: &mut syn::TypeReference) {
            if node.lifetime.is_none() {
                node.lifetime = Some(self.lifetime.clone());
            }
            visit_mut::visit_type_reference_mut(self, node);
        }
    }

    let mut ty = ty.clone();
    RewriteElidedLifetime { lifetime }.visit_type_mut(&mut ty);
    ty
}
