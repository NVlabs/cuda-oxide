/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Expr, Ident, LitStr, Token, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

const MAX_PTX_ASM_INPUTS: usize = 16;
const REGISTER_ONLY_OPTION: &str = "register_only";

pub struct PtxAsmInput {
    template: LitStr,
    operands: Vec<PtxAsmOperand>,
}

enum PtxAsmOperand {
    Out { constraint: LitStr, place: Expr },
    In { constraint: LitStr, expr: Expr },
    Clobber { name: LitStr },
    Options { options: Vec<Ident> },
}

#[derive(Default)]
struct PtxAsmOptions {
    register_only: bool,
}

impl Parse for PtxAsmInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let template: LitStr = input.parse()?;
        let mut operands = Vec::new();

        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }

            if input.peek(Token![in]) {
                input.parse::<Token![in]>()?;
                let constraint = parse_parenthesized_string(input)?;
                let expr: Expr = input.parse()?;
                operands.push(PtxAsmOperand::In { constraint, expr });
                continue;
            }

            let ident: syn::Ident = input.parse()?;
            match ident.to_string().as_str() {
                "out" => {
                    let constraint = parse_parenthesized_string(input)?;
                    let place: Expr = input.parse()?;
                    operands.push(PtxAsmOperand::Out { constraint, place });
                }
                "clobber" => {
                    let name = parse_parenthesized_string(input)?;
                    operands.push(PtxAsmOperand::Clobber { name });
                }
                "options" => {
                    let options = parse_parenthesized_options(input)?;
                    operands.push(PtxAsmOperand::Options { options });
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "unsupported ptx_asm! operand `{other}`; expected `out`, `in`, `clobber`, or `options`"
                        ),
                    ));
                }
            }
        }

        Ok(Self { template, operands })
    }
}

fn parse_parenthesized_string(input: ParseStream) -> syn::Result<LitStr> {
    let content;
    parenthesized!(content in input);
    let lit: LitStr = content.parse()?;
    if !content.is_empty() {
        return Err(syn::Error::new(
            content.span(),
            "expected a single string literal in parentheses",
        ));
    }
    Ok(lit)
}

fn parse_parenthesized_options(input: ParseStream) -> syn::Result<Vec<Ident>> {
    let content;
    parenthesized!(content in input);
    let options: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&content)?;
    if options.is_empty() {
        return Err(syn::Error::new(
            content.span(),
            "options(...) requires at least one option",
        ));
    }
    Ok(options.into_iter().collect())
}

pub fn ptx_asm_impl(input: PtxAsmInput) -> TokenStream2 {
    match build_ptx_asm(input) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

fn build_ptx_asm(input: PtxAsmInput) -> syn::Result<TokenStream2> {
    let mut output: Option<(LitStr, Expr)> = None;
    let mut inputs: Vec<(LitStr, Expr)> = Vec::new();
    let mut clobbers: Vec<LitStr> = Vec::new();
    let mut options = PtxAsmOptions::default();
    let mut saw_input = false;

    for operand in input.operands {
        match operand {
            PtxAsmOperand::Out { constraint, place } => {
                if saw_input {
                    return Err(syn::Error::new(
                        constraint.span(),
                        "`out` operands must appear before `in` operands",
                    ));
                }
                if output.is_some() {
                    return Err(syn::Error::new(
                        constraint.span(),
                        "ptx_asm! currently supports at most one `out` operand",
                    ));
                }
                validate_output_constraint(&constraint)?;
                output = Some((constraint, place));
            }
            PtxAsmOperand::In { constraint, expr } => {
                validate_input_constraint(&constraint)?;
                saw_input = true;
                inputs.push((constraint, expr));
            }
            PtxAsmOperand::Clobber { name } => clobbers.push(name),
            PtxAsmOperand::Options {
                options: option_idents,
            } => {
                for option in option_idents {
                    match option.to_string().as_str() {
                        REGISTER_ONLY_OPTION => {
                            if options.register_only {
                                return Err(syn::Error::new(
                                    option.span(),
                                    "`options(register_only)` was specified more than once",
                                ));
                            }
                            options.register_only = true;
                        }
                        other => {
                            return Err(syn::Error::new(
                                option.span(),
                                format!(
                                    "unsupported ptx_asm! option `{other}`; expected `register_only`"
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }

    if options.register_only && output.is_none() {
        return Err(syn::Error::new(
            input.template.span(),
            "`options(register_only)` requires an `out` operand",
        ));
    }
    if options.register_only && !clobbers.is_empty() {
        return Err(syn::Error::new(
            clobbers[0].span(),
            "`options(register_only)` cannot be used with clobbers",
        ));
    }

    if inputs.len() > MAX_PTX_ASM_INPUTS {
        return Err(syn::Error::new(
            input.template.span(),
            format!("ptx_asm! supports at most {MAX_PTX_ASM_INPUTS} input operands"),
        ));
    }

    let operand_count = usize::from(output.is_some()) + inputs.len();
    let converted_template = convert_cuda_template(&input.template, operand_count)?;
    let template_lit = syn::LitByteStr::new(converted_template.as_bytes(), input.template.span());

    let mut constraints = Vec::new();
    if let Some((constraint, _)) = &output {
        constraints.push(constraint.value());
    }
    constraints.extend(inputs.iter().map(|(constraint, _)| constraint.value()));
    for clobber in &clobbers {
        constraints.push(normalize_clobber(clobber)?);
    }
    let constraints = constraints.join(",");
    let constraints_lit = syn::LitByteStr::new(constraints.as_bytes(), input.template.span());
    let options_marker = if options.register_only {
        REGISTER_ONLY_OPTION
    } else {
        ""
    };
    let options_lit = syn::LitByteStr::new(options_marker.as_bytes(), input.template.span());

    let input_exprs: Vec<&Expr> = inputs.iter().map(|(_, expr)| expr).collect();
    let arity = input_exprs.len();

    if let Some((_, place)) = output {
        let fn_ident = format_ident!("__ptx_asm_out_{arity}");
        Ok(quote! {{
            #place = cuda_device::ptx::#fn_ident(
                #template_lit,
                #constraints_lit,
                #options_lit,
                #(#input_exprs),*
            );
        }})
    } else {
        let fn_ident = format_ident!("__ptx_asm_void_{arity}");
        Ok(quote! {{
            cuda_device::ptx::#fn_ident(
                #template_lit,
                #constraints_lit,
                #options_lit,
                #(#input_exprs),*
            );
        }})
    }
}

fn convert_cuda_template(template: &LitStr, operand_count: usize) -> syn::Result<String> {
    let value = template.value();
    let mut converted = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '%' {
            converted.push(ch);
            continue;
        }

        match chars.peek().copied() {
            Some('%') => {
                chars.next();
                converted.push('%');
            }
            Some(next) if next.is_ascii_digit() => {
                let mut digits = String::new();
                converted.push('$');
                while let Some(digit) = chars.peek().copied() {
                    if digit.is_ascii_digit() {
                        chars.next();
                        digits.push(digit);
                        converted.push(digit);
                    } else {
                        break;
                    }
                }
                let index = digits.parse::<usize>().map_err(|_| {
                    syn::Error::new(
                        template.span(),
                        format!("ptx_asm! template placeholder `%{digits}` is too large"),
                    )
                })?;
                if index >= operand_count {
                    return Err(syn::Error::new(
                        template.span(),
                        format!(
                            "ptx_asm! template placeholder `%{digits}` has no matching operand"
                        ),
                    ));
                }
            }
            Some(other) => {
                let mut literal = String::new();
                for ch in chars.clone() {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.') {
                        literal.push(ch);
                    } else {
                        break;
                    }
                }
                if literal.is_empty() {
                    literal.push(other);
                }
                return Err(syn::Error::new(
                    template.span(),
                    format!(
                        "literal PTX register `%{literal}` must be escaped as `%%{literal}` in ptx_asm!"
                    ),
                ));
            }
            None => {
                return Err(syn::Error::new(
                    template.span(),
                    "trailing `%` in ptx_asm! template",
                ));
            }
        }
    }

    Ok(converted)
}

fn validate_single_constraint(constraint: &LitStr) -> syn::Result<String> {
    let value = constraint.value();
    if value.contains(',') {
        return Err(syn::Error::new(
            constraint.span(),
            "ptx_asm! operand constraints cannot contain `,`; use separate operands and clobber(...)",
        ));
    }
    Ok(value)
}

fn validate_output_constraint(constraint: &LitStr) -> syn::Result<()> {
    let value = validate_single_constraint(constraint)?;
    if !value.starts_with('=') {
        return Err(syn::Error::new(
            constraint.span(),
            "`out` constraints must use output syntax such as `\"=r\"`",
        ));
    }
    Ok(())
}

fn validate_input_constraint(constraint: &LitStr) -> syn::Result<()> {
    let value = validate_single_constraint(constraint)?;
    if value.starts_with('=') || value.starts_with('+') {
        return Err(syn::Error::new(
            constraint.span(),
            "`in` constraints must use input syntax such as `\"r\"`",
        ));
    }
    Ok(())
}

fn normalize_clobber(clobber: &LitStr) -> syn::Result<String> {
    let value = clobber.value();
    if value == "memory" {
        Ok("~{memory}".to_string())
    } else if value.starts_with("~{") && value.ends_with('}') {
        Ok(value)
    } else {
        Err(syn::Error::new(
            clobber.span(),
            "only `clobber(\"memory\")` or raw LLVM clobbers like `clobber(\"~{memory}\")` are supported",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn converts_cuda_placeholders_and_escaped_registers() {
        let template: LitStr = parse_quote!("mov.u32 %0, %%laneid; add.u32 %10, %1, %2;");

        assert_eq!(
            convert_cuda_template(&template, 11).unwrap(),
            "mov.u32 $0, %laneid; add.u32 $10, $1, $2;"
        );
    }

    #[test]
    fn rejects_unescaped_literal_registers() {
        let template: LitStr = parse_quote!("mov.u32 %0, %laneid;");
        let err = convert_cuda_template(&template, 1).unwrap_err();

        assert!(
            err.to_string().contains("must be escaped as `%%laneid`"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_placeholders_without_operands() {
        let template: LitStr = parse_quote!("add.u32 %2, %0, %1;");
        let err = convert_cuda_template(&template, 2).unwrap_err();

        assert!(
            err.to_string()
                .contains("placeholder `%2` has no matching operand"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn normalizes_memory_clobber() {
        let clobber: LitStr = parse_quote!("memory");
        assert_eq!(normalize_clobber(&clobber).unwrap(), "~{memory}");
    }
}
