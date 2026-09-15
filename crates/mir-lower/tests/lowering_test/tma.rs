/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::{MirPtrType, address_space};
use dialect_nvvm::ops as nvvm;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;

use crate::common::{append_return, build_test_kernel, make_test_ctx};

fn g2s_module(destination_address_space: u32) -> (Context, Ptr<Operation>) {
    let mut ctx = make_test_ctx();
    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i16_ty = IntegerType::get(&ctx, 16, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let destination = MirPtrType::get(&mut ctx, u8_ty.into(), true, destination_address_space);
    let barrier = MirPtrType::get_shared(&mut ctx, u8_ty.into(), true);
    let descriptor = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let (module, entry) = build_test_kernel(
        &mut ctx,
        vec![
            destination.into(),
            barrier.into(),
            descriptor.into(),
            i32_ty.into(),
            i32_ty.into(),
            i16_ty.into(),
            i64_ty.into(),
        ],
    );
    let operands = (0..7)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();
    Operation::new(
        &mut ctx,
        nvvm::CpAsyncBulkTensorG2sTile2dOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    )
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);
    (ctx, module)
}

fn lower_g2s(
    destination_address_space: u32,
    target: &str,
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, Ptr<Operation>), String> {
    let (mut ctx, module) = g2s_module(destination_address_space);
    let result = mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            target_arch: Some(target.parse().unwrap()),
            ..Default::default()
        },
    );
    match result {
        Ok(()) => Ok((ctx, module)),
        Err(error) => Err(error.disp(&ctx).to_string()),
    }
}

fn exported_module(ctx: &Context, module: Ptr<Operation>) -> String {
    let module = Operation::get_op::<ModuleOp>(module, ctx).unwrap();
    llvm_export::export::export_module_to_string(ctx, &module).unwrap()
}

#[test]
fn sm120_local_g2s_uses_cta_inline_ptx_on_both_backends() {
    for target in [
        "sm_120", "sm_120a", "sm_120f", "sm_121", "sm_121a", "sm_121f",
    ] {
        for backend in [
            mir_lower::IntrinsicBackend::LlvmNvptx,
            mir_lower::IntrinsicBackend::LibNvvm,
        ] {
            let (ctx, module) = lower_g2s(address_space::SHARED, target, backend).unwrap();
            let ir = exported_module(&ctx, module);
            assert!(
                ir.contains(
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ),
                "target {target}: {ir}"
            );
            assert!(
                ir.contains("\"r,r,l,r,r,~{memory}\""),
                "target {target}: {ir}"
            );
            assert!(!ir.contains("shared::cluster"), "target {target}: {ir}");
            assert!(
                !ir.contains("llvm.nvvm.cp.async.bulk.tensor.g2s"),
                "target {target}: {ir}"
            );
        }
    }
}

#[test]
fn sm90_local_g2s_keeps_cluster_intrinsic_lowering() {
    let (ctx, module) = lower_g2s(
        address_space::SHARED,
        "sm_90a",
        mir_lower::IntrinsicBackend::LlvmNvptx,
    )
    .unwrap();
    let ir = exported_module(&ctx, module);
    assert!(
        ir.contains("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d"),
        "{ir}"
    );
    assert!(
        ir.contains("addrspacecast ptr addrspace(3)") && ir.contains("to ptr addrspace(7)"),
        "{ir}"
    );
    assert!(!ir.contains("shared::cta"), "{ir}");
}

#[test]
fn sm120_rejects_cluster_shared_destination_for_unicast_g2s() {
    let result = lower_g2s(
        address_space::CLUSTER_SHARED,
        "sm_120a",
        mir_lower::IntrinsicBackend::LlvmNvptx,
    );
    let error = match result {
        Ok(_) => panic!("SM120 unicast G2S accepted an AS7 destination"),
        Err(error) => error,
    };
    assert!(
        error.contains("cannot target cluster-shared address space 7"),
        "{error}"
    );
}
