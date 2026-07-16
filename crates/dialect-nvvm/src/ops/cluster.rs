/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread Block Cluster operations (sm_90+ Hopper).
//!
//! This module provides operations for Hopper's Thread Block Cluster features:
//!
//! ```text
//! ┌─────────────────────┬───────────────────────────┬────────────────────────────┐
//! │ Operation           │ PTX instruction           │ Description                │
//! ├─────────────────────┼───────────────────────────┼────────────────────────────┤
//! │ MapaSharedClusterOp │ mapa.shared::cluster      │ Distributed memory mapping │
//! │ DsmemReadU32Op      │ ld.shared::cluster.u32    │ Distributed memory read    │
//! └─────────────────────┴───────────────────────────┴────────────────────────────┘
//! ```
//!
//! # Cluster Hierarchy
//!
//! ```text
//! Grid
//! └── Cluster (cluster_idx: 0..nclusterid)
//!     └── Block (cluster_ctaid: 0..cluster_nctaid per dimension)
//!         └── Thread
//! ```
//!
//! # Hardware Requirements
//!
//! All operations in this module require **sm_90+** (Hopper architecture).

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::IntegerType,
    common_traits::Verify,
    context::Context,
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

// =============================================================================
// Distributed Shared Memory
// =============================================================================

/// Map shared memory address to another block's address space within the cluster.
///
/// Corresponds to PTX `mapa.shared::cluster.u32` or `mapa.shared::cluster.u64`.
///
/// # Operands
///
/// 1. `ptr` (pointer): Source shared memory address
/// 2. `rank` (i32): Target block's rank within cluster (0 to cluster_size - 1)
///
/// # Results
///
/// 1. Mapped pointer that can access target block's shared memory
///
/// # Verification
///
/// - Must have 2 operands (ptr, rank)
/// - Must have 1 result (mapped ptr)
#[pliron_op(
    name = "nvvm.mapa_shared_cluster",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MapaSharedClusterOp;

impl MapaSharedClusterOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        MapaSharedClusterOp { op }
    }
}

/// Combined mapa + ld.shared::cluster.u32 for reading another block's shared memory.
///
/// This combines address mapping and load into a single operation because
/// `mapa.shared::cluster` returns a shared-space address that requires
/// `ld.shared::cluster` to read — a generic load (`ld.b32`) cannot access it.
///
/// Corresponds to PTX:
/// ```ptx
/// mapa.shared::cluster.u64 %rd_tmp, %rd_src, %r_rank;
/// ld.shared::cluster.u32 %r_result, [%rd_tmp];
/// ```
///
/// # Operands
///
/// 1. `ptr` (pointer): Source shared memory address (local CTA)
/// 2. `rank` (i32): Target block's rank within cluster
///
/// # Results
///
/// 1. `u32` value read from the target block's shared memory
#[pliron_op(
    name = "nvvm.dsmem_read_u32",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct DsmemReadU32Op;

impl DsmemReadU32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        DsmemReadU32Op { op }
    }
}

impl Verify for DsmemReadU32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.dsmem_read_u32 result must be integer");
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.dsmem_read_u32 result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Register cluster operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // Distributed shared memory
    MapaSharedClusterOp::register(ctx);
    DsmemReadU32Op::register(ctx);
}
