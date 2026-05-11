/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread Block Cluster operations (sm_90+ Hopper).
//!
//! This module provides operations for Hopper's Thread Block Cluster features:
//!
//! ```text
//! ┌──────────────────────────────┬───────────────────────┬────────────────────────────────┐
//! │ Operation                    │ PTX Register/Instr    │ Description                    │
//! ├──────────────────────────────┼───────────────────────┼────────────────────────────────┤
//! │ ReadPtxSregClusterCtaidXOp   │ %cluster_ctaid.x      │ Block's X pos within cluster   │
//! │ ReadPtxSregClusterCtaidYOp   │ %cluster_ctaid.y      │ Block's Y pos within cluster   │
//! │ ReadPtxSregClusterCtaidZOp   │ %cluster_ctaid.z      │ Block's Z pos within cluster   │
//! │ ReadPtxSregClusterNctaidXOp  │ %cluster_nctaid.x     │ Cluster X dimension            │
//! │ ReadPtxSregClusterNctaidYOp  │ %cluster_nctaid.y     │ Cluster Y dimension            │
//! │ ReadPtxSregClusterNctaidZOp  │ %cluster_nctaid.z     │ Cluster Z dimension            │
//! │ ReadPtxSregClusterIdxOp      │ %cluster_idx          │ Cluster's linear index in grid │
//! │ ReadPtxSregNclusterIdOp      │ %nclusterid           │ Total clusters in grid         │
//! │ ClusterSyncOp                │ cluster.sync.aligned  │ Cluster-wide barrier           │
//! │ MapaSharedClusterOp          │ mapa.shared::cluster  │ Distributed shared mem map     │
//! └──────────────────────────────┴───────────────────────┴────────────────────────────────┘
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
// Block Position Within Cluster (cluster_ctaid)
// =============================================================================

/// Read the X component of the block's position within cluster.
///
/// Corresponds to PTX `%cluster_ctaid.x`.
///
/// Returns a value in range `[0, cluster_nctaid.x)`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_cluster_ctaid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClusterCtaidXOp;

impl ReadPtxSregClusterCtaidXOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregClusterCtaidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_cluster_ctaid_x result must be integer"
                );
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_cluster_ctaid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Y component of the block's position within cluster.
///
/// Corresponds to PTX `%cluster_ctaid.y`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_cluster_ctaid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClusterCtaidYOp;

impl ReadPtxSregClusterCtaidYOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregClusterCtaidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_cluster_ctaid_y result must be integer"
                );
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_cluster_ctaid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Z component of the block's position within cluster.
///
/// Corresponds to PTX `%cluster_ctaid.z`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_cluster_ctaid_z",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClusterCtaidZOp;

impl ReadPtxSregClusterCtaidZOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregClusterCtaidZOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_cluster_ctaid_z result must be integer"
                );
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_cluster_ctaid_z result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Cluster Dimensions (cluster_nctaid)
// =============================================================================

/// Read the X dimension of the cluster (blocks per cluster in X).
///
/// Corresponds to PTX `%cluster_nctaid.x`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_cluster_nctaid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClusterNctaidXOp;

impl ReadPtxSregClusterNctaidXOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregClusterNctaidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_cluster_nctaid_x result must be integer"
                );
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_cluster_nctaid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Y dimension of the cluster (blocks per cluster in Y).
///
/// Corresponds to PTX `%cluster_nctaid.y`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_cluster_nctaid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClusterNctaidYOp;

impl ReadPtxSregClusterNctaidYOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregClusterNctaidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_cluster_nctaid_y result must be integer"
                );
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_cluster_nctaid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Z dimension of the cluster (blocks per cluster in Z).
///
/// Corresponds to PTX `%cluster_nctaid.z`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_cluster_nctaid_z",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClusterNctaidZOp;

impl ReadPtxSregClusterNctaidZOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregClusterNctaidZOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_cluster_nctaid_z result must be integer"
                );
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_cluster_nctaid_z result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Cluster Grid Position
// =============================================================================

/// Read the cluster's linear index within the grid.
///
/// Corresponds to PTX `%cluster_idx`.
///
/// Returns a value in range `[0, nclusterid)`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_cluster_idx",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClusterIdxOp;

impl ReadPtxSregClusterIdxOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregClusterIdxOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_cluster_idx result must be integer"
                );
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_cluster_idx result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the total number of clusters in the grid.
///
/// Corresponds to PTX `%nclusterid`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_nclusterid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNclusterIdOp;

impl ReadPtxSregNclusterIdOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregNclusterIdOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_nclusterid result must be integer"
                );
            }
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_nclusterid result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Cluster Synchronization
// =============================================================================

/// Cluster-wide barrier synchronization.
///
/// All threads in all blocks of the cluster must reach this barrier before any can proceed.
/// Corresponds to PTX `cluster.sync.aligned`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 0 results
#[pliron_op(
    name = "nvvm.cluster_sync",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ClusterSyncOp;

impl ClusterSyncOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

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
        Self { op }
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
        Self { op }
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
    // Block position within cluster
    ReadPtxSregClusterCtaidXOp::register(ctx);
    ReadPtxSregClusterCtaidYOp::register(ctx);
    ReadPtxSregClusterCtaidZOp::register(ctx);
    // Cluster dimensions
    ReadPtxSregClusterNctaidXOp::register(ctx);
    ReadPtxSregClusterNctaidYOp::register(ctx);
    ReadPtxSregClusterNctaidZOp::register(ctx);
    // Cluster grid position
    ReadPtxSregClusterIdxOp::register(ctx);
    ReadPtxSregNclusterIdOp::register(ctx);
    // Synchronization
    ClusterSyncOp::register(ctx);
    // Distributed shared memory
    MapaSharedClusterOp::register(ctx);
    DsmemReadU32Op::register(ctx);
}
