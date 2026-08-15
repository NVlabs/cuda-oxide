/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attributes carried by structured PTX operations.

use pliron::attribute::Attribute;
use pliron::context::Context;
use pliron::derive::pliron_attr;

/// The two callable forms defined by PTX.
#[pliron_attr(name = "ptx.callable_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallableKindAttr {
    Entry,
    Function,
}

/// The control-flow role of a native PTX block terminator.
#[pliron_attr(name = "ptx.terminator_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TerminatorKindAttr {
    Fallthrough,
    Branch,
    IndexedBranch,
    Return,
    ThreadExit,
    Trap,
}

pub fn register(ctx: &mut Context) {
    CallableKindAttr::register(ctx);
    TerminatorKindAttr::register(ctx);
}
