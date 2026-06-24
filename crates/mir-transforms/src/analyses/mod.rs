/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reusable analyses over `dialect-mir` (read-only; they compute facts, they do
//! not mutate IR). Transforms such as the loop unroller consume them, and future
//! loop passes (LICM, strength reduction) will too.
//!
//! Naming follows pliron's convention (`analyses/liveness.rs`,
//! `graph/dominance.rs`): files are named by the concept, with no `-analysis`
//! suffix; the `analyses/` directory marks them as analyses.

pub mod induction;
pub mod loop_info;
