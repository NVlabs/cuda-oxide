/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for the `LoopInfo` analysis: given a `dialect-mir` region, does it
//! find the natural loops, their header/latch/blocks, and the surrounding
//! preheader / exit / exiting blocks?

mod common;

use common::{block, counted_loop, empty_func, mir_ctx, ret};
use mir_transforms::analyses::loop_info::LoopInfo;
use pliron::graph::dominance::DomInfo;

/// A region with no back-edge has no loops.
#[test]
fn finds_no_loop_in_a_straight_line_function() {
    let mut ctx = mir_ctx();
    let (_module, region) = empty_func(&mut ctx);
    let entry = block(&mut ctx, region, vec![]);
    ret(&mut ctx, entry);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, region);
        LoopInfo::compute(&ctx, region, dt)
    };
    assert!(info.is_empty(), "a straight-line function has no loops");
    assert_eq!(info.loops().len(), 0);
}

/// The canonical counted loop: one natural loop, header + latch, with the
/// preheader / exit / exiting blocks we built around it.
#[test]
fn finds_a_single_counted_loop() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 8);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };

    assert_eq!(info.loops().len(), 1, "exactly one natural loop");
    let id = info
        .innermost_loop(lp.header)
        .expect("the header is in a loop");
    let l = &info.loops()[id];

    // header + latch, flat (no parent, no children).
    assert_eq!(l.header, lp.header);
    assert_eq!(l.latches, vec![lp.latch]);
    assert_eq!(l.blocks.len(), 2);
    assert!(l.blocks.contains(&lp.header) && l.blocks.contains(&lp.latch));
    assert!(l.parent.is_none() && l.children.is_empty());

    // It is a top-level loop.
    assert_eq!(info.top_level(), &[id]);

    // preheader / exit / exiting-block are the ones we built.
    assert_eq!(info.preheader(&ctx, lp.region, id), Some(lp.preheader));
    assert_eq!(info.exit_blocks(&ctx, lp.region, id), vec![lp.exit]);
    assert_eq!(info.exiting_blocks(&ctx, lp.region, id), vec![lp.header]);

    // Body blocks map to this loop; the preheader (outside) does not.
    assert_eq!(info.innermost_loop(lp.latch), Some(id));
    assert_eq!(info.innermost_loop(lp.preheader), None);
}
