/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for the induction-variable analysis: given a counted loop, does it
//! classify each carried value (counter vs reduction), read off the counter's
//! `init`/`step`, recover the continue-predicate, and compute the trip count?

mod common;

use common::{counted_loop, counted_loop_from, mir_ctx, multi_latch_counted_loop};
use mir_transforms::analyses::induction::{ArgKind, CmpPred, analyze};
use mir_transforms::analyses::loop_info::LoopInfo;
use pliron::graph::dominance::DomInfo;

/// Run the analysis on `while i < n { acc += i; i += 1 }` and return its facts.
fn recurrences_for(n: i64) -> mir_transforms::analyses::induction::LoopRecurrences {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, n);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    let ph = info.preheader(&ctx, lp.region, id).unwrap();
    analyze(&ctx, &info, id, ph)
}

#[test]
fn analyzes_counted_loop_recurrence() {
    // while i < 8 { acc += i; i += 1 }  =>  header args are (acc, i).
    let rec = recurrences_for(8);

    // `i` (header arg 1) is the counter: starts at 0, steps by 1.
    assert_eq!(rec.primary_iv, Some(1));
    match rec.args[1] {
        ArgKind::BasicIv { init, step } => {
            assert_eq!(init, 0);
            assert_eq!(step, 1);
        }
        ref other => panic!("i should be a BasicIv, got {other:?}"),
    }

    // The loop continues while `i < 8`, so 8 iterations, bound 8.
    assert_eq!(rec.continue_pred, Some(CmpPred::Lt));
    assert_eq!(rec.bound, Some(8));
    assert_eq!(rec.trip_count, Some(8));

    // `acc` (header arg 0) is carried but updated by `acc + i`, so it is a
    // reduction, not the counter.
    assert!(
        matches!(rec.args[0], ArgKind::Reduction),
        "acc should be a reduction, got {:?}",
        rec.args[0]
    );
}

/// The trip count tracks the bound: `while i < n` runs `n` times (init 0, step 1).
#[test]
fn trip_count_tracks_the_bound() {
    for n in [1, 4, 16, 100] {
        let rec = recurrences_for(n);
        assert_eq!(rec.bound, Some(n as i128), "bound for n={n}");
        assert_eq!(rec.trip_count, Some(n as u64), "trip count for n={n}");
    }
}

/// Unsigned constants must be zero-extended. The high bit of both values is set,
/// but this is still a four-trip loop.
#[test]
fn high_bit_unsigned_constants_keep_their_positive_values() {
    let mut ctx = mir_ctx();
    let start = 2_147_483_646i64;
    let bound = 2_147_483_650i64;
    let lp = counted_loop_from(&mut ctx, start, bound);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    let ph = info.preheader(&ctx, lp.region, id).unwrap();
    let rec = analyze(&ctx, &info, id, ph);

    assert!(matches!(
        rec.args[1],
        ArgKind::BasicIv {
            init: 2_147_483_646,
            step: 1
        }
    ));
    assert_eq!(rec.bound, Some(2_147_483_650));
    assert_eq!(rec.trip_count, Some(4));
}

#[test]
fn analyzes_matching_recurrence_on_every_latch() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 1);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    assert_eq!(info.loops()[id].latches.len(), 2);
    let ph = info.preheader(&ctx, lp.region, id).unwrap();
    let rec = analyze(&ctx, &info, id, ph);

    assert_eq!(rec.primary_iv, Some(1));
    match rec.args[1] {
        ArgKind::BasicIv { init, step } => {
            assert_eq!(init, 0);
            assert_eq!(step, 1);
        }
        ref other => panic!("i should be a BasicIv on every latch, got {other:?}"),
    }
    assert!(
        matches!(rec.args[0], ArgKind::Reduction),
        "acc differs between continue and normal paths, so it is a reduction"
    );
    assert_eq!(rec.continue_pred, Some(CmpPred::Lt));
    assert_eq!(rec.bound, Some(4));
    assert_eq!(rec.trip_count, Some(4));
}

#[test]
fn rejects_inconsistent_iv_steps_across_latches() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 2);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    assert_eq!(info.loops()[id].latches.len(), 2);
    let ph = info.preheader(&ctx, lp.region, id).unwrap();
    let rec = analyze(&ctx, &info, id, ph);

    assert_eq!(rec.primary_iv, None, "there is no single affine counter");
    assert!(
        !matches!(rec.args[1], ArgKind::BasicIv { .. }),
        "different latch steps must not be guessed from an arbitrary latch"
    );
    assert_eq!(rec.trip_count, None);
}
