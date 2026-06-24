/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! General natural-loop detection over a CFG region (the LLVM `LoopInfo`
//! analysis, ported to pliron's CFG + dominator infrastructure).
//!
//! This is a **reusable analysis**, not specific to unrolling: it identifies
//! every natural loop in a function, the loop-nesting forest, and per loop the
//! header, latch(es), body block set, preheader, exiting blocks, and exit
//! blocks. Future passes (LICM, strength reduction, induction-variable
//! simplification) consume it.
//!
//! Definitions follow the standard dominator-based formulation:
//!   * A CFG edge `latch -> header` is a **back-edge** when `header` dominates
//!     `latch`.
//!   * The **natural loop** of a back-edge is `header` plus every block that can
//!     reach `latch` without passing through `header`.
//!   * Back-edges sharing a header belong to one loop (multiple latches).
//!
//! For reducible CFGs (which Rust MIR + mem2reg produce) two natural loops are
//! either nested or disjoint, so the nesting forest is well defined.

use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::graph::ControlFlowGraph;
use pliron::graph::dominance::DomTree;
use pliron::region::Region;
use rustc_hash::{FxHashMap, FxHashSet};

/// Index of a [Loop] within a [LoopInfo].
pub type LoopId = usize;

/// One natural loop.
#[derive(Debug, Clone)]
pub struct Loop {
    /// The loop header: the single entry block, dominates every body block.
    pub header: Ptr<BasicBlock>,
    /// Blocks with a back-edge to `header` (>1 when the loop has several
    /// continue-sites; the canonical counted loop has exactly one).
    pub latches: Vec<Ptr<BasicBlock>>,
    /// Every block in the loop, including `header` and the `latches`.
    pub blocks: FxHashSet<Ptr<BasicBlock>>,
    /// Immediately-enclosing loop, if any.
    pub parent: Option<LoopId>,
    /// Immediately-nested loops.
    pub children: Vec<LoopId>,
}

/// All natural loops of a region, plus the nesting forest and a
/// block-to-innermost-loop map.
#[derive(Debug, Default)]
pub struct LoopInfo {
    loops: Vec<Loop>,
    top_level: Vec<LoopId>,
    innermost: FxHashMap<Ptr<BasicBlock>, LoopId>,
}

impl LoopInfo {
    /// Compute the loop forest of `region` given its dominator tree.
    pub fn compute(
        ctx: &Context,
        region: Ptr<Region>,
        dom: &DomTree<Ptr<Region>, Context>,
    ) -> Self {
        // 1. Find back-edges (latch -> header where header dominates latch) and
        //    group them by header: one loop per distinct header.
        let mut latches_by_header: FxHashMap<Ptr<BasicBlock>, Vec<Ptr<BasicBlock>>> =
            FxHashMap::default();
        let all_blocks: Vec<Ptr<BasicBlock>> = region.nodes(ctx).collect();
        for &block in &all_blocks {
            for succ in region.successors(ctx, &block) {
                if dom.dominates(&succ, &block) {
                    latches_by_header.entry(succ).or_default().push(block);
                }
            }
        }

        // 2. For each header, the natural-loop body = header + everything that
        //    reaches a latch without passing through the header (backward walk).
        let mut loops: Vec<Loop> = Vec::with_capacity(latches_by_header.len());
        for (header, latches) in latches_by_header {
            let mut blocks: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
            blocks.insert(header);
            let mut worklist: Vec<Ptr<BasicBlock>> = Vec::new();
            for &latch in &latches {
                if blocks.insert(latch) {
                    worklist.push(latch);
                }
            }
            while let Some(n) = worklist.pop() {
                for pred in region.predecessors(ctx, &n) {
                    if blocks.insert(pred) {
                        worklist.push(pred);
                    }
                }
            }
            loops.push(Loop {
                header,
                latches,
                blocks,
                parent: None,
                children: Vec::new(),
            });
        }

        // 3. Nesting: loop indices ordered by body size (innermost first).
        let mut by_size: Vec<LoopId> = (0..loops.len()).collect();
        by_size.sort_by_key(|&i| loops[i].blocks.len());

        // parent(L) = the smallest strictly-larger loop whose body contains L's
        // header. (Equal-sized loops can't nest.)
        for (pos, &li) in by_size.iter().enumerate() {
            let header = loops[li].header;
            let li_size = loops[li].blocks.len();
            for &mi in by_size.iter().skip(pos + 1) {
                if loops[mi].blocks.len() > li_size && loops[mi].blocks.contains(&header) {
                    loops[li].parent = Some(mi);
                    break;
                }
            }
        }
        let mut top_level = Vec::new();
        for li in 0..loops.len() {
            match loops[li].parent {
                Some(p) => loops[p].children.push(li),
                None => top_level.push(li),
            }
        }

        // 4. block -> innermost (smallest) containing loop.
        let mut innermost: FxHashMap<Ptr<BasicBlock>, LoopId> = FxHashMap::default();
        for &li in &by_size {
            for &b in &loops[li].blocks {
                innermost.entry(b).or_insert(li);
            }
        }

        LoopInfo {
            loops,
            top_level,
            innermost,
        }
    }

    /// All loops (any nesting depth).
    pub fn loops(&self) -> &[Loop] {
        &self.loops
    }

    /// Outermost loops.
    pub fn top_level(&self) -> &[LoopId] {
        &self.top_level
    }

    /// The innermost loop containing `block`, if any.
    pub fn innermost_loop(&self, block: Ptr<BasicBlock>) -> Option<LoopId> {
        self.innermost.get(&block).copied()
    }

    /// `true` if there are no loops.
    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }

    /// The loop's preheader: its header's unique predecessor outside the loop.
    /// `None` when the header has zero or several outside predecessors (the
    /// caller may then need to create one before transforming).
    pub fn preheader(
        &self,
        ctx: &Context,
        region: Ptr<Region>,
        id: LoopId,
    ) -> Option<Ptr<BasicBlock>> {
        let l = &self.loops[id];
        let mut outside = region
            .predecessors(ctx, &l.header)
            .into_iter()
            .filter(|p| !l.blocks.contains(p));
        let first = outside.next()?;
        if outside.next().is_none() {
            Some(first)
        } else {
            None
        }
    }

    /// Blocks inside the loop with at least one successor outside it.
    pub fn exiting_blocks(
        &self,
        ctx: &Context,
        region: Ptr<Region>,
        id: LoopId,
    ) -> Vec<Ptr<BasicBlock>> {
        let l = &self.loops[id];
        l.blocks
            .iter()
            .copied()
            .filter(|&b| {
                region
                    .successors(ctx, &b)
                    .iter()
                    .any(|s| !l.blocks.contains(s))
            })
            .collect()
    }

    /// Blocks outside the loop that are branched to from inside it.
    pub fn exit_blocks(
        &self,
        ctx: &Context,
        region: Ptr<Region>,
        id: LoopId,
    ) -> Vec<Ptr<BasicBlock>> {
        let l = &self.loops[id];
        let mut seen: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
        let mut out = Vec::new();
        for &b in &l.blocks {
            for s in region.successors(ctx, &b) {
                if !l.blocks.contains(&s) && seen.insert(s) {
                    out.push(s);
                }
            }
        }
        out
    }
}
