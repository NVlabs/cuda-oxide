/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conservative rustc-MIR provenance analysis for bounded loops produced from
//! small fixed arrays and standard iterator adapters.
//!
//! The analysis is enabled with:
//!
//! ```text
//! CUDA_OXIDE_MIR_OPTS=small-array-iterators
//! ```
//!
//! A previous version recorded only the block containing a literal
//! `Iterator::next` call. That is insufficient because rustc may inline the
//! iterator chain before public MIR reaches the importer. This implementation
//! follows the array-derived iterator state, computes natural loops from the
//! reachable CFG, and records a loop header only when the loop exit condition
//! is derived from that state. Unknown calls receiving the tracked state still
//! invalidate the root array.

use rustc_hash::{FxHashMap, FxHashSet};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::{RigidTy, Ty, TyConstKind, TyKind};
use std::collections::{BTreeMap, BTreeSet};

const MAX_SMALL_ARRAY_ELEMENTS: u32 = 16;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Origin {
    root: mir::Local,
    max_iterations: u32,
}

#[derive(Debug)]
struct NaturalLoop {
    header: usize,
    blocks: BTreeSet<usize>,
}

pub(crate) fn detect(
    body: &mir::Body,
    reachable: &BTreeSet<usize>,
    num_args: usize,
) -> BTreeMap<usize, u32> {
    if !enabled() || reachable.is_empty() {
        return BTreeMap::new();
    }

    let verbose = std::env::var_os("CUDA_OXIDE_VERBOSE").is_some();
    let local_count = body.locals().len();
    let assignment_counts = assignment_counts(body, reachable);
    let mut origins = vec![None; local_count];
    let mut control_origins = vec![None; local_count];
    let mut invalid_roots = FxHashSet::default();

    for local_index in (num_args + 1)..local_count {
        let local = mir::Local::from(local_index);
        let Some(length) = fixed_array_len(&body.locals()[local].ty) else {
            continue;
        };

        if verbose {
            eprintln!(
                "small-array-iterators: array local={local_index}, length={length}, assignments={}",
                assignment_counts[local_index]
            );
        }

        if length == 0 || length > MAX_SMALL_ARRAY_ELEMENTS || assignment_counts[local_index] != 1 {
            continue;
        }

        origins[local_index] = Some(Origin {
            root: local,
            max_iterations: length,
        });
    }

    let iteration_limit = local_count
        .saturating_mul(2)
        .saturating_add(reachable.len())
        .saturating_add(1);

    for _ in 0..iteration_limit {
        let mut changed = false;

        for &block_index in reachable {
            let block = &body.blocks[block_index];

            for statement in &block.statements {
                let mir::StatementKind::Assign(destination, rvalue) = &statement.kind else {
                    continue;
                };

                if !destination.projection.is_empty() {
                    if let Some(origin) = origin_for_local(&origins, destination.local) {
                        let destination_ty = &body.locals()[destination.local].ty;
                        let mutates_root_array = destination.local == origin.root;
                        let mutates_unknown_derived_value =
                            !is_supported_adapter_type(destination_ty);

                        if mutates_root_array || mutates_unknown_derived_value {
                            changed |= invalid_roots.insert(origin.root);
                        }
                    }
                    continue;
                }

                let destination_index = destination.local;

                if origins[destination_index].is_none()
                    && let Some(origin) = rvalue_origin(
                        body,
                        &origins,
                        &mut invalid_roots,
                        rvalue,
                        destination.local,
                    )
                {
                    origins[destination_index] = Some(origin);
                    changed = true;
                }

                if control_origins[destination_index].is_none()
                    && let Some(origin) =
                        control_rvalue_origin(body, &origins, &control_origins, rvalue)
                {
                    control_origins[destination_index] = Some(origin);
                    changed = true;
                }
            }

            let mir::TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } = &block.terminator.kind
            else {
                continue;
            };

            let argument_origins = args
                .iter()
                .filter_map(|argument| operand_state_origin(body, &origins, argument))
                .collect::<Vec<_>>();
            if argument_origins.is_empty() {
                continue;
            }

            let callee = core_callee_leaf(func);
            let supported = callee.as_deref().is_some_and(is_supported_iterator_method);

            if !supported {
                for origin in argument_origins {
                    changed |= invalid_roots.insert(origin.root);
                }
                continue;
            }

            if !destination.projection.is_empty() || !all_same_origin(&argument_origins) {
                continue;
            }
            let Some(origin) = argument_origins.first().copied() else {
                continue;
            };

            let destination_index = destination.local;
            if callee.as_deref() == Some("next") {
                // `next` returns an element/Option, not iterator state. Keep it
                // out of ordinary provenance, but remember that its result
                // controls the loop's exit discriminant.
                if control_origins[destination_index].is_none() {
                    control_origins[destination_index] = Some(origin);
                    changed = true;
                }
            } else if origins[destination_index].is_none() {
                origins[destination_index] = Some(origin);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    let loops = natural_loops(body, reachable);
    let mut candidates: FxHashMap<usize, Option<u32>> = FxHashMap::default();

    // Recognize loops whose exiting branch is controlled by array-derived
    // iterator state. This covers iterator helpers that remain as calls and
    // iterator machinery that rustc has already inlined into pointer/index
    // comparisons.
    for natural_loop in &loops {
        for &block_index in &natural_loop.blocks {
            let block = &body.blocks[block_index];
            if !is_exiting_block(block, &natural_loop.blocks, reachable) {
                continue;
            }

            let Some(origin) =
                terminator_control_origin(body, &origins, &control_origins, &block.terminator.kind)
            else {
                continue;
            };
            if invalid_roots.contains(&origin.root) {
                continue;
            }

            record_candidate(&mut candidates, natural_loop.header, origin.max_iterations);
        }
    }

    // Keep a direct `next` fallback for projected control chains, but require
    // the relaxed loop-local propagation below to reach an ordinary loop exit.
    for &block_index in reachable {
        let block = &body.blocks[block_index];
        let mir::TerminatorKind::Call { func, args, .. } = &block.terminator.kind else {
            continue;
        };
        if core_callee_leaf(func).as_deref() != Some("next") {
            continue;
        }

        let call_origins = args
            .iter()
            .filter_map(|argument| operand_state_origin(body, &origins, argument))
            .filter(|origin| !invalid_roots.contains(&origin.root))
            .collect::<Vec<_>>();
        if !all_same_origin(&call_origins) {
            continue;
        }
        let Some(origin) = call_origins.first().copied() else {
            continue;
        };

        if let Some(natural_loop) = innermost_loop_containing(&loops, block_index)
            && loop_exit_is_controlled_by_root(
                body,
                natural_loop,
                reachable,
                &origins,
                &control_origins,
                origin.root,
            )
        {
            record_candidate(&mut candidates, natural_loop.header, origin.max_iterations);
        }
    }

    // Final structural fallback for public-MIR shapes where rustc has
    // distributed the iterator exit predicate across projected assignments and
    // merged temporaries that no longer retain a direct local-to-local path.
    //
    // Body-wide provenance is not sufficient: an array-derived iterator may be
    // evaluated before an unrelated loop. Re-run control propagation inside the
    // unique loop while conservatively folding projected destinations into their
    // base locals, and require that the resulting provenance reaches an ordinary
    // loop exit.
    if candidates.is_empty() && loops.len() == 1 {
        let mut valid_roots: FxHashMap<mir::Local, u32> = FxHashMap::default();
        for origin in origins
            .iter()
            .chain(control_origins.iter())
            .filter_map(|origin| *origin)
            .filter(|origin| !invalid_roots.contains(&origin.root))
        {
            valid_roots
                .entry(origin.root)
                .and_modify(|bound| {
                    if *bound != origin.max_iterations {
                        *bound = 0;
                    }
                })
                .or_insert(origin.max_iterations);
        }

        if valid_roots.len() == 1
            && let Some((&root, &max_iterations)) = valid_roots.iter().next()
            && max_iterations != 0
        {
            let derived_state_count = origins
                .iter()
                .enumerate()
                .filter(|(local, origin)| {
                    mir::Local::from(*local) != root
                        && origin.as_ref().is_some_and(|origin| origin.root == root)
                })
                .count();
            let control_state_count = control_origins
                .iter()
                .filter(|origin| origin.as_ref().is_some_and(|origin| origin.root == root))
                .count();
            let natural_loop = &loops[0];
            let has_iterator_step = loop_contains_iterator_step(body, natural_loop);
            let exit_is_controlled = loop_exit_is_controlled_by_root(
                body,
                natural_loop,
                reachable,
                &origins,
                &control_origins,
                root,
            );

            if derived_state_count >= 2
                && control_state_count != 0
                && has_iterator_step
                && exit_is_controlled
            {
                record_candidate(&mut candidates, natural_loop.header, max_iterations);

                if verbose {
                    eprintln!(
                        "small-array-iterators: structural fallback header={}, root={:?}, bound={}, derived={}, control={}, iterator_step={}, exit_control={}",
                        natural_loop.header,
                        root,
                        max_iterations,
                        derived_state_count,
                        control_state_count,
                        has_iterator_step,
                        exit_is_controlled,
                    );
                }
            }
        }
    }

    let result = candidates
        .into_iter()
        .filter_map(|(block, bound)| bound.map(|bound| (block, bound)))
        .collect::<BTreeMap<_, _>>();

    if verbose {
        let tracked = origins
            .iter()
            .enumerate()
            .filter_map(|(local, origin)| origin.map(|origin| (local, origin)))
            .collect::<Vec<_>>();
        let control = control_origins
            .iter()
            .enumerate()
            .filter_map(|(local, origin)| origin.map(|origin| (local, origin)))
            .collect::<Vec<_>>();
        let loop_headers = loops
            .iter()
            .map(|natural_loop| (natural_loop.header, natural_loop.blocks.len()))
            .collect::<Vec<_>>();

        eprintln!("small-array-iterators: tracked={tracked:?}");
        eprintln!("small-array-iterators: control={control:?}");
        eprintln!("small-array-iterators: invalid_roots={invalid_roots:?}");
        eprintln!("small-array-iterators: loops={loop_headers:?}");
        eprintln!("small-array-iterators: hints={result:?}");
    }

    result
}

pub(crate) fn enabled() -> bool {
    std::env::var("CUDA_OXIDE_MIR_OPTS")
        .ok()
        .is_some_and(|value| {
            value
                .split(',')
                .map(str::trim)
                .any(|name| name == "small-array-iterators")
        })
}

fn record_candidate(candidates: &mut FxHashMap<usize, Option<u32>>, block: usize, bound: u32) {
    candidates
        .entry(block)
        .and_modify(|existing| {
            if *existing != Some(bound) {
                *existing = None;
            }
        })
        .or_insert(Some(bound));
}

fn assignment_counts(body: &mir::Body, reachable: &BTreeSet<usize>) -> Vec<usize> {
    let mut counts = vec![0usize; body.locals().len()];

    for &block_index in reachable {
        let block = &body.blocks[block_index];

        for statement in &block.statements {
            if let mir::StatementKind::Assign(destination, _) = &statement.kind
                && destination.projection.is_empty()
            {
                counts[destination.local] += 1;
            }
        }

        if let mir::TerminatorKind::Call { destination, .. } = &block.terminator.kind
            && destination.projection.is_empty()
        {
            counts[destination.local] += 1;
        }
    }

    counts
}

fn rvalue_origin(
    body: &mir::Body,
    origins: &[Option<Origin>],
    invalid_roots: &mut FxHashSet<mir::Local>,
    rvalue: &mir::Rvalue,
    destination: mir::Local,
) -> Option<Origin> {
    match rvalue {
        mir::Rvalue::Use(operand) => operand_origin(origins, operand),
        mir::Rvalue::CopyForDeref(place) => whole_local_origin(origins, place),
        mir::Rvalue::Ref(_, borrow_kind, place) => {
            let origin = base_place_origin(origins, place)?;
            if matches!(borrow_kind, mir::BorrowKind::Mut { .. }) && place.local == origin.root {
                invalid_roots.insert(origin.root);
                return None;
            }
            Some(origin)
        }
        mir::Rvalue::AddressOf(mutability, place) => {
            let origin = base_place_origin(origins, place)?;
            if matches!(mutability, mir::RawPtrKind::Mut) && place.local == origin.root {
                invalid_roots.insert(origin.root);
                return None;
            }
            Some(origin)
        }
        mir::Rvalue::Cast(
            mir::CastKind::PointerCoercion(
                mir::PointerCoercion::Unsize | mir::PointerCoercion::MutToConstPointer,
            ),
            operand,
            _,
        ) => operand_origin(origins, operand),
        mir::Rvalue::BinaryOp(mir::BinOp::Offset, left, right) => same_nonempty_origin(
            [
                operand_origin(origins, left),
                operand_origin(origins, right),
            ]
            .into_iter()
            .flatten(),
        ),
        mir::Rvalue::Aggregate(_, operands)
            if is_supported_adapter_type(&body.locals()[destination].ty) =>
        {
            same_nonempty_origin(
                operands
                    .iter()
                    .filter_map(|operand| operand_origin(origins, operand)),
            )
        }
        _ => None,
    }
}

fn control_rvalue_origin(
    body: &mir::Body,
    origins: &[Option<Origin>],
    control_origins: &[Option<Origin>],
    rvalue: &mir::Rvalue,
) -> Option<Origin> {
    match rvalue {
        mir::Rvalue::Use(operand) => {
            operand_control_origin(body, origins, control_origins, operand)
        }
        mir::Rvalue::CopyForDeref(place) => {
            whole_local_control_origin(body, origins, control_origins, place)
        }
        mir::Rvalue::Cast(_, operand, _) | mir::Rvalue::UnaryOp(_, operand) => {
            operand_control_origin(body, origins, control_origins, operand)
        }
        mir::Rvalue::BinaryOp(operator, left, right)
        | mir::Rvalue::CheckedBinaryOp(operator, left, right) => {
            binary_control_origin(body, origins, control_origins, operator, left, right)
        }
        mir::Rvalue::Len(place) | mir::Rvalue::Discriminant(place) => {
            whole_local_control_origin(body, origins, control_origins, place)
        }
        _ => None,
    }
}

fn binary_control_origin(
    body: &mir::Body,
    origins: &[Option<Origin>],
    control_origins: &[Option<Origin>],
    operator: &mir::BinOp,
    left: &mir::Operand,
    right: &mir::Operand,
) -> Option<Origin> {
    let left_direct = operand_state_origin(body, origins, left);
    let right_direct = operand_state_origin(body, origins, right);
    let left_control = operand_control_only_origin(control_origins, left);
    let right_control = operand_control_only_origin(control_origins, right);

    let combined = [left_direct, right_direct, left_control, right_control]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if !all_same_origin(&combined) {
        return None;
    }
    let origin = combined.first().copied()?;

    let is_comparison = matches!(
        operator,
        mir::BinOp::Eq
            | mir::BinOp::Ne
            | mir::BinOp::Lt
            | mir::BinOp::Le
            | mir::BinOp::Gt
            | mir::BinOp::Ge
    );

    if !is_comparison {
        return Some(origin);
    }

    // A direct derived pointer compared with a constant (for example null) is
    // not a proof of an array-length bound. Accept comparisons only when both
    // sides carry the same direct provenance, or when an earlier scalar
    // operation such as Len/PtrMetadata already derived a control value.
    let direct_count = (left_direct.is_some() as usize) + (right_direct.is_some() as usize);
    let has_derived_control = left_control.is_some() || right_control.is_some();
    (direct_count >= 2 || has_derived_control).then_some(origin)
}

fn operand_origin(origins: &[Option<Origin>], operand: &mir::Operand) -> Option<Origin> {
    match operand {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => whole_local_origin(origins, place),
        _ => None,
    }
}

/// Returns provenance for a whole derived value or for a projection of a
/// supported iterator-adapter value.
///
/// Arbitrary projections of the root array remain rejected: an element value
/// is data, not proof that a loop can execute at most `N` times. Adapter fields
/// are different; their pointers, counters, and nested adapters are precisely
/// the state whose exhaustion bounds iteration.
fn operand_state_origin(
    body: &mir::Body,
    origins: &[Option<Origin>],
    operand: &mir::Operand,
) -> Option<Origin> {
    match operand {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            place_state_origin(body, origins, place)
        }
        _ => None,
    }
}

fn place_state_origin(
    body: &mir::Body,
    origins: &[Option<Origin>],
    place: &mir::Place,
) -> Option<Origin> {
    let origin = origin_for_local(origins, place.local)?;
    if place.projection.is_empty() {
        return Some(origin);
    }

    is_supported_adapter_carrier_type(&body.locals()[place.local].ty).then_some(origin)
}

fn operand_control_origin(
    body: &mir::Body,
    origins: &[Option<Origin>],
    control_origins: &[Option<Origin>],
    operand: &mir::Operand,
) -> Option<Origin> {
    operand_state_origin(body, origins, operand)
        .or_else(|| operand_control_only_origin(control_origins, operand))
}

fn operand_control_only_origin(
    control_origins: &[Option<Origin>],
    operand: &mir::Operand,
) -> Option<Origin> {
    match operand {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            // Once a local has been proven to carry only cardinality/control
            // information, projecting a field or checked-operation component
            // cannot turn it back into arbitrary array data. This is required
            // for `Option` discriminants and `(value, overflow)` projections.
            origin_for_local(control_origins, place.local)
        }
        _ => None,
    }
}

/// Returns the origin of the local that owns a place, including projected places.
///
/// This is used only when creating references or raw pointers. A reference to
/// an element or subplace still derives from the same fixed local array.
fn base_place_origin(origins: &[Option<Origin>], place: &mir::Place) -> Option<Origin> {
    origin_for_local(origins, place.local)
}

/// Returns an origin only when the operand denotes a complete MIR local.
///
/// Propagating array provenance through an arbitrary projected `Copy` or `Move`
/// would incorrectly classify scalar fields and elements as the whole array.
fn whole_local_origin(origins: &[Option<Origin>], place: &mir::Place) -> Option<Origin> {
    if !place.projection.is_empty() {
        return None;
    }
    origin_for_local(origins, place.local)
}

fn whole_local_control_origin(
    body: &mir::Body,
    origins: &[Option<Origin>],
    control_origins: &[Option<Origin>],
    place: &mir::Place,
) -> Option<Origin> {
    // Projections of a value already classified as control-only remain
    // control-only. Typical examples are the discriminant of `Option<Item>`
    // returned by `next` and the boolean field of a checked operation result.
    if let Some(origin) = origin_for_local(control_origins, place.local) {
        return Some(origin);
    }

    place_state_origin(body, origins, place)
}

fn origin_for_local(origins: &[Option<Origin>], local: mir::Local) -> Option<Origin> {
    origins.get(local).copied().flatten()
}

fn terminator_control_origin(
    body: &mir::Body,
    origins: &[Option<Origin>],
    control_origins: &[Option<Origin>],
    terminator: &mir::TerminatorKind,
) -> Option<Origin> {
    match terminator {
        mir::TerminatorKind::SwitchInt { discr, .. } => {
            operand_control_origin(body, origins, control_origins, discr)
        }
        mir::TerminatorKind::Assert { cond, .. } => {
            operand_control_origin(body, origins, control_origins, cond)
        }
        _ => None,
    }
}

fn natural_loops(body: &mir::Body, reachable: &BTreeSet<usize>) -> Vec<NaturalLoop> {
    let Some(&entry) = reachable.first() else {
        return Vec::new();
    };

    let mut predecessors = vec![Vec::<usize>::new(); body.blocks.len()];
    for &source in reachable {
        for target in body.blocks[source].terminator.successors() {
            if reachable.contains(&target) {
                predecessors[target].push(source);
            }
        }
    }

    let all_blocks = reachable.clone();
    let mut dominators = vec![BTreeSet::new(); body.blocks.len()];
    for &block in reachable {
        dominators[block] = if block == entry {
            BTreeSet::from([entry])
        } else {
            all_blocks.clone()
        };
    }

    loop {
        let mut changed = false;
        for &block in reachable {
            if block == entry {
                continue;
            }

            let mut next = if let Some((&first, rest)) = predecessors[block].split_first() {
                let mut intersection = dominators[first].clone();
                for &predecessor in rest {
                    intersection = intersection
                        .intersection(&dominators[predecessor])
                        .copied()
                        .collect();
                }
                intersection
            } else {
                BTreeSet::new()
            };
            next.insert(block);

            if next != dominators[block] {
                dominators[block] = next;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    let mut by_header: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &latch in reachable {
        for header in body.blocks[latch].terminator.successors() {
            if !reachable.contains(&header) || !dominators[latch].contains(&header) {
                continue;
            }

            let mut blocks = BTreeSet::from([header, latch]);
            let mut frontier = vec![latch];
            while let Some(block) = frontier.pop() {
                for &predecessor in &predecessors[block] {
                    if blocks.insert(predecessor) && predecessor != header {
                        frontier.push(predecessor);
                    }
                }
            }

            by_header.entry(header).or_default().extend(blocks);
        }
    }

    let mut loops = by_header
        .into_iter()
        .map(|(header, blocks)| NaturalLoop { header, blocks })
        .collect::<Vec<_>>();
    loops.sort_by_key(|natural_loop| (natural_loop.blocks.len(), natural_loop.header));
    loops
}

fn is_exiting_block(
    block: &mir::BasicBlock,
    loop_blocks: &BTreeSet<usize>,
    reachable: &BTreeSet<usize>,
) -> bool {
    block
        .terminator
        .successors()
        .into_iter()
        .any(|successor| reachable.contains(&successor) && !loop_blocks.contains(&successor))
}

fn innermost_loop_containing(loops: &[NaturalLoop], block: usize) -> Option<&NaturalLoop> {
    loops
        .iter()
        .filter(|natural_loop| natural_loop.blocks.contains(&block))
        .min_by_key(|natural_loop| (natural_loop.blocks.len(), natural_loop.header))
}

fn loop_contains_iterator_step(body: &mir::Body, natural_loop: &NaturalLoop) -> bool {
    natural_loop.blocks.iter().any(|&block_index| {
        let mir::TerminatorKind::Call { func, .. } = &body.blocks[block_index].terminator.kind
        else {
            return false;
        };

        matches!(
            core_callee_leaf(func).as_deref(),
            Some("next") | Some("nth")
        )
    })
}

fn loop_exit_is_controlled_by_root(
    body: &mir::Body,
    natural_loop: &NaturalLoop,
    reachable: &BTreeSet<usize>,
    origins: &[Option<Origin>],
    control_origins: &[Option<Origin>],
    root: mir::Local,
) -> bool {
    let mut relaxed_control_origins = control_origins.to_vec();
    let iteration_limit = body
        .locals()
        .len()
        .saturating_add(natural_loop.blocks.len())
        .saturating_add(1);

    for _ in 0..iteration_limit {
        let mut changed = false;

        for &block_index in &natural_loop.blocks {
            let block = &body.blocks[block_index];

            for statement in &block.statements {
                let mir::StatementKind::Assign(destination, rvalue) = &statement.kind else {
                    continue;
                };
                if relaxed_control_origins[destination.local].is_some() {
                    continue;
                }

                let Some(origin) =
                    control_rvalue_origin(body, origins, &relaxed_control_origins, rvalue)
                else {
                    continue;
                };
                if origin.root != root {
                    continue;
                }

                // The ordinary propagation deliberately rejects projected
                // destinations. Inside this fallback, folding the projection
                // into its base local is safe only because the taint must still
                // reach an actual exit terminator below.
                relaxed_control_origins[destination.local] = Some(origin);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    natural_loop.blocks.iter().any(|&block_index| {
        let block = &body.blocks[block_index];
        if !is_exiting_block(block, &natural_loop.blocks, reachable) {
            return false;
        }

        terminator_control_origin(
            body,
            origins,
            &relaxed_control_origins,
            &block.terminator.kind,
        )
        .is_some_and(|origin| origin.root == root)
    })
}

fn fixed_array_len(ty: &Ty) -> Option<u32> {
    let TyKind::RigidTy(RigidTy::Array(_, length)) = ty.kind() else {
        return None;
    };

    let value = match length.kind() {
        TyConstKind::Value(_, allocation) => allocation.read_uint().ok()?,
        _ => u128::from(length.eval_target_usize().ok()?),
    };

    u32::try_from(value).ok()
}

fn core_callee_leaf(func: &mir::Operand) -> Option<String> {
    let (crate_name, leaf) = callee_leaf(func)?;
    (crate_name == "core").then_some(leaf)
}

fn callee_leaf(func: &mir::Operand) -> Option<(String, String)> {
    let mir::Operand::Constant(constant) = func else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(definition, _)) = constant.const_.ty().kind() else {
        return None;
    };

    let full_name = definition.name().to_string();
    let leaf = full_name
        .rsplit("::")
        .next()
        .unwrap_or(full_name.as_str())
        .to_string();

    Some((definition.krate().name.as_str().to_string(), leaf))
}

fn is_supported_iterator_method(name: &str) -> bool {
    matches!(
        name,
        "iter" | "into_iter" | "copied" | "take" | "skip" | "enumerate" | "next"
    )
}

fn is_supported_adapter_type(ty: &Ty) -> bool {
    let TyKind::RigidTy(RigidTy::Adt(definition, _)) = ty.kind() else {
        return false;
    };

    matches!(
        definition.trimmed_name().as_str(),
        "Iter" | "Copied" | "Take" | "Skip" | "Enumerate"
    )
}

fn is_supported_adapter_carrier_type(ty: &Ty) -> bool {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(..)) => is_supported_adapter_type(ty),
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _))
        | TyKind::RigidTy(RigidTy::RawPtr(pointee, _)) => {
            is_supported_adapter_carrier_type(&pointee)
        }
        _ => false,
    }
}

fn same_nonempty_origin(origins: impl IntoIterator<Item = Origin>) -> Option<Origin> {
    let origins = origins.into_iter().collect::<Vec<_>>();
    if all_same_origin(&origins) {
        origins.first().copied()
    } else {
        None
    }
}

fn all_same_origin(origins: &[Origin]) -> bool {
    let Some(first) = origins.first() else {
        return false;
    };
    origins.iter().all(|origin| origin == first)
}
