/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Report how much of the pinned LLVM metadata the generator actually covers.
//!
//! `extract` pulls every NVVM intrinsic record LLVM knows about into
//! `intrinsics/imported.json`. `generate` then emits bindings for the subset
//! described by `intrinsics/catalog.json`. The difference is intrinsics that
//! reached the repository and stopped there - visible in a JSON file nobody
//! reads, rather than in any error.
//!
//! That difference is large, and most of it is not worth closing. Surface and
//! texture operations account for roughly a third of it and are irrelevant to
//! compute kernels, so a raw "72% uncovered" figure overstates the real backlog
//! by a wide margin. The useful output is therefore per family, not a single
//! percentage: it tells a contributor which families are worth aiming at and
//! which to ignore.
//!
//! ```text
//! cargo run -p cuda-intrinsics-gen -- coverage
//! cargo run -p cuda-intrinsics-gen -- coverage --family shfl
//! ```
//!
//! This is a reporting command. It reads the two JSON files and writes nothing.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Families that exist in the metadata but are not compute functionality.
///
/// Surface, texture, and their query forms. Counted separately so the headline
/// backlog reflects work someone might actually want to do.
const NON_COMPUTE_PREFIXES: &[&str] = &["sust", "suld", "suq", "tex", "tld4", "txq"];

/// One family's coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyCoverage {
    /// Family name, taken from the intrinsic's leading path segment.
    pub family: String,
    /// Records present in `catalog.json`.
    pub generated: usize,
    /// Records in `imported.json` with no catalog entry.
    pub ungenerated: usize,
    /// Whether this family is compute-relevant.
    pub compute: bool,
}

impl FamilyCoverage {
    /// Records seen in either file.
    #[must_use]
    pub fn total(&self) -> usize {
        self.generated + self.ungenerated
    }
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    /// Per family, ordered by ungenerated count descending.
    pub families: Vec<FamilyCoverage>,
    /// Distinct intrinsics in `imported.json`.
    pub imported: usize,
    /// Distinct intrinsics in `catalog.json`.
    pub generated: usize,
}

impl Coverage {
    /// Ungenerated records across every family.
    #[must_use]
    pub fn ungenerated(&self) -> usize {
        self.imported.saturating_sub(self.generated)
    }

    /// Ungenerated records in compute-relevant families only.
    ///
    /// The number worth quoting: the raw total is dominated by surface and
    /// texture work no compute kernel needs.
    #[must_use]
    pub fn ungenerated_compute(&self) -> usize {
        self.families
            .iter()
            .filter(|f| f.compute)
            .map(|f| f.ungenerated)
            .sum()
    }
}

/// Family an intrinsic belongs to.
///
/// Uses the leading segment after the `int_nvvm_` prefix, which is how the
/// metadata already groups them. Numeric suffixes are kept, so `f2i` and `d2i`
/// stay distinct - they have different coverage stories.
fn family_of(name: &str) -> String {
    let stem = name.strip_prefix("int_nvvm_").unwrap_or(name);
    // Longest known prefix wins, so `shfl_sync` groups under `shfl` rather than
    // splitting into its own family.
    for prefix in NON_COMPUTE_PREFIXES {
        if stem.starts_with(prefix) {
            return (*prefix).to_string();
        }
    }
    stem.split('_').next().unwrap_or(stem).to_string()
}

/// Every `int_nvvm_*` name appearing anywhere in a JSON document.
///
/// The two files have different shapes, so this walks rather than assuming a
/// schema; a schema change should not silently zero the report.
fn collect_names(value: &Value, out: &mut BTreeSet<String>) {
    match value {
        Value::String(s) if s.starts_with("int_nvvm_") => {
            out.insert(s.clone());
        }
        Value::Array(items) => items.iter().for_each(|v| collect_names(v, out)),
        Value::Object(map) => map.values().for_each(|v| collect_names(v, out)),
        _ => {}
    }
}

/// Build the report from the two pinned JSON files.
pub fn compute(repo_root: &Path) -> Result<Coverage> {
    let read = |rel: &str| -> Result<Value> {
        let path = repo_root.join(rel);
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    };

    let mut imported = BTreeSet::new();
    collect_names(&read("intrinsics/imported.json")?, &mut imported);
    let mut generated = BTreeSet::new();
    collect_names(&read("intrinsics/catalog.json")?, &mut generated);

    let mut per_family: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for name in &imported {
        let entry = per_family.entry(family_of(name)).or_default();
        if generated.contains(name) {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    let mut families: Vec<FamilyCoverage> = per_family
        .into_iter()
        .map(|(family, (done, ungen))| FamilyCoverage {
            compute: !NON_COMPUTE_PREFIXES.contains(&family.as_str()),
            family,
            generated: done,
            ungenerated: ungen,
        })
        .collect();
    // Largest gap first: that is the order a contributor reads it in.
    families.sort_by(|a, b| {
        b.ungenerated
            .cmp(&a.ungenerated)
            .then_with(|| a.family.cmp(&b.family))
    });

    Ok(Coverage {
        families,
        imported: imported.len(),
        generated: generated.len(),
    })
}

/// Print the report.
///
/// `family` filters to one family, for checking a single area before working
/// on it.
pub fn run(repo_root: &Path, family: Option<&str>) -> Result<()> {
    let coverage = compute(repo_root)?;

    if let Some(wanted) = family {
        let Some(f) = coverage.families.iter().find(|f| f.family == wanted) else {
            println!("no family `{wanted}` in the pinned metadata");
            return Ok(());
        };
        println!(
            "{}: {} generated, {} ungenerated ({} total){}",
            f.family,
            f.generated,
            f.ungenerated,
            f.total(),
            if f.compute { "" } else { "  [not compute]" }
        );
        return Ok(());
    }

    println!(
        "pinned NVVM intrinsics: {} imported, {} generated, {} ungenerated",
        coverage.imported,
        coverage.generated,
        coverage.ungenerated()
    );
    println!(
        "ungenerated in compute families: {}  (the rest is surface/texture)",
        coverage.ungenerated_compute()
    );
    println!();
    println!("{:<20}{:>10}{:>13}", "family", "generated", "ungenerated");
    for f in coverage.families.iter().filter(|f| f.ungenerated > 0) {
        println!(
            "{:<20}{:>10}{:>13}{}",
            f.family,
            f.generated,
            f.ungenerated,
            if f.compute { "" } else { "   [not compute]" }
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn family_uses_the_leading_segment() {
        assert_eq!(family_of("int_nvvm_fma_rn_f32"), "fma");
        assert_eq!(family_of("int_nvvm_mbarrier_arrive"), "mbarrier");
        // `shfl_sync_*` groups with `shfl`, not as its own family.
        assert_eq!(family_of("int_nvvm_shfl_sync_bfly_f32p"), "shfl");
        assert_eq!(family_of("int_nvvm_shfl_bfly_i32"), "shfl");
    }

    /// Surface and texture forms must land in the non-compute bucket, since
    /// separating them is the whole point of the report.
    #[test]
    fn surface_and_texture_are_not_compute() {
        for n in [
            "int_nvvm_sust_b_1d_i32_clamp",
            "int_nvvm_suld_1d_i8_trap",
            "int_nvvm_tex_1d_v4f32_s32",
            "int_nvvm_tld4_r_2d_v4f32_f32",
            "int_nvvm_suq_width",
            "int_nvvm_txq_height",
        ] {
            let f = family_of(n);
            assert!(
                NON_COMPUTE_PREFIXES.contains(&f.as_str()),
                "{n} -> {f} should be non-compute"
            );
        }
        // And compute families must not be swept up with them.
        for n in ["int_nvvm_fma_rn_f32", "int_nvvm_texsurf_handle"] {
            let f = family_of(n);
            if n.contains("fma") {
                assert!(!NON_COMPUTE_PREFIXES.contains(&f.as_str()));
            }
        }
    }

    #[test]
    fn counts_split_generated_from_ungenerated() {
        let mut imported = BTreeSet::new();
        collect_names(
            &json!({"records": [
                {"id": "int_nvvm_fma_rn_f32"},
                {"id": "int_nvvm_fma_rn_f64"},
                {"id": "int_nvvm_sust_b_1d_i32_clamp"},
                {"nested": {"deep": "int_nvvm_shfl_sync_idx_f32p"}},
            ]}),
            &mut imported,
        );
        assert_eq!(imported.len(), 4, "walks nested objects and arrays");
        assert!(imported.contains("int_nvvm_shfl_sync_idx_f32p"));
    }

    /// A totals identity, so a schema change that stops matching one file
    /// shows up as a broken invariant rather than a quietly wrong percentage.
    #[test]
    fn totals_are_consistent() {
        let c = Coverage {
            families: vec![
                FamilyCoverage {
                    family: "fma".into(),
                    generated: 10,
                    ungenerated: 5,
                    compute: true,
                },
                FamilyCoverage {
                    family: "sust".into(),
                    generated: 0,
                    ungenerated: 210,
                    compute: false,
                },
            ],
            imported: 225,
            generated: 10,
        };
        assert_eq!(c.ungenerated(), 215);
        assert_eq!(c.ungenerated_compute(), 5, "surface work is excluded");
        assert_eq!(c.families[0].total(), 15);
    }
}
