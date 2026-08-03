/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optional MIR passes selected by `CUDA_OXIDE_MIR_PASSES`.
//!
//! This registry runs after the standard MIR preparation passes. Add only
//! workload-specific passes that are safe at that point; defaults belong in
//! the normal pipeline instead. New entries need correctness coverage and a
//! measured workload benefit.

use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    pass::{AnalysisManager, Pass, PassResult, Passes},
    result::Result,
};
use thiserror::Error;

type OptCtor = fn() -> Box<dyn Pass>;

struct OptEntry {
    name: &'static str,
    build: OptCtor,
}

/// Errors from selecting a MIR pass pipeline.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MirPassPipelineError {
    #[error("empty opt name in pipeline")]
    EmptyName,
    #[error("unknown MIR pass \"{name}\"; available passes: {available}")]
    UnknownName { name: String, available: String },
}

/// The cuda-oxide-owned registry of post-preparation MIR passes.
#[derive(Default)]
pub struct MirPassRegistry {
    entries: Vec<OptEntry>,
}

impl MirPassRegistry {
    /// Build a comma-separated pipeline. Empty specs select no passes.
    pub fn build_pipeline(&self, spec: &str) -> std::result::Result<Passes, MirPassPipelineError> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Ok(Passes::default());
        }
        let constructors = spec
            .split(',')
            .map(str::trim)
            .map(|name| self.lookup(name))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut passes = Passes::default();
        for build in constructors {
            passes.add_pass(BoxedPass(build()));
        }
        Ok(passes)
    }

    fn lookup(&self, name: &str) -> std::result::Result<OptCtor, MirPassPipelineError> {
        if name.is_empty() {
            return Err(MirPassPipelineError::EmptyName);
        }
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.build)
            .ok_or_else(|| MirPassPipelineError::UnknownName {
                name: name.to_owned(),
                available: self
                    .entries
                    .iter()
                    .map(|entry| entry.name)
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }
}

/// Build the registry of supported optional CUDA Oxide MIR passes.
pub fn registry() -> MirPassRegistry {
    MirPassRegistry::default()
}

struct BoxedPass(Box<dyn Pass>);

impl Pass for BoxedPass {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn run(
        &mut self,
        operation: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        self.0.run(operation, ctx, analyses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::ops::ModuleOp;
    use pliron::op::Op;
    use std::sync::Mutex;

    static RUNS: Mutex<Vec<&str>> = Mutex::new(Vec::new());

    struct TestPass(&'static str);

    impl Pass for TestPass {
        fn name(&self) -> &str {
            self.0
        }

        fn run(
            &mut self,
            _operation: Ptr<Operation>,
            _ctx: &mut Context,
            _analyses: &mut AnalysisManager,
        ) -> Result<PassResult> {
            RUNS.lock().unwrap().push(self.0);
            Ok(PassResult::default())
        }
    }

    fn first() -> Box<dyn Pass> {
        Box::new(TestPass("first"))
    }

    fn second() -> Box<dyn Pass> {
        Box::new(TestPass("second"))
    }

    fn run(passes: &mut Passes) {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        passes
            .run(
                module.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();
    }

    fn registry_with_test_passes() -> MirPassRegistry {
        MirPassRegistry {
            entries: vec![
                OptEntry {
                    name: "first",
                    build: first,
                },
                OptEntry {
                    name: "second",
                    build: second,
                },
            ],
        }
    }

    #[test]
    fn default_registry_is_empty() {
        assert!(registry().build_pipeline("").is_ok());
        assert!(matches!(
            registry().build_pipeline("first"),
            Err(MirPassPipelineError::UnknownName { .. })
        ));
    }

    #[test]
    fn selected_passes_run_in_order() {
        RUNS.lock().unwrap().clear();
        let mut passes = registry_with_test_passes()
            .build_pipeline("first,second")
            .unwrap();
        run(&mut passes);
        assert_eq!(*RUNS.lock().unwrap(), ["first", "second"]);
    }

    #[test]
    fn invalid_pipeline_does_not_run_a_prefix() {
        RUNS.lock().unwrap().clear();
        assert!(
            registry_with_test_passes()
                .build_pipeline("first,missing")
                .is_err()
        );
        assert!(matches!(
            registry_with_test_passes().build_pipeline("first,"),
            Err(MirPassPipelineError::EmptyName)
        ));
        assert!(RUNS.lock().unwrap().is_empty());
    }

    #[test]
    fn empty_pipeline_runs_nothing() {
        RUNS.lock().unwrap().clear();
        let mut passes = registry_with_test_passes().build_pipeline("").unwrap();
        run(&mut passes);
        assert!(RUNS.lock().unwrap().is_empty());
    }
}
