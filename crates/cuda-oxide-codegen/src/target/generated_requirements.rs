/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::arch::{arch_compute_capability_and_suffix, is_known_cuda_target};
use super::features::{ModuleRequirements, PtxIsaRequirement};
use crate::generated::{
    GeneratedModuleRequirements, GeneratedResolvedRequirement, GeneratedResolvedTarget,
};
use crate::generated_intrinsic_targets::{
    GeneratedHardwareAlternative, GeneratedHardwareTarget, GeneratedTargetContract,
    GeneratedTargetRequirement,
};
use cuda_target_spec::spelling_at_least;

/// Convert catalog PTX floors to the discrete `llc` feature spellings this
/// compiler supports. A floor between two spellings rounds upward; a future
/// floor beyond the newest supported spelling is rejected instead of ignored.
pub(crate) fn generated_ptx_isa_requirement(
    generated: &GeneratedModuleRequirements,
) -> Result<PtxIsaRequirement, String> {
    let mut requirement = PtxIsaRequirement::Default;
    for resolved in generated.resolved_targets() {
        let target_requirement = resolved_requirement(generated, resolved)?;
        let minimum_ptx =
            resolved_requirement_minimum_ptx(target_requirement).ok_or_else(|| {
                format!(
                    "generated intrinsic `{}` (`{}`) has an empty resolved target contract",
                    resolved.target.id, resolved.target.marker
                )
            })?;
        requirement = requirement.max(ptx_isa_requirement_for_floor(
            minimum_ptx,
            resolved.target.id,
            resolved.target.marker,
        )?);
    }
    Ok(requirement)
}

/// Resolve the PTX floor for the selected hardware alternative.
pub(crate) fn generated_ptx_isa_requirement_for_target(
    generated: &GeneratedModuleRequirements,
    arch: &str,
) -> Result<PtxIsaRequirement, String> {
    let mut requirement = PtxIsaRequirement::Default;
    for resolved in generated.resolved_targets() {
        let target_requirement = resolved_requirement(generated, resolved)?;
        let floor = resolved_requirement_ptx_floor(arch, target_requirement).ok_or_else(|| {
            format!(
                "CUDA target {arch} cannot lower generated intrinsic `{}` (`{}`); requires {}",
                resolved.target.id,
                resolved.target.marker,
                describe_resolved_requirement(target_requirement)
            )
        })?;
        requirement = requirement.max(ptx_isa_requirement_for_floor(
            floor,
            resolved.target.id,
            resolved.target.marker,
        )?);
    }
    validate_sm101_ptx_pair(arch, requirement)?;
    Ok(requirement)
}

pub(super) fn resolved_requirement(
    generated: &GeneratedModuleRequirements,
    resolved: &GeneratedResolvedTarget,
) -> Result<GeneratedResolvedRequirement, String> {
    generated.resolved_requirement(resolved).ok_or_else(|| {
        let selector = resolved
            .selector
            .map(|selector| format!("{}={}", selector.name, selector.value))
            .unwrap_or_else(|| "<none>".to_string());
        format!(
            "generated intrinsic `{}` (`{}`) has no unique target contract for {selector}",
            resolved.target.id, resolved.target.marker
        )
    })
}

fn resolved_requirement_minimum_ptx(requirement: GeneratedResolvedRequirement) -> Option<u16> {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => {
            Some(requirement.minimum_ptx.encoded())
        }
        GeneratedResolvedRequirement::Contract(contract) => contract
            .alternatives
            .iter()
            .map(|alternative| alternative.minimum_ptx.encoded())
            .min(),
    }
}

pub(super) fn ptx_isa_requirement_for_floor(
    encoded: u16,
    id: &str,
    marker: &str,
) -> Result<PtxIsaRequirement, String> {
    if encoded <= 60 {
        return Ok(PtxIsaRequirement::Default);
    }
    spelling_at_least(encoded)
        .and_then(PtxIsaRequirement::from_spelling)
        .ok_or_else(|| format!(
            "generated intrinsic `{id}` (`{marker}`) requires PTX {}.{}, newer than cuda-oxide can request",
            encoded / 10,
            encoded % 10
        ))
}

pub(crate) fn merge_generated_module_requirements(
    mut text: ModuleRequirements,
    generated: &GeneratedModuleRequirements,
) -> Result<ModuleRequirements, String> {
    text.ptx_isa = text.ptx_isa.max(generated_ptx_isa_requirement(generated)?);
    Ok(text)
}

pub(crate) fn merge_generated_module_requirements_for_target(
    mut text: ModuleRequirements,
    generated: &GeneratedModuleRequirements,
    arch: &str,
) -> Result<ModuleRequirements, String> {
    text.ptx_isa = text
        .ptx_isa
        .max(generated_ptx_isa_requirement_for_target(generated, arch)?);
    validate_sm101_ptx_pair(arch, text.ptx_isa)?;
    Ok(text)
}

/// Reject LLVM target names whose meaning changes at a newer PTX ISA.
fn validate_sm101_ptx_pair(arch: &str, requirement: PtxIsaRequirement) -> Result<(), String> {
    let Some((capability, suffix)) = arch_compute_capability_and_suffix(arch) else {
        return Ok(());
    };
    if capability == 101
        && matches!(suffix, Some('a' | 'f'))
        && requirement >= PtxIsaRequirement::Ptx90
    {
        return Err(format!(
            "CUDA target {arch} cannot be combined with PTX 9.0 or newer; LLVM renamed the sm_101 target to sm_110 at that PTX level"
        ));
    }
    Ok(())
}

pub(crate) fn generated_target_satisfied(
    arch: &str,
    generated: &GeneratedModuleRequirements,
) -> bool {
    generated.resolved_targets().iter().all(|resolved| {
        let Ok(requirement) = resolved_requirement(generated, resolved) else {
            return false;
        };
        resolved_hardware_satisfied(arch, requirement)
            && resolved_requirement_ptx_floor(arch, requirement).is_some_and(|floor| {
                ptx_isa_requirement_for_floor(floor, resolved.target.id, resolved.target.marker)
                    .is_ok()
            })
    })
}

fn resolved_hardware_satisfied(arch: &str, requirement: GeneratedResolvedRequirement) -> bool {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => {
            generated_hardware_satisfied(arch, requirement.hardware)
        }
        GeneratedResolvedRequirement::Contract(contract) => {
            generated_contract_satisfied(arch, contract)
        }
    }
}

fn generated_hardware_satisfied(arch: &str, hardware: GeneratedHardwareTarget) -> bool {
    let Some((capability, suffix)) = arch_compute_capability_and_suffix(arch) else {
        return false;
    };
    if !is_known_cuda_target(capability, suffix) {
        return false;
    }

    match hardware {
        GeneratedHardwareTarget::All => true,
        GeneratedHardwareTarget::AnyOf(alternatives) => alternatives.iter().any(|alternative| {
            generated_hardware_alternative_satisfied(capability, suffix, *alternative)
        }),
        GeneratedHardwareTarget::TargetMatrix { contracts } => contracts.iter().any(|contract| {
            contract.alternatives.iter().any(|alternative| {
                generated_hardware_alternative_satisfied(capability, suffix, alternative.hardware)
            })
        }),
    }
}

fn generated_contract_satisfied(arch: &str, contract: &GeneratedTargetContract) -> bool {
    let Some((capability, suffix)) = arch_compute_capability_and_suffix(arch) else {
        return false;
    };
    is_known_cuda_target(capability, suffix)
        && contract.alternatives.iter().any(|alternative| {
            generated_hardware_alternative_satisfied(capability, suffix, alternative.hardware)
        })
}

pub(super) fn for_each_resolved_hardware_alternative(
    requirement: GeneratedResolvedRequirement,
    mut visit: impl FnMut(GeneratedHardwareAlternative),
) {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => match requirement.hardware {
            GeneratedHardwareTarget::All => {}
            GeneratedHardwareTarget::AnyOf(alternatives) => {
                for alternative in alternatives {
                    visit(*alternative);
                }
            }
            GeneratedHardwareTarget::TargetMatrix { contracts } => {
                for contract in contracts {
                    for alternative in contract.alternatives {
                        visit(alternative.hardware);
                    }
                }
            }
        },
        GeneratedResolvedRequirement::Contract(contract) => {
            for alternative in contract.alternatives {
                visit(alternative.hardware);
            }
        }
    }
}

pub(super) fn generated_hardware_candidate(alternative: GeneratedHardwareAlternative) -> String {
    match alternative {
        GeneratedHardwareAlternative::MinimumSm(capability) => format!("sm_{capability}"),
        GeneratedHardwareAlternative::ExactArchitecture(capability) => {
            format!("sm_{capability}a")
        }
        GeneratedHardwareAlternative::FamilyTarget(capability) => format!("sm_{capability}f"),
    }
}

fn generated_hardware_requirement_label(alternative: GeneratedHardwareAlternative) -> String {
    match alternative {
        GeneratedHardwareAlternative::MinimumSm(capability) => {
            format!("sm_{capability} or newer")
        }
        GeneratedHardwareAlternative::ExactArchitecture(capability) => {
            format!("sm_{capability}a exactly")
        }
        GeneratedHardwareAlternative::FamilyTarget(capability) => {
            format!("sm_{capability}f exactly")
        }
    }
}

fn generated_hardware_alternative_satisfied(
    capability: u32,
    suffix: Option<char>,
    alternative: GeneratedHardwareAlternative,
) -> bool {
    match alternative {
        GeneratedHardwareAlternative::MinimumSm(minimum) => capability >= u32::from(minimum),
        GeneratedHardwareAlternative::ExactArchitecture(exact) => {
            capability == u32::from(exact) && suffix == Some('a')
        }
        // Family targets match only the named `sm_Nf` spelling.
        GeneratedHardwareAlternative::FamilyTarget(family) => {
            capability == u32::from(family) && suffix == Some('f')
        }
    }
}

pub(super) fn generated_requirement_ptx_floor(
    arch: &str,
    requirement: GeneratedTargetRequirement,
) -> Option<u16> {
    let (capability, suffix) = arch_compute_capability_and_suffix(arch)?;
    if !is_known_cuda_target(capability, suffix) {
        return None;
    }
    match requirement.hardware {
        GeneratedHardwareTarget::All => Some(requirement.minimum_ptx.encoded()),
        GeneratedHardwareTarget::AnyOf(alternatives) => alternatives
            .iter()
            .any(|alternative| {
                generated_hardware_alternative_satisfied(capability, suffix, *alternative)
            })
            .then(|| requirement.minimum_ptx.encoded()),
        GeneratedHardwareTarget::TargetMatrix { contracts } => contracts
            .iter()
            .flat_map(|contract| contract.alternatives.iter())
            .filter(|alternative| {
                generated_hardware_alternative_satisfied(capability, suffix, alternative.hardware)
            })
            .map(|alternative| alternative.minimum_ptx.encoded())
            .min(),
    }
}

fn resolved_requirement_ptx_floor(
    arch: &str,
    requirement: GeneratedResolvedRequirement,
) -> Option<u16> {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => {
            generated_requirement_ptx_floor(arch, requirement)
        }
        GeneratedResolvedRequirement::Contract(contract) => {
            let (capability, suffix) = arch_compute_capability_and_suffix(arch)?;
            is_known_cuda_target(capability, suffix).then_some(())?;
            contract
                .alternatives
                .iter()
                .filter(|alternative| {
                    generated_hardware_alternative_satisfied(
                        capability,
                        suffix,
                        alternative.hardware,
                    )
                })
                .map(|alternative| alternative.minimum_ptx.encoded())
                .min()
        }
    }
}

pub(crate) fn validate_generated_target(
    arch: &str,
    generated: &GeneratedModuleRequirements,
) -> Result<(), String> {
    for resolved in generated.resolved_targets() {
        let requirement = resolved_requirement(generated, resolved)?;
        if !resolved_hardware_satisfied(arch, requirement) {
            return Err(format!(
                "CUDA target {arch} cannot lower generated intrinsic `{}` (`{}`); requires {}",
                resolved.target.id,
                resolved.target.marker,
                describe_resolved_requirement(requirement)
            ));
        }
        ptx_isa_requirement_for_floor(
            resolved_requirement_ptx_floor(arch, requirement).unwrap(),
            resolved.target.id,
            resolved.target.marker,
        )?;
    }
    Ok(())
}

pub(super) fn describe_generated_hardware(hardware: GeneratedHardwareTarget) -> String {
    match hardware {
        GeneratedHardwareTarget::All => "any supported CUDA target".to_string(),
        GeneratedHardwareTarget::AnyOf(alternatives) => alternatives
            .iter()
            .map(|alternative| generated_hardware_requirement_label(*alternative))
            .collect::<Vec<_>>()
            .join(" or "),
        GeneratedHardwareTarget::TargetMatrix { contracts } => contracts
            .iter()
            .map(describe_generated_contract)
            .collect::<Vec<_>>()
            .join(" or "),
    }
}

fn describe_resolved_requirement(requirement: GeneratedResolvedRequirement) -> String {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => {
            describe_generated_hardware(requirement.hardware)
        }
        GeneratedResolvedRequirement::Contract(contract) => describe_generated_contract(contract),
    }
}

fn describe_generated_contract(contract: &GeneratedTargetContract) -> String {
    let alternatives = contract
        .alternatives
        .iter()
        .map(|alternative| {
            format!(
                "{} at PTX {}.{}",
                generated_hardware_requirement_label(alternative.hardware),
                alternative.minimum_ptx.major(),
                alternative.minimum_ptx.minor()
            )
        })
        .collect::<Vec<_>>()
        .join(" or ");
    if contract.selectors.is_empty() {
        alternatives
    } else {
        let selectors = contract
            .selectors
            .iter()
            .map(|selector| format!("{}={}", selector.name, selector.value))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{alternatives} for {selectors}")
    }
}
