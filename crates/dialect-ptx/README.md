# dialect-ptx

`dialect-ptx` is CUDA Oxide's structured terminal PTX dialect. Its operation
tree can be constructed directly with `PtxBuilder`, emitted deterministically
with the dedicated PTX emitter, or projected from the lossless CST in
`ptx-syntax`.

The two representations and writers have distinct authority:

- `ptx-syntax::Document` owns exact external source, trivia, unknown syntax,
  and byte-preserving edits.
- `dialect-ptx` owns canonical structure for analysis, construction,
  transformation, and deterministic emission.
- `EditScript::apply_with_map` is the lossless patch path. It preserves all
  untouched bytes and returns original/normalized byte lineage.
- `emit_canonical_module` is the constructed/transformed-IR path. It verifies
  native CFG invariants and may normalize the complete module's formatting.

When operations originate in source, `Projection` keeps statement/scope and
byte-span lineage in a side table. Source lineage is not a required operation
attribute, so generated operations never need synthetic source locations.

One `ptx.callable` identity owns either a single-block `ptx.surface_body` or a
multi-block `ptx.cfg_body`; raising changes the body form without duplicating
callable identity or header attributes. Native indexed-branch tables derive
their emitted targets from ordered CFG successor slots, while fallthrough is
accepted only when it names the next emitted block.

The dialect currently models module and lexical scopes, callable declarations
and definitions, directives, labels, generic instructions, and a raw escape hatch.
Typed ISA operations can be added incrementally without making the lossless
parser reject newer PTX spellings.

`Projection::control_flow` recovers a conservative intraprocedural CFG for
direct and indexed branches, predicated fallthrough, and terminal instructions.
It retains CST statement/scope lineage and fails closed for unsupported PTX
versions or unresolved targets instead of attaching guessed successors to the
operation tree.
