# dialect-ptx

`dialect-ptx` is CUDA Oxide's structured terminal PTX dialect. Its operation
tree can be constructed directly with `PtxBuilder`, emitted deterministically
with the dedicated PTX emitter, or projected from the lossless CST in
`ptx-parse`.

The two representations have distinct authority:

- `ptx-parse::Document` owns exact external source, trivia, unknown syntax,
  and byte-preserving edits.
- `dialect-ptx` owns canonical structure for analysis, construction,
  transformation, and deterministic emission.

When operations originate in source, `Projection` keeps statement/scope and
byte-span lineage in a side table. Source lineage is not a required operation
attribute, so generated operations never need synthetic source locations.

The dialect currently models module and lexical scopes, callable declarations
and definitions, directives, labels, generic instructions, and a raw escape hatch.
Typed ISA operations can be added incrementally without making the lossless
parser reject newer PTX spellings.
