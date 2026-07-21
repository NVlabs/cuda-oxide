# `set_discriminant_niche`

This positive test proves that niche-encoded enums have the same bytes on the
host and device.

It covers five niche shapes, a direct-tagged `bool` payload, and one signed
direct-tag shape:

- `Option<NonZeroU32>`: `0` means `None` in one `u32`.
- `Option<bool>`: normal booleans use `0` and `1`, while `2` means `None` in
  the one-byte memory representation. This is important because ordinary
  `bool` values use an LLVM `i1`, but the enum carrier is an `i8`.
- `enum E { A, B(bool), C }`: `B(false)` and `B(true)` are the ordinary,
  untagged payload values `0` and `1`. `A` uses invalid bool value `2`, the
  range position for untagged `B` (`3`) is skipped, and `C` uses `4`. This
  checks the non-obvious case where the untagged `B` index lies inside rustc's
  overall `A..=C` niche-variant range.
- `Option<&u32>`: `SetDiscriminant(None)` must write a null generic pointer,
  preserving pointer provenance instead of inventing an integer tag.
- `MaybeWrapper`: the value used to distinguish `None` is a nested
  `NonZeroU32` at byte 4, after an ordinary `u32` field.
- `DirectBoolean`: a non-overlapping `bool` payload still occupies one complete
  byte after its four-byte direct tag. The lowered aggregate must use `i8`
  storage there, with explicit `i1` conversions at the value boundary.
- `Negative`: its `-1` discriminant must remain `-1` when widened from `i8`
  to `i32`, rather than becoming `255`.

The host creates both `None` and `Some` values plus every `A/B(false)/B(true)/C`
case. The GPU reads them, records their payloads, constructs new `Option<bool>`,
`E`, and direct-tagged `bool` values, and uses MIR `SetDiscriminant` to change
the input options to `None`. A device-local `Option<&u32>` separately verifies
the null-pointer write. The host then checks the recorded values and all
returned enums.

The compile-only probe also imports zero-variant, one-variant, and
multiple-source-variant enums whose physical layout is Empty or Single. This
ensures an uninhabited enum is rejected semantically without making the
compiler assume that an Empty layout has no source variant metadata.

Small example:

```text
MaybeWrapper::Some({ pad: 7, nz: 9 })
                         byte 0  byte 4
SetDiscriminant(None) changes the carrier at byte 4 to 0.
```

Run it with:

```bash
cargo oxide run set_discriminant_niche
```
