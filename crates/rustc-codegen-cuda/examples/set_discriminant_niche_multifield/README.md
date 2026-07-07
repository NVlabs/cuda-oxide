# `set_discriminant_niche_multifield`

Positive test: `SetDiscriminant` on a niche-encoded enum whose **data variant
has multiple fields**, with the niche in a **later** field.

```rust
enum Multi { Nothing, Something(u32, NonZeroU32) }
```

`Something`'s second field (`NonZeroU32`) provides the niche, so rustc stores
`Nothing` as `Something` with that field `== 0` — no separate tag. Setting the
niche variant must write the niche bit pattern into the **second** payload
field, not the first `u32`. This is the case where the old single-field-chain
locator silently wrote the niche into the wrong (first) field.

Each of 64 threads builds `Something(7, NonZeroU32::new(42))`, emits
`StatementKind::SetDiscriminant(*e, 0)` via custom MIR, and checks the observed
variant flipped to `Nothing`. Output `1` = pass.

## Scope

The device reads the live variant from cuda-oxide's synthetic discriminant tag,
and a niche field value (`NonZeroU32 == 0`) is invalid and cannot be read back
without UB — so this example verifies the multi-field niche enum **lowers,
loads, and runs** end-to-end (exercising the importer's `niche_field_location`
on a real multi-field layout). The **field-precise** correctness (niche lands in
slot 1, not slot 0) is pinned by the mir-lower unit test
`convert_set_discriminant_niche_targets_correct_multifield_slot`.

## Usage

```bash
cargo oxide run set_discriminant_niche_multifield
```
