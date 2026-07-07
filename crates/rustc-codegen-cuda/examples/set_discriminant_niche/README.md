# `set_discriminant_niche`

Positive test: niche-encoded enum discriminant writes via `SetDiscriminant`
are lowered correctly on the device.

`Option<NonZeroU32>` is niche-encoded: Rust stores it as a single `u32`
where the value `0` means `None` and any non-zero value means `Some`. Our
pipeline keeps a synthetic `{ discriminant, payload }` representation for
such enums inside kernels. This example verifies that
`StatementKind::SetDiscriminant` to the niche variant (`None`, variant index 0)
writes both
the synthetic discriminant and the payload niche value (`0`).

Usage:
  cargo oxide run set_discriminant_niche
