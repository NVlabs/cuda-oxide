# error_set_discriminant_niche

Negative test for the niche-encoded half of `SetDiscriminant`.

A *niche* is a payload bit pattern Rust normally forbids and reuses as the
variant marker. `NonZeroU32` cannot contain zero, so `Option<NonZeroU32>` uses
zero for `None` and has no separate tag:

```text
Some(7): [payload = 7]
None:    [payload = 0]  <- niche value, not a tag field
```

Until cuda-oxide implements that payload rewrite, the compiler must reject
the operation instead of writing its device-private synthetic tag.

## Usage

```bash
cargo oxide run error_set_discriminant_niche
```

The build must fail with a message containing:

```text
SetDiscriminant for niche-encoded enums is not yet supported
```

This is a support-gap fixture for the niche portion of issue #306.
