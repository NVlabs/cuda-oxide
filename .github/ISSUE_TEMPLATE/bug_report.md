---
name: Bug report
about: Something compiled wrong, crashed, or produced bad PTX
labels: bug, TBD
assignees: ''
---

**Description**
A clear, one-paragraph description of the bug.

**Minimal reproducer**
Paste the smallest kernel + host code that triggers the issue.

```rust
// kernel
```

**Expected behavior**
What should happen.

**Actual behavior**
What actually happens (error message, wrong output, panic, etc.).

**Environment**
- GPU: <!-- e.g. RTX 4090, H100 -->
- CUDA driver version:
- `rustc --version --verbose`:
- `llc --version` (or `llc-22 --version`):

**Additional context**
Attach `.ll` / `.ptx` files if the pipeline produces them before failing.

---
> Not sure if this is a bug? Ask in [#help on Discord](https://discord.gg/ZUEr4AhH5C) first.
