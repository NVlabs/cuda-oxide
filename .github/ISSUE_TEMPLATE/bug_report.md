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
Paste the output of `cargo oxide doctor`. It reports the GPU and driver, the
toolchain and its components, the resolved `llc`, libNVVM, nvJitLink and
libdevice, and the backend it would load -- which is most of what a triage
question would otherwise ask for one field at a time.

<details><summary><code>cargo oxide doctor</code></summary>

```text

```

</details>

If `doctor` itself is what fails, the individual facts still help:
- GPU: <!-- e.g. RTX 4090, H100 -->
- CUDA driver version:
- `rustc --version --verbose`:
- `llc --version` (or `llc-22 --version`):

**Additional context**
Attach `.ll` / `.ptx` files if the pipeline produces them before failing.

---
> Not sure if this is a bug? Ask in [#help on Discord](https://discord.gg/ZUEr4AhH5C) first.
