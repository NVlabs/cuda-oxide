# ptx-schedule

Structural PTX schedule analysis and deterministic perturbation.

A kernel that races only under one interleaving passes every run until it does
not. This crate makes that interleaving reachable on purpose: it finds the
points in a PTX module where thread progress is observable, then inserts a
seeded `nanosleep.u32` at a subset of them so the same seed always produces the
same schedule.

## What it deliberately does not do

It owns neither CUDA execution nor an input generator. It reads PTX text and
writes PTX text. Building, launching, watchdogging and verdict assignment are
the campaign driver's job, and the kernel inputs stay whatever the example
already uses. Static site discovery, mutation and triage therefore all read
the same source model, so a site named in a finding is the site the analyzer
found.

## The model

`analyze_ptx` turns a module into an ordered list of `ScheduleSite`s -- the
places where another thread's progress can be observed:

| `SiteKind` | what it marks |
|:-----------|:--------------|
| `Atomic` | an atomic read-modify-write |
| `Reduction` | a `red.*` reduction with no returned value |
| `Barrier` | `bar.*` / `barrier.*` synchronization |
| `Fence` | `fence.*` / `membar.*` ordering |
| `OrderedMemory` | a load or store carrying an explicit ordering or scope |
| `WarpCollective` | `shfl`, `vote`, `match`, `redux` and friends |
| `Backedge` | a branch back to a dominating label, i.e. a loop |

Each site keeps its ordinal, enclosing callable, byte span, basic block, the
instruction text and any guarding predicate, so a rewrite can be described
without re-parsing.

`perturb_ptx` then applies `InjectionOptions` -- `seed`, `intensity` (the
fraction of sites to touch), `max_sleep_ns` (default
`DEFAULT_MAX_SLEEP_NS`, 64 µs) and an optional `focus` substring to restrict
the search to one callable -- and returns the rewritten module plus an
`InjectionDecision` per site. Nothing is inserted when the intensity selects
no site, so a seed that changes nothing is reported rather than silently
producing the original text.

## Structure

```text
src/
├── lib.rs       # site discovery (analyze_ptx) and injection (perturb_ptx)
├── campaign.rs  # the seed-sweep driver: build, run, watchdog, confirm
└── main.rs      # single-file CLI over one .ptx
```

## Campaign verdicts

`campaign::run_campaign` sweeps a seed range and classifies each run. A
finding is re-run (`confirm_runs`) before it is reported, so a one-off is not
mistaken for a reproducible schedule bug:

| `RunKind` | meaning |
|:----------|:--------|
| `Pass` | the perturbed build behaved like the baseline |
| `Skipped` | the seed inserted nothing, or the example declined to run |
| `Hang` | the watchdog fired |
| `Crash` | the process died |
| `Mismatch` | the example reported its own failure |
| `OutputChanged` | stdout differed with no explicit failure marker (opt-in) |
| `GpuWedged` | the device stopped responding |
| `HarnessError` | the campaign itself failed, not the kernel |

## Consumers

| Crate | Uses it for |
|:------|:------------|
| `cargo-oxide` | `cargo oxide fuzz-schedule <example>`, the user-facing campaign |

The crate also ships a `ptx-schedule` binary for one PTX file at a time:

```bash
ptx-schedule kernel.ptx --list-sites
ptx-schedule kernel.ptx --seed 7 --intensity 0.5 -o perturbed.ptx \
    --decisions-json decisions.json
```

## License

Apache-2.0. See [LICENSE](https://github.com/NVlabs/cuda-oxide/blob/main/LICENSE).
