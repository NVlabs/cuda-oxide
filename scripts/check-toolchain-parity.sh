#!/usr/bin/env bash
# Verify every copy of the toolchain pin still agrees with rust-toolchain.toml.
#
# The pin is duplicated three ways, and nothing else checks that the copies
# move together:
#
#   1. crates/rustc-codegen-cuda/rust-toolchain.toml.  That crate carries its
#      own [workspace] for the rustc_private dylibs, so rustup resolves it
#      against this file rather than the repo root's, and its own header says
#      "Must match the parent cuda-oxide toolchain exactly!".  A backend built
#      against a different nightly than the driver fails to load, so the two
#      disagreeing is a build break rather than a style nit -- but the header
#      is a comment, and a comment enforces nothing.
#
#   2. The `[toolchain]` blocks quoted in the book.  These are presented as
#      the repo's actual file -- one is even labelled "already in the repo
#      root" -- so a reader copies them into their own project.  When they go
#      stale the book hands out a pin that silently omits a component, and the
#      symptom lands later and elsewhere: a missing `rustfmt` surfaces as
#      `cargo oxide fmt` failing, not as a bad toolchain file.
#
# This is exactly what happened: #727 added `rustfmt` to both real files and
# left both book quotes at five components.
#
# Channel and components only.  Anything else in a [toolchain] block (a
# `targets` list, `profile`) is free to differ between the copies, and the
# book quotes elide freely -- the check is that what a copy *does* state about
# these two keys is right, not that it restates everything.
#
# Run this after bumping the pin or adding a component.
set -euo pipefail

# Both python and the comparisons below are byte-wise; pin the locale so an
# ambient UTF-8 one cannot reorder a components list under dictionary
# collation and turn an agreeing pair into a spurious failure.
export LC_ALL=C

cd "$(dirname "$0")/.."

ROOT_PIN=rust-toolchain.toml
NESTED_PIN=crates/rustc-codegen-cuda/rust-toolchain.toml

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the toolchain pin" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -s "${ROOT_PIN}"
test -s "${NESTED_PIN}"

python3 - "${ROOT_PIN}" "${NESTED_PIN}" <<'PY'
import glob
import re
import sys

root_path, nested_path = sys.argv[1], sys.argv[2]

CHANNEL = re.compile(r'^\s*channel\s*=\s*"([^"]+)"', re.M)
# Both layouts are in the tree: the root file spreads the list over one entry
# per line, the nested one keeps it on a single line.  Match the whole
# bracketed span and pull the quoted names out of it, so neither layout needs
# its own pattern and reflowing a list never breaks this guard.
COMPONENTS = re.compile(r"^\s*components\s*=\s*\[(.*?)\]", re.M | re.S)


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def pin(text, source):
    """(channel, components) from a [toolchain] block; None where unstated."""
    channel = CHANNEL.search(text)
    components = COMPONENTS.search(text)
    return (
        channel.group(1) if channel else None,
        re.findall(r'"([^"]+)"', components.group(1)) if components else None,
    )


root_channel, root_components = pin(read(root_path), root_path)

# Parse self-test.  A guard whose failure mode is "matched nothing" has to
# prove it still reads its reference before a clean result means anything.
if not root_channel or not root_components:
    sys.exit(
        f"parse self-test failed: read channel={root_channel!r} "
        f"components={root_components!r} from {root_path}; "
        "the file layout changed, fix this script before trusting it"
    )

failures = []

nested_channel, nested_components = pin(read(nested_path), nested_path)
if nested_channel != root_channel:
    failures.append(
        f"{nested_path} pins channel {nested_channel!r}, "
        f"{root_path} pins {root_channel!r}"
    )
if nested_components != root_components:
    failures.append(
        f"{nested_path} lists components {nested_components!r}, "
        f"{root_path} lists {root_components!r}"
    )

# The book's quoted blocks.  Only fenced ```toml blocks that actually contain
# a [toolchain] header are candidates: prose that merely names the file, and
# the `rustup component add` command lines (which are deliberately not
# exhaustive -- rustup installs whatever the pin names), are not quotes of it.
docs = sorted(glob.glob("cuda-oxide-book/**/*.md", recursive=True))
if len(docs) < 20:
    sys.exit(f"parse self-test failed: found {len(docs)} book pages")

TOML_BLOCK = re.compile(r"```toml\n(.*?)```", re.S)
quoted = 0
for path in docs:
    for block in TOML_BLOCK.findall(read(path)):
        if "[toolchain]" not in block:
            continue
        quoted += 1
        channel, components = pin(block, path)
        if channel is not None and channel != root_channel:
            failures.append(
                f"{path} quotes channel {channel!r}, {root_path} pins {root_channel!r}"
            )
        if components is not None and components != root_components:
            missing = [c for c in root_components if c not in components]
            extra = [c for c in components if c not in root_components]
            detail = ", ".join(
                part
                for part in (
                    "missing " + " ".join(missing) if missing else "",
                    "unknown " + " ".join(extra) if extra else "",
                )
                if part
            )
            failures.append(
                f"{path} quotes components {components!r}"
                + (f" ({detail})" if detail else "")
                + f"; {root_path} lists {root_components!r}"
            )

if not quoted:
    sys.exit(
        "parse self-test failed: no [toolchain] block found in any book page; "
        "either the book stopped quoting the pin (delete that half of this "
        "guard) or the fence style changed (fix the pattern)"
    )

if failures:
    print("error: the toolchain pin is not consistent across its copies", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        f"Every copy must state the same channel and components as {root_path}. "
        "The nested\npin is required to match exactly (rustc_private dylibs); "
        "the book's blocks are\npresented to readers as the repo's real file.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"OK: {root_path} pins {root_channel} with {len(root_components)} components, "
    f"and the nested pin plus all {quoted} block(s) quoted in the book agree."
)
PY
