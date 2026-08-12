#!/usr/bin/env bash
# Verify every device-API call the book shows in a Rust code block resolves to a
# function `cuda-device` actually exports.
#
# The failure this catches is silent and has now happened three times. #797
# found the intrinsics guide pointing at op files that no longer existed; the
# same sweep later found `warp::shuffle_xor_i32` in the API quick reference,
# advertised as an "i32 variant" that has never existed, and a dispatch helper
# in the compiler pages that had been replaced by generated code. Nothing fails
# when the book names a function the tree does not have: it renders, it builds,
# and a reader finds out by pasting it.
#
# Scope, chosen so the guard is precise rather than merely broad:
#
#   * Only ```rust fenced blocks. A name in prose may legitimately be a PTX
#     instruction (`shfl.sync.bfly`), a module path, or a function from another
#     project; inside a Rust block a `warp::foo(...)` is a call and has to exist.
#     Restricting to code blocks is what takes this from dozens of false
#     positives to none.
#   * Only the device modules -- `warp`, `thread`, `grid`, `cluster`. Those are
#     the paths the book uses in kernel examples and the ones that rot. Host
#     APIs are checked by rustdoc, which builds under `-D warnings`.
#   * Existence only, never arity or types. Those change for good reasons and
#     the compiler catches them; a name that is simply absent is the silent case.
#
# Run this after renaming or removing anything in `cuda-device`.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

BOOK=cuda-oxide-book
DEVICE=crates/cuda-device/src

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the book's API names" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -d "${BOOK}"
test -d "${DEVICE}"

python3 - "${BOOK}" "${DEVICE}" <<'PY'
import glob
import os
import re
import sys

book_root, device_root = sys.argv[1], sys.argv[2]

RUST_BLOCK = re.compile(r"```rust[^\n]*\n(.*?)```", re.S)
# `mod::name(` inside a code block: a call, so the name must exist.
#
# The lookbehind is load-bearing. `warp` and `thread` are also module names deep
# inside the compiler -- the intrinsics guide legitimately shows
# `intrinsics::warp::emit_two_operand_intrinsic(...)`, a mir-importer path that
# has no business resolving against cuda-device. Rejecting a `::` immediately
# before the module keeps this to the device paths a kernel actually calls,
# while an explicit `cuda_device::` prefix stays accepted.
CALL = re.compile(
    r"(?<!::)\b(?:cuda_device::)?(warp|thread|grid|cluster)::([a-z_][a-z0-9_]*)\s*\("
)
EXPORTED = re.compile(r"pub (?:unsafe )?fn ([a-z_][a-z0-9_]*)")

pages = sorted(glob.glob(os.path.join(book_root, "**", "*.md"), recursive=True))
if len(pages) < 20:
    sys.exit(f"parse self-test failed: found {len(pages)} book pages under {book_root}")

calls = {}
blocks = 0
for page in pages:
    with open(page, encoding="utf-8") as handle:
        text = handle.read()
    for block in RUST_BLOCK.findall(text):
        blocks += 1
        for match in CALL.finditer(block):
            calls.setdefault((match.group(1), match.group(2)), set()).add(page)

if blocks < 20:
    sys.exit(f"parse self-test failed: read {blocks} rust code blocks from the book")

exported = set()
sources = sorted(glob.glob(os.path.join(device_root, "**", "*.rs"), recursive=True))
for source in sources:
    with open(source, encoding="utf-8") as handle:
        exported |= set(EXPORTED.findall(handle.read()))

# The other half of the silent-blindness guard: if the export scan rotted, every
# name would look missing rather than every name looking fine, but say so
# explicitly rather than dumping hundreds of failures.
if len(exported) < 200:
    sys.exit(
        f"parse self-test failed: read {len(exported)} exported fns from "
        f"{len(sources)} files under {device_root}"
    )

missing = sorted(
    (module, name, sorted(where))
    for (module, name), where in calls.items()
    if name not in exported
)

if missing:
    print(
        "error: the book calls device functions that cuda-device does not export:",
        file=sys.stderr,
    )
    for module, name, where in missing:
        pages_text = ", ".join(os.path.relpath(p, book_root) for p in where)
        print(f"  {module}::{name}   in {pages_text}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        "A Rust code block is something a reader pastes. Either the function was\n"
        "renamed and the book missed it, or the example was written from memory.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"OK: all {len(calls)} device-API calls in the book's {blocks} Rust blocks "
    f"resolve against cuda-device's {len(exported)} exported functions."
)
PY
