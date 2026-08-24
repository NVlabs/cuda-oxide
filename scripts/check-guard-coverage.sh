#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every guard script is actually executed by both things that are
# supposed to execute it: a pull-request workflow, and `just check`.
#
# The Justfile states that as a contract -- "`check-guards` covers the
# status-guard, naming-guard and cargo-deny workflows in full" -- and nothing
# enforced it. `check-book-catalog-stamp.sh` landed in #1106 wired into
# status-guard.yml and never added to `check-guards`, so twelve of the thirteen
# guards ran locally and "in full" was false.
#
# The first attempt at this guard searched both files for the script's name.
# That is not enforcement evidence, and #1133's review gave the counterexamples:
# the same text inside a `run:` block that only echoes it, a step behind
# `if: false`, or a reusable workflow nothing calls. All three left the search
# green while no pull request executed the guard. So this reads structure, not
# text:
#
#   * workflow YAML is parsed, and reachability is computed as a graph -- a
#     workflow is an entry point only if it triggers on `pull_request`, a
#     reusable workflow counts only if a reachable job `uses:` it, a job or step
#     proven `if: false` is dropped, and anything that may `continue-on-error`
#     does not enforce;
#   * a `run:` script is split into commands, and a guard counts only in command
#     position -- `echo "bash scripts/x.sh"` runs `echo`, not the guard;
#   * the Justfile gets the same treatment: recipes and their dependencies are a
#     graph walked from `check`, a body line prefixed with `-` does not enforce,
#     and the same command-position rule applies.
#
# The walk, in one picture:
#
#   .github/workflows/*.yml            Justfile
#          |                              |
#   on: pull_request ?              recipe graph from `check`
#          | yes                         | reachable
#   job.if false ? -----> dropped   line starts `-` ? --> dropped
#   job.continue-on-error ? -> dropped   |
#          |                             |
#   uses: ./...yml --> +reusable         |
#          |            (needs workflow_call)
#   step.if / continue-on-error ? -> dropped
#          |                             |
#      run: script ------------------> commands
#          |                             |
#     command position? ------------ command position?
#          |                             |
#       in_ci  <---- compare both ways ----> in_just
#                   vs scripts/check-*.sh
#
# Reachability is deliberately conservative in one direction only. An `if:` that
# is not literally false is treated as reachable, because `github.event_name ==
# 'pull_request'` cannot be evaluated here; the guard therefore never claims a
# step is dead when it might run. It is strict the other way: a step that may be
# skipped or may swallow its failure is not credited as coverage.
#
# `smoketest.sh` and `debug-smoketest.sh` are not guards -- the first is the
# example suite with its own CI job and `just smoketest` recipe, the second a
# developer tool. Only the `check-*` family is in scope.
#
# Needs python3 with PyYAML. Structural parsing is the requirement, and a
# hand-rolled YAML reader would be the textual approximation this replaces.
# No cargo, no network, no toolchain.
#
# Run this after adding, renaming or removing a guard script, or after editing a
# workflow step or the `check-guards` recipe.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

JUSTFILE=Justfile
WORKFLOWS=.github/workflows
ENTRY_RECIPE=check
GUARD_RECIPE=check-guards

test -s "${JUSTFILE}"
test -d "${WORKFLOWS}"

# Only python3 is required. The YAML this needs is parsed by a strict subset
# reader in the script itself, for two reasons: PyYAML is *not* in the
# `actions/runner-images` preinstalled manifest, so depending on it would risk
# the job failing on merge; and no other guard in this family needs a
# third-party package, on a repo that already goes out of its way for macOS
# contributors (see check-error-example-status.sh). When PyYAML *is* importable
# the script uses it as an oracle and refuses on any disagreement, so the
# subset reader is checked rather than trusted.
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to parse the workflows" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi
PYTHON=python3

"${PYTHON}" - "${WORKFLOWS}" "${JUSTFILE}" "${ENTRY_RECIPE}" "${GUARD_RECIPE}" <<'PYEOF'
import os
import re
import shlex
import sys

workflows_dir, justfile_path, entry_recipe, guard_recipe = sys.argv[1:5]

# ---------------------------------------------------------------------------
# A strict-subset YAML reader.
#
# Only the constructs these workflows actually use: block mappings, block
# sequences, plain and quoted scalars, literal (`|`) and folded (`>`) block
# scalars with chomping indicators, flow sequences of scalars, comments, and one
# leading document marker. Anything else -- a tab, an anchor, an alias, a tag, a
# flow mapping, a second document -- is a refusal, never a guess. That is what
# separates this from the text search it replaces: it cannot silently misread,
# only decline to read.
#
# Bare `on`, `true` and `false` resolve to booleans, as YAML 1.1 requires and as
# PyYAML does, because the whole reachability walk depends on it: `on:` written
# unquoted is the key `True`, not `"on"`, in every one of these files.
# ---------------------------------------------------------------------------

BOOLEAN_TRUE = {"y", "Y", "yes", "Yes", "YES", "true", "True", "TRUE", "on", "On", "ON"}
BOOLEAN_FALSE = {"n", "N", "no", "No", "NO", "false", "False", "FALSE", "off", "Off", "OFF"}
NULL = {"", "~", "null", "Null", "NULL"}
BLOCK_SCALAR = re.compile(r"^([|>])([-+]?)$")


class YamlSubsetError(Exception):
    pass


def _strip_comment(text):
    """Drop a trailing comment, respecting quotes."""
    quote = None
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or text[index - 1] in " \t"):
            return text[:index]
    return text


def _scalar(text):
    text = _strip_comment(text).strip()
    if text.startswith("&"):
        raise YamlSubsetError("anchors are not supported")
    if text.startswith("*"):
        raise YamlSubsetError("aliases are not supported")
    if text.startswith("!"):
        raise YamlSubsetError("tags are not supported")
    if text.startswith("{"):
        raise YamlSubsetError("flow mappings are not supported")
    if text.startswith("["):
        if not text.endswith("]"):
            raise YamlSubsetError("multi-line flow sequences are not supported")
        inner = text[1:-1].strip()
        return [_scalar(item) for item in inner.split(",")] if inner else []
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        body = text[1:-1]
        return body.replace("''", "'") if text[0] == "'" else body
    if text in NULL:
        return None
    if text in BOOLEAN_TRUE:
        return True
    if text in BOOLEAN_FALSE:
        return False
    return text


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _significant(line):
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _read_block_scalar(lines, index, parent_indent, style, chomp):
    """Consume a `|`/`>` block and return (text, next_index)."""
    body = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and _indent(line) <= parent_indent:
            break
        body.append(line)
        index += 1
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return ("", index)
    block_indent = min(_indent(line) for line in body if line.strip())
    body = [line[block_indent:] if line.strip() else "" for line in body]
    if style == "|":
        text = "\n".join(body)
    else:
        # Folded: a newline between two non-empty lines becomes a space; a blank
        # line becomes a newline. This matters -- book.yml folds one command
        # across two lines, and splitting it would invent a second command.
        folded, previous_blank = [], False
        for line in body:
            if not line:
                folded.append("\n")
                previous_blank = True
            else:
                if folded and not previous_blank:
                    folded.append(" ")
                folded.append(line)
                previous_blank = False
        text = "".join(folded)
    if chomp == "-":
        return (text, index)
    return (text + "\n", index)


def _parse_node(lines, index, minimum_indent):
    """Parse the block starting at `lines[index]`, at least `minimum_indent` deep.

    The caller knows only that a child block must be deeper than its key, not by
    how much, so the block's own indent is taken from its first significant line
    and every later line is measured against that.
    """
    while index < len(lines) and not _significant(lines[index]):
        index += 1
    if index >= len(lines) or _indent(lines[index]) < minimum_indent:
        return (None, index)
    indent = _indent(lines[index])

    if lines[index].strip().startswith("- "):
        items = []
        while index < len(lines):
            if not _significant(lines[index]):
                index += 1
                continue
            if _indent(lines[index]) < indent:
                break
            content = lines[index].strip()
            if not content.startswith("- "):
                break
            item_indent = _indent(lines[index]) + 2
            rest = content[2:]
            if ":" in rest and not rest.startswith(("'", '"')):
                # `- key: value` opens a mapping whose first key is inline.
                synthetic = [" " * item_indent + rest] + lines[index + 1 :]
                value, consumed = _parse_node(synthetic, 0, item_indent)
                items.append(value)
                index = index + 1 + (consumed - 1)
            else:
                items.append(_scalar(rest))
                index += 1
        return (items, index)

    mapping = {}
    while index < len(lines):
        if not _significant(lines[index]):
            index += 1
            continue
        line_indent = _indent(lines[index])
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YamlSubsetError(f"unexpected indentation on line {index + 1}")
        content = lines[index].strip()
        if content.startswith("- "):
            break
        match = re.match(r"^(\"[^\"]*\"|'[^']*'|[^:]+):(.*)$", content)
        if not match:
            raise YamlSubsetError(f"line {index + 1} is neither a mapping key nor a sequence item")
        key = _scalar(match.group(1))
        rest = _strip_comment(match.group(2)).strip()
        index += 1
        block = BLOCK_SCALAR.match(rest)
        if block:
            mapping[key], index = _read_block_scalar(
                lines, index, line_indent, block.group(1), block.group(2)
            )
        elif rest:
            mapping[key] = _scalar(rest)
        else:
            mapping[key], index = _parse_node(lines, index, line_indent + 1)
    return (mapping, index)


def parse_yaml_subset(text):
    if "\t" in text:
        raise YamlSubsetError("tabs are not valid YAML indentation")
    lines = text.split("\n")
    start = 0
    seen_marker = False
    while start < len(lines):
        stripped = lines[start].strip()
        if stripped == "---" and not seen_marker:
            seen_marker = True
            start += 1
            continue
        if stripped in ("---", "..."):
            raise YamlSubsetError("multi-document streams are not supported")
        if not _significant(lines[start]):
            start += 1
            continue
        break
    for line in lines[start:]:
        if line.strip() in ("---", "..."):
            raise YamlSubsetError("multi-document streams are not supported")
    value, _ = _parse_node(lines, start, _indent(lines[start]) if start < len(lines) else 0)
    return value


# ---------------------------------------------------------------------------
# Command-position matching, shared by both halves.
# ---------------------------------------------------------------------------

SEPARATORS = re.compile(r"&&|\|\||[;|&]")
# Leading `VAR=value` assignments, and `env`-style prefixes, are not the command.
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
WRAPPERS = {"bash", "sh", "dash", "zsh", "env", ".", "source", "exec", "time", "sudo"}


def commands(script):
    """Every command in a shell script, as a token list.

    Approximate on purpose: only the leading tokens are needed to decide whether
    a guard is being *executed*, and that is exactly what a text search cannot
    tell. A `#` line is a comment, and a quoted argument stays one token, so
    `echo "bash scripts/x.sh"` yields `['echo', 'bash scripts/x.sh']`.
    """
    pending = ""
    for raw in script.splitlines():
        line = raw.strip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line, pending = pending + line, ""
        if not line or line.startswith("#"):
            continue
        for segment in SEPARATORS.split(line):
            segment = segment.strip()
            if not segment:
                continue
            try:
                tokens = shlex.split(segment, comments=True)
            except ValueError:
                # Unbalanced quotes, or a `${{ }}` expression shlex dislikes.
                tokens = segment.split()
            if tokens:
                yield tokens


def invoked_recipes(script):
    """Just recipes this script runs, as `just <name>...`.

    A workflow step that runs `just check-guards` executes every guard in that
    recipe. Reading only the literal `bash scripts/...` lines would report all of
    them as absent from CI -- a false failure, and the wrong verdict. The
    Justfile is the authoritative list in that case, so the walk follows it
    rather than re-deriving one.
    """
    found = set()
    for tokens in commands(script):
        while tokens and (ASSIGNMENT.match(tokens[0]) or tokens[0] in {"-", "@"}):
            tokens = tokens[1:]
        while tokens and tokens[0] in WRAPPERS:
            tokens = tokens[1:]
        if not tokens or os.path.basename(tokens[0]) != "just":
            continue
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            if re.fullmatch(r"[a-zA-Z0-9_-]+", token):
                found.add(token)
    return found


def invoked_scripts(script):
    """Guard basenames this script executes, as opposed to merely mentions."""
    found = set()
    for tokens in commands(script):
        while tokens and (ASSIGNMENT.match(tokens[0]) or tokens[0] in {"-", "@"}):
            tokens = tokens[1:]
        while tokens and tokens[0] in WRAPPERS:
            tokens = tokens[1:]
        if not tokens:
            continue
        name = os.path.basename(tokens[0])
        if name.startswith("check-") and name.endswith(".sh"):
            found.add(name)
    return found


# ---------------------------------------------------------------------------
# Workflow graph.
# ---------------------------------------------------------------------------

PR_EVENTS = {"pull_request", "pull_request_target"}


def triggers(document):
    """The event names a workflow's `on:` declares.

    YAML 1.1 reads a bare `on` as the boolean true, which is why this looks up
    both keys: PyYAML hands back `{True: ...}` for `on:` written unquoted, as
    every workflow in this repo writes it.
    """
    section = document.get("on", document.get(True))
    if section is None:
        return set()
    if isinstance(section, str):
        return {section}
    if isinstance(section, list):
        return {str(event) for event in section}
    if isinstance(section, dict):
        return {str(event) for event in section}
    return set()


def statically_false(condition):
    """True only when an `if:` provably never runs.

    Anything else is treated as reachable. `github.event_name == 'pull_request'`
    cannot be evaluated here, and a guard that wrongly called a live step dead
    would be worse than one that misses a dead step.
    """
    if condition is None:
        return False
    if isinstance(condition, bool):
        return condition is False
    text = str(condition).strip()
    text = re.sub(r"^\$\{\{(.*)\}\}$", r"\1", text).strip().strip("'\"").lower()
    return text == "false"


def enforcing(node):
    """Whether a failure here fails the run.

    `continue-on-error` that is anything but a literal `false`/absent is not
    credited: an expression may evaluate true, and a step whose failure is
    swallowed is not enforcement.
    """
    value = node.get("continue-on-error")
    return value is None or value is False


def load_workflows(directory):
    documents = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith((".yml", ".yaml")):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        try:
            document = parse_yaml_subset(text)
        except YamlSubsetError as error:
            sys.exit(
                f"error: {path} uses YAML this guard does not read ({error}). It "
                "refuses rather than guessing; extend parse_yaml_subset."
            )
        if not isinstance(document, dict):
            sys.exit(f"error: {path} does not parse to a mapping")
        documents[name] = document
    return documents


def called_workflow(job):
    """The reusable workflow a job calls, if it calls one in this repo."""
    uses = job.get("uses")
    if not isinstance(uses, str):
        return None
    if not uses.startswith("./.github/workflows/"):
        return None
    return os.path.basename(uses.split("@", 1)[0])


def _projection(document):
    """Exactly the fields the reachability walk reads, and nothing else.

    Comparing this rather than the whole document keeps the oracle honest about
    what matters: if the subset reader agrees here, it is fit for this guard,
    and trivia like whether `timeout-minutes: 5` came back an int or a string
    cannot mask a real disagreement or manufacture a false one.
    """
    section = document.get("on", document.get(True))
    if isinstance(section, str):
        events = [section]
    elif isinstance(section, list):
        events = [str(event) for event in section]
    elif isinstance(section, dict):
        events = [str(event) for event in section]
    else:
        events = []
    jobs = {}
    for name, job in (document.get("jobs") or {}).items():
        if not isinstance(job, dict):
            jobs[str(name)] = None
            continue
        steps = [
            (step.get("if"), step.get("continue-on-error"), step.get("run"))
            for step in (job.get("steps") or [])
            if isinstance(step, dict)
        ]
        jobs[str(name)] = (
            job.get("if"),
            job.get("continue-on-error"),
            job.get("uses"),
            steps,
        )
    return (sorted(events), jobs)


def oracle_check(directory, parsed):
    """Cross-check the subset reader against PyYAML, when PyYAML is available.

    The reader is the requirement -- a third-party parser cannot be depended on,
    since PyYAML is absent from the runner-images manifest -- but where a
    reference implementation *is* installed there is no excuse for not using it.
    A disagreement is a refusal, not a warning.
    """
    try:
        import yaml
    except ImportError:
        return "not checked (PyYAML absent)"
    for name, document in parsed.items():
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as handle:
            try:
                reference = yaml.safe_load(handle)
            except yaml.YAMLError as error:
                sys.exit(f"error: PyYAML cannot parse {path}: {error}")
        mine, theirs = _projection(document), _projection(reference)
        if mine != theirs:
            sys.exit(
                f"error: the subset YAML reader disagrees with PyYAML on {path}.\n"
                f"       subset reader: {mine}\n"
                f"       PyYAML:        {theirs}\n"
                "       Fix parse_yaml_subset; a reader that misreads a workflow "
                "is worse than the text search this replaced."
            )
    return f"agrees with PyYAML on all {len(parsed)} workflows"


PR_EVENTS_NOTE = "pull_request / pull_request_target"


def reachable_workflow_set(documents):
    """Workflow files a pull request can reach, following `workflow_call`."""
    reachable = {name for name, doc in documents.items() if triggers(doc) & PR_EVENTS}
    while True:
        discovered = set()
        for name in reachable:
            for job in (documents[name].get("jobs") or {}).values():
                if not isinstance(job, dict) or statically_false(job.get("if")):
                    continue
                called = called_workflow(job)
                if called is None or called in reachable:
                    continue
                if called not in documents:
                    sys.exit(
                        f"error: {name} calls ./.github/workflows/{called}, which "
                        "does not exist"
                    )
                if "workflow_call" not in triggers(documents[called]):
                    sys.exit(
                        f"error: {name} calls {called}, which does not declare "
                        "`workflow_call`; it would never run"
                    )
                discovered.add(called)
        if not discovered:
            return reachable
        reachable |= discovered


def workflow_coverage(documents, recipes=None):
    """Guard -> "workflow:job" for every guard a pull request actually executes."""
    recipes = recipes or {}
    found = {}
    for name in sorted(reachable_workflow_set(documents)):
        for job_name, job in (documents[name].get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            if statically_false(job.get("if")) or not enforcing(job):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                if statically_false(step.get("if")) or not enforcing(step):
                    continue
                script = step.get("run")
                if not isinstance(script, str):
                    continue
                for guard in invoked_scripts(script):
                    found.setdefault(guard, f"{name}:{job_name}")
                for recipe in invoked_recipes(script):
                    reached = recipes_reachable_from(recipes, [recipe])
                    for guard in recipe_guards(recipes, reached):
                        found.setdefault(guard, f"{name}:{job_name} (just {recipe})")
    return found


RECIPE_HEADER = re.compile(r"^([a-zA-Z0-9_-]+)(?:\s+[^:]*)?:(.*)$")


def parse_recipes(text):
    """Recipe name -> (dependency names, body lines).

    A header starts in column one; the body is the indented block under it.
    The names after the colon are its dependencies, which is what makes the
    Justfile a graph rather than a list -- and what lets a guard be moved into a
    recipe `check` never reaches.
    """
    recipes = {}
    current = None
    for raw in text.splitlines():
        if raw.strip() and not raw[0].isspace():
            match = RECIPE_HEADER.match(raw)
            if match and not raw.lstrip().startswith("#"):
                name = match.group(1)
                dependencies = [
                    token
                    for token in match.group(2).split()
                    if re.fullmatch(r"[a-zA-Z0-9_-]+", token)
                ]
                recipes[name] = (dependencies, [])
                current = name
            else:
                current = None
            continue
        if current is not None and raw.strip():
            recipes[current][1].append(raw.strip())
    return recipes


def recipes_reachable_from(recipes, entries):
    """Recipes `just <entry>` runs, following dependencies."""
    reachable, frontier = set(), list(entries)
    while frontier:
        name = frontier.pop()
        if name in reachable or name not in recipes:
            continue
        reachable.add(name)
        frontier.extend(recipes[name][0])
    return reachable


def recipe_guards(recipes, names):
    """Guard -> recipe, for the enforcing lines of the named recipes."""
    found = {}
    for name in sorted(names):
        if name not in recipes:
            continue
        for line in recipes[name][1]:
            # `@` only silences the echo; `-` makes just ignore a failure, which
            # is the Justfile's `continue-on-error` and does not enforce.
            stripped = line.lstrip("@")
            if stripped.startswith("-"):
                continue
            for guard in invoked_scripts(stripped):
                found.setdefault(guard, name)
    return found


def justfile_coverage(text, entry_recipe, guard_recipe):
    """Guard -> recipe, for recipes `just <entry_recipe>` actually reaches."""
    recipes = parse_recipes(text)
    for required in (entry_recipe, guard_recipe):
        if required not in recipes:
            sys.exit(
                f"parse self-test failed: no `{required}:` recipe in the Justfile; "
                "it was renamed, so fix this script"
            )
    reachable = recipes_reachable_from(recipes, [entry_recipe])
    if guard_recipe not in reachable:
        sys.exit(
            f"error: `{guard_recipe}` is not reachable from `{entry_recipe}`, so "
            f"`just {entry_recipe}` does not run any guard in it"
        )
    return recipe_guards(recipes, reachable), reachable


def self_test():
    """Prove every classification rule still bites, before believing a clean run.

    The rules were verified once by hand-mutating this repository, which is not
    repeatable and does not survive into CI. Each case below is one of those
    mutations, reduced to a fixture: the `counted` cases are the controls that
    catch a rule grown too strict, the `ignored` cases are the defeats #1133's
    review named. A guard that cannot detect its own blind spots is the thing
    this replaced.
    """
    def workflow(body):
        return {"probe.yml": parse_yaml_subset(body)}

    live = """
on:
  pull_request:
jobs:
  one:
    runs-on: ubuntu-latest
    steps:
      - run: bash scripts/check-probe.sh
"""
    counted = {
        "a live step in a pull_request workflow": live,
        "a guard mid-pipeline": live.replace(
            "bash scripts/check-probe.sh", "cd . && bash scripts/check-probe.sh"
        ),
        "a guard reached through a called reusable workflow": None,
    }
    ignored = {
        "a step behind `if: false`": live.replace(
            "      - run:", "      - if: false\n        run:"
        ),
        "a step with continue-on-error": live.replace(
            "      - run:", "      - continue-on-error: true\n        run:"
        ),
        "a job with continue-on-error": live.replace(
            "    runs-on:", "    continue-on-error: true\n    runs-on:"
        ),
        "the invocation only echoed": live.replace(
            "run: bash scripts/check-probe.sh",
            'run: echo "bash scripts/check-probe.sh"',
        ),
        "a workflow with no pull_request trigger": live.replace(
            "  pull_request:", "  schedule:\n    - cron: '0 0 * * *'"
        ),
    }

    for label, body in counted.items():
        if body is None:
            continue
        if "check-probe.sh" not in workflow_coverage(workflow(body)):
            sys.exit(f"self-test failed: {label} was not counted")
    for label, body in ignored.items():
        if "check-probe.sh" in workflow_coverage(workflow(body)):
            sys.exit(f"self-test failed: {label} was counted as coverage")

    # A step that runs `just <recipe>` executes every guard that recipe reaches.
    # Reading only literal `bash scripts/...` lines would report all of them as
    # absent from CI, which is a false failure and the wrong verdict.
    indirect_recipes = parse_recipes(
        "check: check-guards\n\ncheck-guards: inner\n    true\n\ninner:\n"
        "    bash scripts/check-probe.sh\n"
    )
    indirect_workflow = workflow(live.replace("bash scripts/check-probe.sh", "just check-guards"))
    if "check-probe.sh" not in workflow_coverage(indirect_workflow, indirect_recipes):
        sys.exit("self-test failed: a guard run through `just <recipe>` was not counted")
    if "check-probe.sh" in workflow_coverage(indirect_workflow, {}):
        sys.exit("self-test failed: `just <recipe>` credited a guard with no Justfile")
    # A `-` prefixed line inside the recipe `just` runs still does not enforce.
    guarded = parse_recipes(
        "check: check-guards\n\ncheck-guards:\n    -bash scripts/check-probe.sh\n"
    )
    if "check-probe.sh" in workflow_coverage(indirect_workflow, guarded):
        sys.exit("self-test failed: `just` credited a `-` prefixed recipe line")

    # A guard only inside a reusable workflow: ignored when nothing calls it,
    # counted when a reachable job does.
    reusable = {
        "probe.yml": parse_yaml_subset(
            """
on:
  pull_request:
jobs:
  one:
    runs-on: ubuntu-latest
    steps:
      - run: 'true'
"""
        ),
        "called.yml": parse_yaml_subset(
            """
on:
  workflow_call:
jobs:
  two:
    runs-on: ubuntu-latest
    steps:
      - run: bash scripts/check-probe.sh
"""
        ),
    }
    if "check-probe.sh" in workflow_coverage(reusable):
        sys.exit("self-test failed: a reusable workflow nothing calls was counted")
    reusable["probe.yml"]["jobs"]["one"] = {"uses": "./.github/workflows/called.yml"}
    if "check-probe.sh" not in workflow_coverage(reusable):
        sys.exit("self-test failed: a called reusable workflow was not counted")

    # The Justfile half.
    def just(body):
        return justfile_coverage(body, "check", "check-guards")[0]

    base = "check: check-guards\n\ncheck-guards:\n    bash scripts/check-probe.sh\n"
    if "check-probe.sh" not in just(base):
        sys.exit("self-test failed: a live Justfile line was not counted")
    if "check-probe.sh" not in just(base.replace("    bash", "    @bash")):
        sys.exit("self-test failed: `@` suppression was treated as non-enforcing")
    # Both `-` spellings just accepts. The second is the one that needs the
    # explicit skip: `os.path.basename("-scripts/check-probe.sh")` is
    # `check-probe.sh`, so without it the ignore-failure prefix would be
    # credited as coverage.
    if "check-probe.sh" in just(base.replace("    bash", "    -bash")):
        sys.exit("self-test failed: a `-bash` prefixed line was counted")
    if "check-probe.sh" in just(
        base.replace("    bash scripts/check-probe.sh", "    -scripts/check-probe.sh")
    ):
        sys.exit("self-test failed: a `-` prefixed script path was counted")
    if "check-probe.sh" in just(
        base.replace("    bash scripts/check-probe.sh", '    echo "bash scripts/check-probe.sh"')
    ):
        sys.exit("self-test failed: an echoed invocation was counted")
    unreachable = (
        "check: check-guards\n\ncheck-guards:\n    true\n\nextra:\n"
        "    bash scripts/check-probe.sh\n"
    )
    if "check-probe.sh" in just(unreachable):
        sys.exit("self-test failed: a recipe `check` does not reach was counted")

    # The reader refuses rather than guessing.
    for label, body in {
        "a tab": "on:\n\tpull_request:\n",
        "an anchor": "on:\n  pull_request:\nx: &a 1\n",
        "a flow mapping": "on: {pull_request: null}\n",
        "a second document": "on:\n  pull_request:\n---\non:\n  push:\n",
    }.items():
        try:
            parse_yaml_subset(body)
        except YamlSubsetError:
            continue
        sys.exit(f"self-test failed: the reader accepted {label} instead of refusing")


self_test()

documents = load_workflows(workflows_dir)
if len(documents) < 5:
    sys.exit(
        f"parse self-test failed: read {len(documents)} workflows from "
        f"{workflows_dir}; the scan broke, so a clean result means nothing"
    )
oracle_verdict = oracle_check(workflows_dir, documents)
with open(justfile_path, encoding="utf-8") as handle:
    justfile_text = handle.read()
justfile_recipes = parse_recipes(justfile_text)
reachable_workflows = reachable_workflow_set(documents)
if not reachable_workflows:
    sys.exit(
        "parse self-test failed: no workflow triggers on pull_request. Either "
        "`on:` moved or the YAML 1.1 boolean-key case is no longer handled; fix "
        "this script rather than trusting an empty reachable set."
    )
in_ci = workflow_coverage(documents, justfile_recipes)

in_just, reachable_recipes = justfile_coverage(justfile_text, entry_recipe, guard_recipe)

on_disk = sorted(
    name
    for name in os.listdir("scripts")
    if name.startswith("check-") and name.endswith(".sh")
)
if len(on_disk) < 5:
    sys.exit(f"parse self-test failed: found {len(on_disk)} check-*.sh under scripts/")
if not in_ci or not in_just:
    sys.exit(
        f"parse self-test failed: extracted {len(in_ci)} guards from workflows and "
        f"{len(in_just)} from {justfile_path}; a command-position match that finds "
        "nothing means the extraction broke, not that nothing is wired"
    )

failures = []

missing_locally = sorted(set(in_ci) - set(in_just))
if missing_locally:
    failures.append(
        "these guards are executed by a pull-request workflow but not by "
        f"`just {entry_recipe}`:\n    "
        + "\n    ".join(f"{name}  (CI: {in_ci[name]})" for name in missing_locally)
        + f"\n  A contributor running the documented local gate before pushing does\n"
        f"  not run them. Add each to the `{guard_recipe}` recipe."
    )

missing_in_ci = sorted(set(in_just) - set(in_ci))
if missing_in_ci:
    failures.append(
        f"these guards are executed by `just {entry_recipe}` but by no "
        "pull-request workflow:\n    "
        + "\n    ".join(f"{name}  (just: {in_just[name]})" for name in missing_in_ci)
        + "\n  They read as enforced and are not: no pull request runs them. A step\n"
        "  whose `if:` is false, whose `continue-on-error` is set, or that sits in\n"
        "  a reusable workflow nothing calls, does not count."
    )

phantom = sorted((set(in_ci) | set(in_just)) - set(on_disk))
if phantom:
    failures.append(
        "these guards are invoked but do not exist under scripts/:\n    "
        + "\n    ".join(phantom)
    )

unwired = sorted(set(on_disk) - (set(in_ci) | set(in_just)))
if unwired:
    failures.append(
        "these guard scripts exist but nothing executes them:\n    "
        + "\n    ".join(unwired)
        + f"\n  Wire each into a workflow and the `{guard_recipe}` recipe, or delete it."
    )

if failures:
    print("error: guard coverage is incomplete", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}\n", file=sys.stderr)
    sys.exit(1)

print(f"OK: subset YAML reader {oracle_verdict}.")
print(
    f"OK: all {len(on_disk)} guard scripts are executed by a pull-request "
    f"workflow ({len(reachable_workflows)} reachable of {len(documents)}) and by "
    f"`just {entry_recipe}` ({len(reachable_recipes)} reachable recipes)."
)
PYEOF
