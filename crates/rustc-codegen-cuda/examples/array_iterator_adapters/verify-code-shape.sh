#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
ptx="${1:-$root/array_iterator_adapters.ptx}"

if [[ ! -f "$ptx" ]]; then
  echo "PTX file not found: $ptx" >&2
  exit 1
fi

extract_entry() {
  local symbol="$1"
  awk -v symbol="$symbol" '
    $0 ~ "\\.visible[[:space:]]+\\.entry[[:space:]]+" symbol "\\(" {
      inside = 1
    }
    inside {
      print
      opens += gsub(/\{/, "{")
      closes += gsub(/\}/, "}")
      if (opens > 0 && opens == closes) {
        exit
      }
    }
  ' "$ptx"
}

for symbol in \
  iter_copied_take \
  iter_copied_skip_take_enumerate
do
  body="$(extract_entry "$symbol")"
  if [[ -z "$body" ]]; then
    echo "entry not found: $symbol" >&2
    exit 1
  fi

  if grep -Eq '(^|[[:space:]])\.local|ld\.local|st\.local' <<<"$body"; then
    echo "unexpected local-memory operation in $symbol" >&2
    printf '%s\n' "$body" >&2
    exit 1
  fi
done

echo "array_iterator_adapters PTX shape: PASS"
