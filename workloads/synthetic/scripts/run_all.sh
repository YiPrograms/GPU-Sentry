#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
usage: run_all.sh [runtime_seconds] [binary args...]

Run every executable in the workloads/synthetic/binaries directory.

Arguments:
  runtime_seconds   Runtime passed to each binary. Default: 10
  binary args       Extra arguments passed to every binary.

Environment:
  RUNTIME_SECONDS        Default runtime if runtime_seconds is omitted.
  BINARIES_DIR          Directory containing binaries.
  TIMEOUT_GRACE_SECONDS Extra seconds before an external timeout kills a run.
                         Default: 30

Examples:
  run_all.sh
  run_all.sh 30
  run_all.sh 5 --blocks 1024 --threads 128
EOF
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"

binaries_dir="${BINARIES_DIR:-$project_dir/binaries}"
runtime_seconds="${RUNTIME_SECONDS:-10}"
timeout_grace_seconds="${TIMEOUT_GRACE_SECONDS:-30}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  runtime_seconds="$1"
  shift
fi

if ! is_positive_integer "$runtime_seconds"; then
  echo "error: runtime_seconds must be a positive integer, got: $runtime_seconds" >&2
  exit 2
fi

if ! is_nonnegative_integer "$timeout_grace_seconds"; then
  echo "error: TIMEOUT_GRACE_SECONDS must be a nonnegative integer, got: $timeout_grace_seconds" >&2
  exit 2
fi

if [[ ! -d "$binaries_dir" ]]; then
  echo "error: binaries directory not found: $binaries_dir" >&2
  exit 2
fi

mapfile -d '' binaries < <(
  find "$binaries_dir" \
    -maxdepth 1 \
    -type f \
    -perm -u+x \
    ! -name 'manifest.jsonl' \
    -print0 | sort -z
)

if [[ "${#binaries[@]}" -eq 0 ]]; then
  echo "error: no executable binaries found in $binaries_dir" >&2
  exit 2
fi

timeout_seconds=$((runtime_seconds + timeout_grace_seconds))
use_timeout=0
if command -v timeout >/dev/null 2>&1; then
  use_timeout=1
fi

echo "Running ${#binaries[@]} binaries from $binaries_dir"
echo "Per-binary runtime: ${runtime_seconds}s"
if [[ "$use_timeout" -eq 1 ]]; then
  echo "External timeout: ${timeout_seconds}s"
fi

passed=0
failed=0

for binary in "${binaries[@]}"; do
  name="$(basename "$binary")"
  echo
  echo "==> $name"

  if [[ "$use_timeout" -eq 1 ]]; then
    timeout --foreground "${timeout_seconds}s" "$binary" "$runtime_seconds" "$@"
  else
    "$binary" "$runtime_seconds" "$@"
  fi
  status=$?

  if [[ "$status" -eq 0 ]]; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
    if [[ "$status" -eq 124 ]]; then
      echo "!! $name exceeded external timeout (${timeout_seconds}s)" >&2
    else
      echo "!! $name failed with exit code $status" >&2
    fi
  fi
done

echo
echo "Done. passed=$passed failed=$failed"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
