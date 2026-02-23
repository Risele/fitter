#!/usr/bin/env bash
set -euo pipefail

try_python() {
  local cmd="$1"
  if eval "$cmd -c 'import sys'" >/dev/null 2>&1; then
    PYTHON_BIN="$cmd"
    return 0
  fi
  return 1
}

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  try_python "python3" || true
fi
if [ -z "$PYTHON_BIN" ] && command -v python >/dev/null 2>&1; then
  try_python "python" || true
fi
if [ -z "$PYTHON_BIN" ] && command -v py >/dev/null 2>&1; then
  try_python "py -3" || true
fi
if [ -z "$PYTHON_BIN" ] && command -v py >/dev/null 2>&1; then
  try_python "py" || true
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "FAIL: python not found (python3/python/py)"
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

if command -v rg >/dev/null 2>&1; then
  GREP_BIN=rg
else
  GREP_BIN=grep
fi

echo "Using Grep: $GREP_BIN"

GREP_ARGS=""


cd "$(dirname "$0")/.."

OUT_BASE="${SMOKE_OUT_DIR:-tests/smoke_out}"
mkdir -p "$OUT_BASE"
RUN_DIR="$OUT_BASE/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "Outputs will be kept in: $RUN_DIR"

TMPDIR="$RUN_DIR/tmp"
mkdir -p "$TMPDIR"

check_stdout() {
  local out="$1"
  if [ ! -s "$out" ]; then echo "FAIL: stdout empty"; exit 1; fi
  if ! head -n 1 "$out" | $GREP_BIN -F -q "/*"; then echo "FAIL: stdout not starting with /*"; exit 1; fi
  if ! $GREP_BIN -F -q "Memory usage (bytes):" "$out"; then echo "FAIL: summary missing"; exit 1; fi
  echo "ok output format"
}

run_and_check() {
  local name="$1"; shift
  local cmd="$1"; shift
  echo "=== $name ==="
  local out="$RUN_DIR/${name}.stdout.txt"
  local err="$RUN_DIR/${name}.stderr.txt"
  if ! eval "$cmd" >"$out" 2>"$err"; then
    echo "FAIL: $name (non-zero exit)"; cat "$err"; exit 1; fi
  echo "ok exit"
  check_stdout "$out"
  echo "stdout saved: $out"
}


run_and_check "csv-linear-float" "${PYTHON_BIN} -m app.cli tests/test.csv --linear --float --test-count 0"
run_and_check "csv-spline-double" "${PYTHON_BIN} -m app.cli tests/test.csv --spline --double --with-slopes --test-count 0"
run_and_check "analytic-builtin" "${PYTHON_BIN} -m app.cli --func sin --xmin -2 --xmax 5 --test-count 0"
run_and_check "analytic-inline" "${PYTHON_BIN} -m app.cli --func 'sin(x)+0.1*x*x' --xmin -2 --xmax 5 --test-count 0"

expr_file="$RUN_DIR/expr.txt"
printf 'cos(x)+pow(x,2)' > "$expr_file"
run_and_check "analytic-file" "${PYTHON_BIN} -m app.cli --func-file '$expr_file' --xmin -1 --xmax 1 --test-count 0"

out_noext="$RUN_DIR/fit"
${PYTHON_BIN} -m app.cli tests/test.csv --out "$out_noext" --test-count 0 >"$RUN_DIR/export-noext.stdout.txt" 2>"$RUN_DIR/export-noext.stderr.txt"
if [ -s "$RUN_DIR/export-noext.stdout.txt" ]; then echo "FAIL: export-noext produced stdout"; exit 1; fi
if [ ! -f "$out_noext.txt" ]; then echo "FAIL: missing $out_noext.txt"; exit 1; fi
if ! head -n 1 "$out_noext.txt" | $GREP_BIN -F -q "/*"; then echo "FAIL: file not starting with /*"; exit 1; fi
if ! $GREP_BIN -F -q "Memory usage (bytes):" "$out_noext.txt"; then echo "FAIL: summary missing in file"; exit 1; fi

echo "ok export noext"

out_ext="$RUN_DIR/fit.c"
${PYTHON_BIN} -m app.cli tests/test.csv --out "$out_ext" --test-count 0 >"$RUN_DIR/export-ext.stdout.txt" 2>"$RUN_DIR/export-ext.stderr.txt"
if [ -s "$RUN_DIR/export-ext.stdout.txt" ]; then echo "FAIL: export-ext produced stdout"; exit 1; fi
if [ ! -f "$out_ext" ]; then echo "FAIL: missing $out_ext"; exit 1; fi
if ! head -n 1 "$out_ext" | $GREP_BIN -F -q "/*"; then echo "FAIL: file not starting with /*"; exit 1; fi
if ! $GREP_BIN -F -q "Memory usage (bytes):" "$out_ext"; then echo "FAIL: summary missing in file"; exit 1; fi

echo "ok export ext"

echo "ALL TESTS PASSED"
read