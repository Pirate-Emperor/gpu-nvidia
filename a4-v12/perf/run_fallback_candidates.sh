#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")" && pwd)
CASES=(
  "wk32|wildfire_wk32.cu"
  "fg16|wildfire_fg16.cu"
  "fg8x8|wildfire_fg8x8.cu"
)
mkdir -p "$ROOT/perf/out/fallback"
for spec in "${CASES[@]}"; do
  IFS='|' read -r tag file <<< "$spec"
  candidate="$ROOT/perf/candidates/$file"
  passed=1
  for pass in 1 2; do
    out="$ROOT/perf/out/fallback/$tag/pass$pass"
    mkdir -p "$out"
    echo "=== CANDIDATE=$tag PASS=$pass ARCH=sm_70 REP=9 WARM=1 CHK=3 ==="
    REP=9 WARM=1 CHK=3 ARCH=sm_70 CAND_SRC="$candidate" OUT="$out" \
      bash "$ROOT/perf/compare_candidate.sh" >"$out.step.log" 2>&1
    rc=$?
    if (( rc != 0 )) || ! grep -Fq 'Promotion gate: **PASS**' "$out/report.md" 2>/dev/null; then
      echo "CANDIDATE_REJECTED tag=$tag pass=$pass rc=$rc"
      passed=0
      break
    fi
    echo "CANDIDATE_GATE_PASS tag=$tag pass=$pass"
  done
  if (( passed )); then
    printf '%s\n' "$tag" > "$ROOT/perf/out/fallback/winner.txt"
    cp "$candidate" "$ROOT/perf/out/fallback/wildfire_v12_candidate.cu"
    sha256sum "$candidate" > "$ROOT/perf/out/fallback/winner.sha256"
    echo "V12_CANDIDATE_ELIGIBLE=$tag"
    exit 0
  fi
done
echo "NO_FALLBACK_CANDIDATE_PASSED_BOTH_INDEPENDENT_GATES"
exit 1
