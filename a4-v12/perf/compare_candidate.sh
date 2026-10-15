#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE_SRC="$ROOT/wildfire.cu"
CAND_SRC="${CAND_SRC:-$ROOT/perf/candidates/wildfire_grid2.cu}"
OUT=$(printenv OUT 2>/dev/null || true)
[[ -n "$OUT" ]] || OUT="$ROOT/perf/out/candidate-compare"
REP=$(printenv REP); WARM=$(printenv WARM); CHK=$(printenv CHK)
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-60}
ARCH=${ARCH:-sm_70}
if [[ "$ARCH" != "sm_70" && "$ARCH" != "sm_80" ]]; then echo "ARCH must be sm_70 or sm_80" >&2; exit 2; fi
[[ -n "$REP" ]] || REP=15
[[ -n "$WARM" ]] || WARM=2
[[ -n "$CHK" ]] || CHK=5
CASES="01_sample 02_calm_spread 03_wind 04_slope 05_heat_rounds_down 06_fuel_gaps 07_stations_dont_burn 08_crew_tie_break 09_crew_refill 10_wet_beats_burnt 11_crews_together 12_random_crews 13_random_full"

mkdir -p "$OUT/base" "$OUT/candidate"
rm -f "$OUT"/samples.csv "$OUT"/summary.csv "$OUT"/report.md "$OUT"/compare.log "$OUT"/base/*.log "$OUT"/candidate/*.log
exec > >(tee "$OUT/compare.log") 2>&1
if (( REP < 9 )); then echo "REP must be >= 9 for a useful paired comparison" >&2; exit 2; fi
if (( CHK < 3 )); then echo "CHK must be >= 3 repeat rounds" >&2; exit 2; fi
if (( WARM < 1 )); then echo "WARM must be >= 1 untimed warm-up per case and source" >&2; exit 2; fi
if (( TIMEOUT_SECONDS < 1 )); then echo "TIMEOUT_SECONDS must be positive" >&2; exit 2; fi
[[ -f "$BASE_SRC" && -f "$CAND_SRC" ]] || { echo "Missing baseline or candidate source"; exit 2; }
for nm in $CASES; do
  [[ -f "$ROOT/input/$nm.in.bin" && -f "$ROOT/output/$nm.out.bin" ]] || { echo "Missing input/golden for $nm"; exit 2; }
done

echo "GPU: $(nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu --format=csv,noheader | head -1)"
echo "NVCC: $(nvcc --version | tail -1)"
echo "ARCH=$ARCH (sm_70 is required; sm_80 informational only)"
echo "REPS=$REP WARMUPS=$WARM REPEAT_ROUNDS=$CHK CASES=13 SOLVER_TIMEOUT=${TIMEOUT_SECONDS}s"
echo "BASE_SRC=$BASE_SRC"
echo "CAND_SRC=$CAND_SRC"

nvcc -O3 -arch="$ARCH" -I "$ROOT" "$ROOT/harness.cu" "$BASE_SRC" -o "$OUT/base/main" > "$OUT/base/build.log" 2>&1 || { cat "$OUT/base/build.log"; exit 1; }
nvcc -O3 -arch="$ARCH" -I "$ROOT" "$ROOT/harness.cu" "$CAND_SRC" -o "$OUT/candidate/main" > "$OUT/candidate/build.log" 2>&1 || { cat "$OUT/candidate/build.log"; exit 1; }
echo "BUILD=PASS baseline candidate"

run_solver() {
  local id="$1" nm="$2" out="$3" log="$4" rc tm
  rm -f "$out"
  set +e
  timeout --signal=TERM --kill-after=5s "${TIMEOUT_SECONDS}s" "$OUT/$id/main" "$ROOT/input/$nm.in.bin" "$out" >/dev/null 2>"$log"
  rc=$?
  set -e
  if (( rc != 0 )) || [[ ! -f "$out" ]]; then echo "SOLVER_FAIL id=$id case=$nm rc=$rc" >&2; return 1; fi
  tm=$(awk '/Simulation time:/{print $3; exit}' "$log")
  [[ "$tm" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "TIME_PARSE_FAIL id=$id case=$nm: $tm" >&2; return 1; }
  printf '%s' "$tm"
}

validate_golden() {
  local id="$1" nm out log
  echo "VALIDATE_START id=$id"
  for nm in $CASES; do
    out="$OUT/$id/$nm.gold.out"
    log="$OUT/$id/$nm.gold.log"
    run_solver "$id" "$nm" "$out" "$log" >/dev/null || return 1
    cmp -s "$out" "$ROOT/output/$nm.out.bin" || { echo "GOLDEN_FAIL id=$id case=$nm"; return 1; }
  done
  echo "GOLDEN=PASS id=$id cases=13"
}

repeat_candidate() {
  local nm rep out log
  echo "REPEAT_START rounds=$CHK"
  for ((rep=1;rep<=CHK;rep++)); do
    for nm in $CASES; do
      out="$OUT/candidate/$nm.repeat$rep.out"
      log="$OUT/candidate/$nm.repeat$rep.log"
      run_solver candidate "$nm" "$out" "$log" >/dev/null || return 1
      cmp -s "$out" "$ROOT/output/$nm.out.bin" || { echo "REPEAT_FAIL case=$nm round=$rep golden_mismatch"; return 1; }
    done
  done
  echo "REPEAT=PASS rounds=$CHK cases=13"
}

validate_golden base
validate_golden candidate
repeat_candidate
printf 'case,rep,order,baseline_ms,candidate_ms,paired_speedup,baseline_match,candidate_match\n' > "$OUT/samples.csv"


for nm in $CASES; do
  echo "CASE_START=$nm UNTIMED_WARMUPS=$WARM (excluded from timing)"
  for ((rep=1;rep<=WARM;rep++)); do
    for id in base candidate; do
      out="$OUT/$id/$nm.warm$rep.out"
      log="$OUT/$id/$nm.warm$rep.log"
      run_solver "$id" "$nm" "$out" "$log" >/dev/null
      cmp -s "$out" "$ROOT/output/$nm.out.bin" || { echo "WARMUP_CORRECTNESS_FAIL id=$id case=$nm"; exit 1; }
    done
  done
  for ((rep=1;rep<=REP;rep++)); do
    if (( rep % 2 )); then order=base-first; first=base; second=candidate; else order=candidate-first; first=candidate; second=base; fi
    bt=; ct=
    for id in "$first" "$second"; do
      out="$OUT/$id/$nm.bench$rep.out"
      log="$OUT/$id/$nm.bench$rep.log"
      tm=$(run_solver "$id" "$nm" "$out" "$log")
      if ! cmp -s "$out" "$ROOT/output/$nm.out.bin"; then
        echo "TIMED_CORRECTNESS_FAIL id=$id case=$nm rep=$rep"
        exit 1
      fi
      if [[ "$id" == base ]]; then bt=$tm; else ct=$tm; fi
    done
    speedup=$(awk -v b="$bt" -v c="$ct" 'BEGIN { if (c > 0) printf "%.8f", b/c; else exit 1 }')
    printf '%s,%s,%s,%s,%s,%s,PASS,PASS\n' "$nm" "$rep" "$order" "$bt" "$ct" "$speedup" >> "$OUT/samples.csv"
  done
  echo "TIMED_CASE_PASS case=$nm reps=$REP"
done

WARM="$WARM" ARCH="$ARCH" python3 "$ROOT/perf/summarize_candidate.py" "$OUT/samples.csv" "$OUT/summary.csv" "$OUT/report.md"
echo "CANDIDATE_COMPARE=PASS"
