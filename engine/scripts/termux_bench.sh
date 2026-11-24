#!/usr/bin/env bash
# Termux device benchmark for engine_render. Builds natively on the phone, then runs the
# timed benchmark RUNS times and writes one report file to share back.
#
# One-time setup in Termux:
#   pkg install cmake ninja clang glslang vulkan-headers vulkan-loader-android vulkan-tools
# If a package name is not found, check it with: pkg search <name>
#
# Usage, from anywhere (the engine source is found from this script's location):
#   bash scripts/termux_bench.sh              # 1920x1080, 3 runs
#   bash scripts/termux_bench.sh 1280 720     # another resolution
#   RUNS=5 bash scripts/termux_bench.sh       # more runs
#
# Validation is off (ENGINE_VALIDATION is not set). ENGINE_BENCH=1 stops after the timing
# report, so the CPU-reference suite does not run on the phone.
set -eu -o pipefail

W=${1:-1920}
H=${2:-1080}
RUNS=${RUNS:-3}
SRC=$(cd "$(dirname "$0")/.." && pwd)
BUILD=$SRC/build-termux
REPORT=$SRC/bench-$(date +%Y%m%d-%H%M%S)-${W}x${H}.txt

for tool in cmake ninja clang++ glslangValidator vulkaninfo; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing: $tool. Install the packages listed at the top of this script." >&2
    exit 1
  fi
done

{
  echo "== device and driver =="
  date
  uname -a
  # sed reads all of vulkaninfo's output; piping into head could kill vulkaninfo early.
  vulkaninfo --summary 2>&1 | sed -n '1,60p' || true
  echo
  echo "== build =="
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD" --target engine_render
  for run in $(seq 1 "$RUNS"); do
    echo
    echo "== run $run of $RUNS, ${W}x${H} =="
    ENGINE_BENCH=1 ENGINE_PROFILE=1 ENGINE_WIDTH="$W" ENGINE_HEIGHT="$H" "$BUILD/engine_render"
    if [ "$run" -lt "$RUNS" ]; then sleep 30; fi  # let the phone cool between runs
  done
} 2>&1 | tee "$REPORT"

echo "report saved to: $REPORT"
