#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Build an offscreen Vulkan renderer for Android/Termux (arm64-v8a).
# Required in Termux: cmake, ninja, glslang, and an installed Android NDK.
ROOT="$(cd "$(dirname "$0")" && pwd)"
: "${ANDROID_NDK_ROOT:=${ANDROID_NDK_HOME:-}}"
if [[ -z "$ANDROID_NDK_ROOT" || ! -f "$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" ]]; then
  echo "Set ANDROID_NDK_ROOT to your Android NDK directory." >&2
  exit 2
fi
if ! command -v cmake >/dev/null 2>&1 || ! command -v glslangValidator >/dev/null 2>&1; then
  echo "Install cmake, ninja, and glslang in Termux (pkg install cmake ninja glslang)." >&2
  exit 2
fi

BUILD_DIR="$ROOT/build-android-arm64"
cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGLSLANG_VALIDATOR="$(command -v glslangValidator)"
cmake --build "$BUILD_DIR" --target engine_render -j"$(nproc)"

OUT="$HOME/storage/downloads/engine-render-arm64"
mkdir -p "$OUT/shaders" "$OUT/assets"
cp "$BUILD_DIR/engine_render" "$OUT/engine_render"
cp "$BUILD_DIR"/shaders/*.spv "$OUT/shaders/"
cp "$ROOT"/assets/*.rgba "$OUT/assets/"
cat > "$OUT/run.sh" <<'RUN'
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Conservative REDMAGIC Astra starting profile. Raise resolution gradually after testing.
export ENGINE_WIDTH="${ENGINE_WIDTH:-960}"
export ENGINE_HEIGHT="${ENGINE_HEIGHT:-540}"
export ENGINE_SHADOW_SIZE="${ENGINE_SHADOW_SIZE:-1024}"
export ENGINE_QUICK="${ENGINE_QUICK:-1}"
# Shaders and textures are read from the folder this script lives in.
export ENGINE_SHADER_DIR="$(pwd)/shaders"
export ENGINE_ASSET_DIR="$(pwd)/assets"
exec ./engine_render
RUN
chmod +x "$OUT/run.sh" "$OUT/engine_render"
echo "Android build copied to: $OUT"
echo "Run: $OUT/run.sh"
