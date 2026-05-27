#!/usr/bin/env bash
# Build LHAPDF + the CT18NNLO PDF set into vendor/lhapdf, internal to the
# project (no conda, no system install). Idempotent; rebuilds from scratch.
#
#   make lhapdf      # or: bash scripts/install_lhapdf.sh
#
# The result is gitignored (vendor/); modules/analytic_smeft/pdfs.py
# self-discovers it. LHAPDF has no Python 3.12 wheels, hence the source build.
set -euo pipefail

LHAPDF_VERSION="6.5.6"
PDF_SET="CT18NNLO"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$REPO_ROOT/vendor/lhapdf"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
PATCHELF="$REPO_ROOT/.venv/bin/patchelf"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

[ -x "$VENV_PYTHON" ] || { echo "ERROR: venv python missing; run 'make setup' first." >&2; exit 1; }
command -v "$PATCHELF" >/dev/null 2>&1 || PATCHELF="patchelf"
command -v "$PATCHELF" >/dev/null 2>&1 || { echo "ERROR: patchelf missing; 'uv sync' installs it." >&2; exit 1; }

# Build with the system toolchain; strip any conda compiler/flags that would
# otherwise leak in from an activated conda base environment.
unset CXXFLAGS CFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH \
      LIBRARY_PATH CONDA_PREFIX 2>/dev/null || true
export PATH="/usr/bin:/bin:$PATH" CC="/usr/bin/gcc" CXX="/usr/bin/g++"

echo "==> downloading LHAPDF $LHAPDF_VERSION"
cd "$BUILD_DIR"
curl -fsSL -o lhapdf.tar.gz "https://lhapdf.hepforge.org/downloads/?f=LHAPDF-$LHAPDF_VERSION.tar.gz"
tar -xzf lhapdf.tar.gz
cd "LHAPDF-$LHAPDF_VERSION"

echo "==> building LHAPDF into $PREFIX"
rm -rf "$PREFIX"
./configure --prefix="$PREFIX" PYTHON="$VENV_PYTHON" --disable-static \
    CC="/usr/bin/gcc" CXX="/usr/bin/g++" >/dev/null
make -j"$(nproc)" >/dev/null
make install >/dev/null

echo "==> setting RPATH on the Python extension"
SO="$(find "$PREFIX/lib" -path '*site-packages*' -name 'lhapdf*.so' | head -1)"
[ -n "$SO" ] || { echo "ERROR: built lhapdf extension not found." >&2; exit 1; }
"$PATCHELF" --set-rpath '$ORIGIN/../../..' "$SO"

echo "==> installing the $PDF_SET PDF set"
mkdir -p "$PREFIX/share/LHAPDF"
curl -fsSL "https://lhapdfsets.web.cern.ch/lhapdfsets/current/$PDF_SET.tar.gz" \
    | tar -xz -C "$PREFIX/share/LHAPDF"

echo "==> verifying"
"$VENV_PYTHON" - <<'PY'
import glob, sys
site = glob.glob("vendor/lhapdf/lib/python*/site-packages")[0]
sys.path.insert(0, site)
import lhapdf
lhapdf.setVerbosity(0)
pdf = lhapdf.mkPDF("CT18NNLO", 0)
print(f"   lhapdf {lhapdf.version()} OK; CT18NNLO xf(u, x=0.1, Q=100) = {pdf.xfxQ(2, 0.1, 100.0):.6f}")
PY
echo "==> done. modules/analytic_smeft uses this automatically (pdf='auto')."
