#!/bin/bash
# recompile_core.sh - Compile ShakerMaker core module for NumPy 2.x compatibility
#
# Usage: 
#   ./recompile_core.sh                    # Use current Python
#   ./recompile_core.sh /path/to/venv      # Use specific virtualenv
#
# This script recompiles the Fortran core module using f2py, which is necessary
# when switching between NumPy versions (especially 1.x -> 2.x) since the C API
# changed and compiled modules are not compatible.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$SCRIPT_DIR/shakermaker/core"

# Determine Python to use
if [ -n "$1" ]; then
    VENV_PATH="$1"
    if [ -f "$VENV_PATH/bin/python" ]; then
        PYTHON="$VENV_PATH/bin/python"
    elif [ -f "$VENV_PATH/Scripts/python.exe" ]; then
        PYTHON="$VENV_PATH/Scripts/python.exe"
    else
        echo "Error: Cannot find Python in $VENV_PATH"
        exit 1
    fi
else
    PYTHON="python"
fi

echo "=============================================="
echo "ShakerMaker Core Module Recompilation"
echo "=============================================="
echo ""

# Check Python and NumPy versions
echo "Python: $($PYTHON --version)"
NUMPY_VERSION=$($PYTHON -c "import numpy; print(numpy.__version__)")
echo "NumPy:  $NUMPY_VERSION"
echo ""

# Check for required tools
echo "Checking requirements..."
$PYTHON -c "import numpy" || { echo "Error: NumPy not installed"; exit 1; }
which gfortran > /dev/null 2>&1 || { echo "Error: gfortran not found"; exit 1; }
which gcc > /dev/null 2>&1 || { echo "Error: gcc not found"; exit 1; }
echo "  [OK] All requirements met"
echo ""

# Navigate to core directory
cd "$CORE_DIR"
echo "Working directory: $CORE_DIR"
echo ""

# Clean old compiled files
echo "Cleaning old compiled files..."
rm -f core.cpython-*.so core.cp*.pyd coremodule.c *.o *.mod 2>/dev/null || true
echo "  [OK] Cleaned"
echo ""

# Source files
FORTRAN_SOURCES="subgreen.f subgreen2.f subfk.f subfocal.f subtrav.f tau_p.f kernel.f prop.f source.f bessel.f haskell.f"
C_SOURCES="fft.c Complex.c"

# Detect platform and set flags
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    echo "Platform: Windows"
    F77_FLAGS="/Qopenmp /extend-source:132"
    F90_FLAGS="/Qopenmp"
    FCOMPILER="--fcompiler=intelvem"
    LINK_FLAGS=""
else
    echo "Platform: Linux/macOS"
    F77_FLAGS="-ffixed-line-length-132 -fPIC -O2 -Wno-all -fopenmp"
    F90_FLAGS="-fPIC -O2 -Wno-all -fopenmp"
    FCOMPILER="--fcompiler=gnu95"
    LINK_FLAGS="-lgomp"
fi
echo ""

# Compile with f2py
echo "Compiling with f2py..."
echo "Command:"
echo "  $PYTHON -m numpy.f2py core.pyf \\"
echo "    $FORTRAN_SOURCES \\"
echo "    $C_SOURCES \\"
echo "    -c $FCOMPILER \\"
echo "    --f77flags=\"$F77_FLAGS\" \\"
echo "    --f90flags=\"$F90_FLAGS\" \\"
echo "    $LINK_FLAGS \\"
echo "    -m core"
echo ""

CC=gcc $PYTHON -m numpy.f2py core.pyf \
    $FORTRAN_SOURCES \
    $C_SOURCES \
    -c $FCOMPILER \
    --f77flags="$F77_FLAGS" \
    --f90flags="$F90_FLAGS" \
    $LINK_FLAGS \
    -m core

# Find compiled file
SO_FILE=$(ls core.cpython-*.so 2>/dev/null | head -1 || ls core.cp*.pyd 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    echo ""
    echo "Error: Compilation failed - no .so/.pyd file found"
    exit 1
fi

echo ""
echo "  [OK] Compiled: $SO_FILE"

# Copy to shakermaker package directory
cp "$SO_FILE" ../
echo "  [OK] Copied to: $SCRIPT_DIR/shakermaker/"

# Copy to virtualenv site-packages if specified
if [ -n "$1" ]; then
    PY_VERSION=$($PYTHON -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
    
    # Try common site-packages locations
    for SITE_PKG in \
        "$VENV_PATH/lib/$PY_VERSION/site-packages/shakermaker" \
        "$VENV_PATH/Lib/site-packages/shakermaker" \
        "$VENV_PATH/lib/site-packages/shakermaker"
    do
        if [ -d "$SITE_PKG" ]; then
            cp "$SO_FILE" "$SITE_PKG/"
            echo "  [OK] Copied to: $SITE_PKG/"
            break
        fi
    done
fi

echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="
echo ""

# Verify import works
$PYTHON -c "
from shakermaker import core
print(f'  Module location: {core.__file__}')
print(f'  Has subgreen: {hasattr(core, \"subgreen\")}')
print(f'  Has subgreen2: {hasattr(core, \"subgreen2\")}')
" || {
    echo ""
    echo "Warning: Verification failed. You may need to reinstall shakermaker:"
    echo "  pip install -e ."
}

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo ""
echo "If you still have import errors, try:"
echo "  pip uninstall shakermaker"
echo "  pip install -e ."
