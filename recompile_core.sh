#!/bin/bash
# recompile_core.sh - Compile ShakerMaker core module
# Usage: ./recompile_core.sh /path/to/virtualenv

set -e

# Check virtual environment argument
if [ -z "$1" ]; then
    echo "Usage: ./recompile_core.sh /path/to/virtualenv"
    echo "Example: ./recompile_core.sh /home/pxpalacios/diana_prince"
    exit 1
fi

VENV_PATH="$1"

# Validate virtual environment
if [ ! -d "$VENV_PATH/lib" ]; then
    echo "Error: Invalid virtual environment path: $VENV_PATH"
    exit 1
fi

# Detect Python version
PY_VERSION=$(ls "$VENV_PATH/lib" | grep python | head -1)
if [ -z "$PY_VERSION" ]; then
    echo "Error: Could not detect Python version in $VENV_PATH"
    exit 1
fi

echo "Detected Python: $PY_VERSION"

# Get script directory (where the repo is)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$SCRIPT_DIR/shakermaker/core"

cd "$CORE_DIR"
echo "Compiling in: $CORE_DIR"

# Compile with f2py
CC=gcc python -m numpy.f2py core.pyf \
    subgreen.f subgreen2.f subfk.f subfocal.f subtrav.f \
    tau_p.f kernel.f prop.f source.f bessel.f haskell.f \
    fft.c Complex.c \
    -c --fcompiler=gnu95 \
    --f77flags="-ffixed-line-length-132 -fPIC -O2 -Wno-all -fopenmp" \
    --f90flags="-fPIC -O2 -Wno-all -fopenmp" \
    -lgomp \
    -m core

# Find compiled .so file
SO_FILE=$(ls core.cpython-*.so 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    echo "Error: Compilation failed - no .so file found"
    exit 1
fi

echo "Compiled: $SO_FILE"

# Copy to shakermaker package
cp "$SO_FILE" ../
echo "Copied to: $SCRIPT_DIR/shakermaker/"

# Copy to virtual environment
DEST_DIR="$VENV_PATH/lib/$PY_VERSION/site-packages/shakermaker"
if [ -d "$DEST_DIR" ]; then
    cp "$SO_FILE" "$DEST_DIR/"
    echo "Copied to: $DEST_DIR/"
else
    echo "Warning: $DEST_DIR does not exist. Run 'pip install .' first."
fi

# Verify
echo ""
echo "Verifying..."
python -c "from shakermaker import core; print('Location:', core.__file__); print('subgreen2:', hasattr(core, 'subgreen2'))"

echo ""
echo "Done!"