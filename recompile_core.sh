# recompile_core.sh
cd /mnt/deadmanschest/pxpalacios/REPO/ShakerMaker_OP/ShakerMaker/shakermaker/core

python -m numpy.f2py core.pyf \
    subgreen.f subgreen2.f subfk.f subfocal.f subtrav.f \
    tau_p.f kernel.f prop.f source.f bessel.f haskell.f \
    fft.c Complex.c radiats.c \
    -c --fcompiler=gnu95 \
    --f77flags="-ffixed-line-length-132 -fPIC -O1 -Wno-all" \
    --f90flags="-fPIC -O1 -Wno-all" \
    -m core

cp core.cpython-310-x86_64-linux-gnu.so ../
cp core.cpython-310-x86_64-linux-gnu.so /mnt/deadmanschest/pxpalacios/v_ENV/diana_prince/lib/python3.10/site-packages/shakermaker/

echo "Done. Verifying..."
python -c "from shakermaker import core; print(core.__file__); print(hasattr(core, 'subgreen2'))"