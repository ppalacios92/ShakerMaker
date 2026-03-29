from setuptools import setup
from numpy.distutils.core import Extension, setup as np_setup
import importlib.util
import os
import sys

from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
import shutil

import subprocess

on_rtd = os.environ.get('READTHEDOCS') == 'True'

name = "shakermaker"
version = "1.4"
release = "0.01"
author = "Jose A. Abell, Jorge Crempien D., and Matias Recabarren"

profile = False

srcdir = "shakermaker/core/"
ffiles = ["subgreen.f", "subgreen2.f", "subfocal.f", "subfk.f", "subtrav.f", "tau_p.f", "kernel.f", "prop.f", "source.f", "bessel.f", "haskell.f", "fft.c", "Complex.c"]
srcs = [srcdir+'core.pyf']
for f in ffiles:
    srcs.append(srcdir+f)

if on_rtd:
    ext_modules = []
else:
    if profile:
        ext1 = Extension(
            name='shakermaker.core',
            sources=srcs,
            extra_f77_compile_args=["-ffixed-line-length-132", "-Wno-tabs", "-Wno-unused-dummy-argument", "-pg", "-fopenmp"],
            extra_link_args=["-pg",  "-fopenmp"]
        )
    else:
        # ext1 = Extension(
        #     name='shakermaker.core',
        #     sources=srcs,
        #     extra_f77_compile_args=["-ffixed-line-length-132", "-Wno-tabs", "-Wno-unused-dummy-argument", "-fPIC", "-fopenmp"],
        #     extra_compile_args=["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
        #     extra_link_args=["-fopenmp"]
        if sys.platform == 'win32':
            f77_args     = ["/Qopenmp", "/extend-source:132"]
            link_args    = ["/Qopenmp"]
            compile_args = ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]
        else:
            f77_args     = ["-ffixed-line-length-132", "-Wno-tabs", "-Wno-unused-dummy-argument", "-fPIC", "-fopenmp"]
            link_args    = ["-fopenmp"]
            compile_args = ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]

        ext1 = Extension(
            name='shakermaker.core',
            sources=srcs,
            extra_f77_compile_args=f77_args,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )

    ext_modules = [ext1]

# ========== FFSP compilation support
# def compile_ffsp():
#     """Compile FFSP wrapper module using f2py"""
#     ffsp_dir = os.path.join(os.path.dirname(__file__), 'shakermaker', 'ffsp')    
#     try:
#         subprocess.run(['make', '-f', 'Makefile_f2py', 'clean'], 
#                       cwd=ffsp_dir, check=False, capture_output=True)
        
#         result = subprocess.run(['make', '-f', 'Makefile_f2py'], 
#                                cwd=ffsp_dir, check=True, capture_output=True, text=True)
        
#         so_files = [f for f in os.listdir(ffsp_dir) if f.startswith('ffsp_core') and f.endswith('.so')]
        
#         if so_files:
#             print(f"[OK] FFSP wrapper compiled: {so_files[0]}")
#         else:
#             raise RuntimeError("FFSP wrapper not compiled")
            
#     except subprocess.CalledProcessError as e:
#         print("✗ FFSP compilation failed")
#         raise
def compile_ffsp():
    ffsp_dir = os.path.join(os.path.dirname(__file__), 'shakermaker', 'ffsp')
    if sys.platform == 'win32':
        _compile_ffsp_windows(ffsp_dir)
    else:
        _compile_ffsp_linux(ffsp_dir)

def _compile_ffsp_linux(ffsp_dir):
    fortran_sources = [
        "ffsp_wrapper.f90", "ffsp_comm.f90", "spfield_n.f90",
        "dcf_subs_1.f90", "slip_rate.f90", "ffsp_tool.f",
    ]
    # Remove existing artifact so f2py always does a full rebuild
    for f in os.listdir(ffsp_dir):
        if f.startswith('ffsp_core') and (f.endswith('.so') or f.endswith('.pyd')):
            os.remove(os.path.join(ffsp_dir, f))
    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-c", "ffsp.pyf",
        *fortran_sources,
        "--f90flags=-O3 -fPIC",
        "--f77flags=-O3 -std=legacy -fPIC",
        "-m", "ffsp_core",
    ]
    print("[ffsp] compiling on linux...")
    result = subprocess.run(cmd, cwd=ffsp_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError("[ffsp] compilation failed -- see output above")
    so_files = [f for f in os.listdir(ffsp_dir)
                if f.startswith('ffsp_core') and f.endswith('.so')]
    if so_files:
        print(f"[OK] FFSP wrapper compiled (Linux): {so_files[0]}")
    else:
        raise RuntimeError("FFSP .so not found after compilation")

def _compile_ffsp_windows(ffsp_dir):
    fortran_sources = [
        "ffsp_wrapper.f90", "ffsp_comm.f90", "spfield_n.f90",
        "dcf_subs_1.f90", "slip_rate.f90", "ffsp_tool.f",
    ]
    for f in os.listdir(ffsp_dir):
        if f.startswith('ffsp_core') and (f.endswith('.pyd') or f.endswith('.so')):
            os.remove(os.path.join(ffsp_dir, f))
    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-c", "ffsp.pyf",
        *fortran_sources,
        "--fcompiler=intelvem",
        "--f90flags=/Qopenmp",
        "--f77flags=/Qopenmp",
        "-m", "ffsp_core",
    ]
    # Windows change: capture_output=False so ifx compiler errors print
    # directly to the console instead of being silently swallowed.
    result = subprocess.run(cmd, cwd=ffsp_dir, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"[ffsp] f2py/ifx failed (exit {result.returncode}) "
            "-- read the ifx error lines above this message."
        )
    pyd_files = [f for f in os.listdir(ffsp_dir)
                 if f.startswith('ffsp_core') and f.endswith('.pyd')]
    if pyd_files:
        print(f"[OK] FFSP wrapper compiled (Windows): {pyd_files[0]}")
    else:
        raise RuntimeError("FFSP .pyd not found after compilation despite returncode=0")
        
    print(f"[ffsp] compiling from: {ffsp_dir}")
    import glob
    for f in glob.glob(os.path.join(ffsp_dir, "*.f90")) + glob.glob(os.path.join(ffsp_dir, "*.f")):
        print(f"[ffsp] source: {f}")

        
def _install_ffsp_binary(ffsp_dir):
    """Copy compiled .pyd/.so into site-packages after install creates the destination folder."""
    import site
    ext = ".pyd" if sys.platform == "win32" else ".so"
    for sp in site.getsitepackages():
        dest = os.path.join(sp, "shakermaker", "ffsp")
        if os.path.isdir(dest):
            for f in os.listdir(ffsp_dir):
                if f.startswith("ffsp_core") and f.endswith(ext):
                    shutil.copy2(os.path.join(ffsp_dir, f), os.path.join(dest, f))
                    print(f"[ffsp] installed {f} -> {dest}")
            return

class PostInstallCommand(_install):
    def run(self):
        _install.run(self)
        ffsp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakermaker", "ffsp")
        compile_ffsp()
        _install_ffsp_binary(ffsp_dir)

# Compile FFSP before the setup (source tree only, post-install handles site-packages)
# compile_ffsp()

# ============================================================

# Check for sphinx
found_sphinx = importlib.util.find_spec('sphinx') is not None

# ============================================================
cmdclass = {'install': PostInstallCommand}
command_options = {}

# if found_sphinx:
#     print("Configuring Sphinx autodocumentation")
#     from sphinx.setup_command import BuildDoc
    
#     cmdclass['build_sphinx'] = BuildDoc
#     command_options['build_sphinx'] = {
#         'project': ('setup.py', name),
#         'version': ('setup.py', version),
#         'release': ('setup.py', release),
#         'source_dir': ('setup.py', 'docs')
#     }

np_setup(
    name=name,
    package_dir={
        'shakermaker': 'shakermaker'
    },
    packages=[
        "shakermaker",
        "shakermaker.cm_library",
        "shakermaker.sl_extensions",
        "shakermaker.slw_extensions",
        "shakermaker.stf_extensions",
        "shakermaker.tools",
        "shakermaker.ffsp", 
    ],
    package_data={
        'shakermaker.ffsp': ['ffsp_core.cpython-*.so', 'ffsp_core.cpython-*.pyd', '*.f90', '*.f', 'makefile', 'Makefile_f2py', 'ffsp.pyf'], 
    },
    ext_modules=ext_modules,
    version=version,
    description="README.md",
    author=author,
    author_email="info@joseabell.com",
    url="http://www.joseabell.com",
    download_url="tbd",
    keywords=["earthquake", "engineering", "drm", "simulation"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Fortran",
        "Development Status :: Released",
        "Environment :: Other Environment",
        "Intended Audience :: Savvy Earthquake Engineers",
        "License :: GPL",
        "Operating System :: OS Independent",
        "Topic :: TBD",
        "Topic :: TBD2",
    ],
    long_description="""\
        shakermaker
        -------------------------------------
        
        Create realistic seismograms!
        
        """,
    cmdclass=cmdclass,
    command_options=command_options,
)