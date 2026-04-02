"""
ShakerMaker setup.py - Compatible with NumPy 1.x and 2.x

This setup uses subprocess to call f2py directly instead of numpy.distutils
(which was removed in NumPy 2.0).
"""

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
from setuptools.command.egg_info import egg_info as _egg_info
import subprocess
import sys
import os
import shutil
import glob
import sysconfig

name = "shakermaker"
version = "1.4"
release = "0.01"
author = "Jose A. Abell, Jorge Crempien D., and Matias Recabarren"

on_rtd = os.environ.get('READTHEDOCS') == 'True'


def get_ext_suffix():
    """Get the extension suffix for compiled modules (e.g., .cpython-310-x86_64-linux-gnu.so)"""
    return sysconfig.get_config_var('EXT_SUFFIX') or '.so'


def compile_core_module(build_dir=None):
    """
    Compile the core Fortran module using f2py.
    
    This replaces numpy.distutils.Extension with direct f2py calls,
    which works with both NumPy 1.x and 2.x.
    """
    core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shakermaker', 'core')
    
    # Source files
    fortran_sources = [
        "subgreen.f", "subgreen2.f", "subfk.f", "subfocal.f", "subtrav.f",
        "tau_p.f", "kernel.f", "prop.f", "source.f", "bessel.f", "haskell.f",
    ]
    c_sources = ["fft.c", "Complex.c"]
    
    # Clean old compiled files
    for pattern in ['core.cpython-*.so', 'core.cp*.pyd', 'coremodule.c', '*.o', '*.mod']:
        for f in glob.glob(os.path.join(core_dir, pattern)):
            try:
                os.remove(f)
                print(f"[core] removed old: {os.path.basename(f)}")
            except OSError:
                pass
    
    # Platform-specific compiler flags
    if sys.platform == 'win32':
        f77_flags = "/Qopenmp /extend-source:132"
        f90_flags = "/Qopenmp"
        fcompiler = "--fcompiler=intelvem"
        extra_link = []
    else:
        f77_flags = "-ffixed-line-length-132 -fPIC -O2 -Wno-all -fopenmp"
        f90_flags = "-fPIC -O2 -Wno-all -fopenmp"
        fcompiler = "--fcompiler=gnu95"
        extra_link = ["-lgomp"]
    
    # Build f2py command
    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "core.pyf",
        *fortran_sources,
        *c_sources,
        "-c",
        fcompiler,
        f"--f77flags={f77_flags}",
        f"--f90flags={f90_flags}",
        "-m", "core",
    ]
    
    # Add link flags
    for flag in extra_link:
        cmd.append(flag)
    
    print(f"[core] Compiling Fortran extension with NumPy f2py...")
    print(f"[core] Working directory: {core_dir}")
    print(f"[core] Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=core_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[core] STDOUT:", result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
        print("[core] STDERR:", result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
        raise RuntimeError(f"[core] f2py compilation failed with exit code {result.returncode}")
    
    # Find compiled file
    ext_suffix = get_ext_suffix()
    so_files = glob.glob(os.path.join(core_dir, f'core*{ext_suffix}'))
    if not so_files:
        # Try alternative patterns
        so_files = glob.glob(os.path.join(core_dir, 'core.cpython-*.so'))
        so_files.extend(glob.glob(os.path.join(core_dir, 'core.cp*.pyd')))
    
    if not so_files:
        print("[core] STDOUT:", result.stdout)
        print("[core] STDERR:", result.stderr)
        raise RuntimeError("[core] Compilation succeeded but no .so/.pyd file found")
    
    so_file = so_files[0]
    print(f"[core] Compiled: {os.path.basename(so_file)}")
    
    # Copy to shakermaker package directory
    dest = os.path.join(os.path.dirname(core_dir), os.path.basename(so_file))
    shutil.copy2(so_file, dest)
    print(f"[core] Copied to: {dest}")
    
    return so_file


def compile_ffsp_module():
    """Compile FFSP wrapper module using f2py."""
    ffsp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shakermaker', 'ffsp')
    
    fortran_sources = [
        "ffsp_wrapper.f90", "ffsp_comm.f90", "spfield_n.f90",
        "dcf_subs_1.f90", "slip_rate.f90", "ffsp_tool.f",
    ]
    
    # Clean old compiled files
    for pattern in ['ffsp_core.cpython-*.so', 'ffsp_core.cp*.pyd']:
        for f in glob.glob(os.path.join(ffsp_dir, pattern)):
            try:
                os.remove(f)
                print(f"[ffsp] removed old: {os.path.basename(f)}")
            except OSError:
                pass
    
    if sys.platform == 'win32':
        cmd = [
            sys.executable, "-m", "numpy.f2py",
            "-c", "ffsp.pyf",
            *fortran_sources,
            "--fcompiler=intelvem",
            "--f90flags=/Qopenmp",
            "--f77flags=/Qopenmp",
            "-m", "ffsp_core",
        ]
    else:
        cmd = [
            sys.executable, "-m", "numpy.f2py",
            "-c", "ffsp.pyf",
            *fortran_sources,
            "--f90flags=-O3 -fPIC",
            "--f77flags=-O3 -std=legacy -fPIC",
            "-m", "ffsp_core",
        ]
    
    print(f"[ffsp] Compiling FFSP wrapper...")
    result = subprocess.run(cmd, cwd=ffsp_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[ffsp] STDOUT:", result.stdout[-3000:])
        print("[ffsp] STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"[ffsp] compilation failed with exit code {result.returncode}")
    
    # Find compiled file
    ext_suffix = get_ext_suffix()
    so_files = glob.glob(os.path.join(ffsp_dir, f'ffsp_core*{ext_suffix}'))
    if not so_files:
        so_files = glob.glob(os.path.join(ffsp_dir, 'ffsp_core.cpython-*.so'))
        so_files.extend(glob.glob(os.path.join(ffsp_dir, 'ffsp_core.cp*.pyd')))
    
    if so_files:
        print(f"[ffsp] Compiled: {os.path.basename(so_files[0])}")
    else:
        raise RuntimeError("[ffsp] Compilation succeeded but no .so/.pyd file found")
    
    return so_files[0] if so_files else None


def install_compiled_modules_to_site_packages():
    """Copy compiled modules to site-packages after installation."""
    import site
    
    ext_suffix = get_ext_suffix()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find site-packages destination
    site_packages = None
    for sp in site.getsitepackages():
        dest = os.path.join(sp, 'shakermaker')
        if os.path.isdir(dest):
            site_packages = sp
            break
    
    if not site_packages:
        print("[install] Warning: shakermaker not found in site-packages")
        return
    
    # Copy core module
    core_src_dir = os.path.join(base_dir, 'shakermaker')
    core_dest_dir = os.path.join(site_packages, 'shakermaker')
    for f in glob.glob(os.path.join(core_src_dir, f'core*{ext_suffix}')):
        dest = os.path.join(core_dest_dir, os.path.basename(f))
        shutil.copy2(f, dest)
        print(f"[install] Copied {os.path.basename(f)} to site-packages")
    
    # Copy ffsp module
    ffsp_src_dir = os.path.join(base_dir, 'shakermaker', 'ffsp')
    ffsp_dest_dir = os.path.join(site_packages, 'shakermaker', 'ffsp')
    if os.path.isdir(ffsp_dest_dir):
        for f in glob.glob(os.path.join(ffsp_src_dir, f'ffsp_core*{ext_suffix}')):
            dest = os.path.join(ffsp_dest_dir, os.path.basename(f))
            shutil.copy2(f, dest)
            print(f"[install] Copied {os.path.basename(f)} to site-packages")


class BuildExtCommand(_build_ext):
    """Custom build_ext that compiles Fortran modules using f2py."""
    
    def run(self):
        if not on_rtd:
            compile_core_module()
            try:
                compile_ffsp_module()
            except Exception as e:
                print(f"[ffsp] Warning: FFSP compilation failed: {e}")
                print("[ffsp] FFSP features will not be available")
        _build_ext.run(self)


class InstallCommand(_install):
    """Custom install that compiles and installs Fortran modules."""
    
    def run(self):
        if not on_rtd:
            compile_core_module()
            try:
                compile_ffsp_module()
            except Exception as e:
                print(f"[ffsp] Warning: FFSP compilation failed: {e}")
        _install.run(self)
        if not on_rtd:
            install_compiled_modules_to_site_packages()


class DevelopCommand(_develop):
    """Custom develop that compiles Fortran modules for development mode."""
    
    def run(self):
        if not on_rtd:
            compile_core_module()
            try:
                compile_ffsp_module()
            except Exception as e:
                print(f"[ffsp] Warning: FFSP compilation failed: {e}")
        _develop.run(self)


class EggInfoCommand(_egg_info):
    """Custom egg_info that ensures modules are compiled."""
    
    def run(self):
        # Don't compile during egg_info - it's just for metadata
        _egg_info.run(self)


# Determine which extension files to include in package_data
def get_package_data():
    """Get list of compiled extension files to include."""
    ext_patterns = [
        'core.cpython-*.so',
        'core.cp*.pyd', 
        'core*.so',
    ]
    ffsp_patterns = [
        'ffsp_core.cpython-*.so',
        'ffsp_core.cp*.pyd',
        '*.f90', '*.f', 
        'makefile', 'Makefile_f2py', 'ffsp.pyf',
    ]
    return {
        'shakermaker': ext_patterns,
        'shakermaker.ffsp': ffsp_patterns,
    }


setup(
    name=name,
    version=version,
    description="Create realistic seismograms using the frequency-wavenumber method",
    long_description="""
ShakerMaker
-----------

Create realistic seismograms for earthquake engineering applications
using the frequency-wavenumber method.

Features:
- Point source and extended fault models
- Multiple crustal velocity models
- DRM (Domain Reduction Method) support
- HDF5 output format
- MPI parallel support
""",
    author=author,
    author_email="info@joseabell.com",
    url="https://github.com/jaabell/ShakerMaker",
    download_url="https://github.com/jaabell/ShakerMaker",
    keywords=["earthquake", "engineering", "drm", "simulation", "seismology"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Fortran",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    package_data=get_package_data(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "h5py>=3.0",
        "numba>=0.50",
    ],
    extras_require={
        "gpu": ["cupy>=10.0"],
        "mpi": ["mpi4py>=3.0"],
        "dev": ["pytest", "matplotlib"],
    },
    cmdclass={
        'build_ext': BuildExtCommand,
        'install': InstallCommand,
        'develop': DevelopCommand,
        'egg_info': EggInfoCommand,
    },
    zip_safe=False,
)
