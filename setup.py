from setuptools import setup
from numpy.distutils.core import Extension, setup as np_setup
import importlib.util
import os

from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess

on_rtd = os.environ.get('READTHEDOCS') == 'True'

name = "shakermaker"
version = "0.1"
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
        ext1 = Extension(
            name='shakermaker.core',
            sources=srcs,
            extra_f77_compile_args=["-ffixed-line-length-132", "-Wno-tabs", "-Wno-unused-dummy-argument", "-fPIC", "-fopenmp"],
            extra_compile_args=["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
            extra_link_args=["-fopenmp"]
        )

    ext_modules = [ext1]

# ========== FFSP compilation support
def compile_ffsp():
    """Compile FFSP executable using makefile"""
    ffsp_dir = os.path.join(os.path.dirname(__file__), 'shakermaker', 'ffsp')    
    try:
        subprocess.run(['make', 'clean'], cwd=ffsp_dir, check=False, capture_output=True)
        subprocess.run(['make', 'all'], cwd=ffsp_dir, check=True, capture_output=True)        
        ffsp_exec = os.path.join(ffsp_dir, 'ffsp_dcf_v2')
        if os.path.exists(ffsp_exec):
            os.chmod(ffsp_exec, 0o755)
            print("OK: FFSP compiled")            
    except subprocess.CalledProcessError:
        print("X: FFSP compilation failed")
        raise
class CustomInstallCommand(install):
    """Custom install command that compiles FFSP first"""
    def run(self):
        compile_ffsp()
        install.run(self)
        self.set_ffsp_permissions() # para permiso chmod +x

    def set_ffsp_permissions(self):
        import site
        for site_pkg in site.getsitepackages() + [site.getusersitepackages()]:
            ffsp_exec = os.path.join(site_pkg, 'shakermaker', 'ffsp', 'ffsp_dcf_v2')
            if os.path.exists(ffsp_exec):
                os.chmod(ffsp_exec, 0o755)
                print(f"Permission OK to: {ffsp_exec}")
                break

class CustomDevelopCommand(develop):
    """Custom develop command that compiles FFSP first"""
    def run(self):
        compile_ffsp()
        develop.run(self)
        ffsp_exec = os.path.join('shakermaker', 'ffsp', 'ffsp_dcf_v2')
        if os.path.exists(ffsp_exec):
            os.chmod(ffsp_exec, 0o755)
            # print(f"Permission to: {ffsp_exec}")
# ============================================================

# Check for sphinx
found_sphinx = importlib.util.find_spec('sphinx') is not None

# ============================================================
# cmdclass = {}  me tomo esta variable. Consultar si es pertinente
cmdclass = {
    'install': CustomInstallCommand,
    'develop': CustomDevelopCommand,
}
# ============================================================

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
        'shakermaker.ffsp': ['ffsp_dcf_v2', '*.f90', '*.f', 'makefile'], 
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