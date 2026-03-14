# ShakerMaker — Windows Compilation Files

This repository contains the automated scripts to compile and install **ShakerMaker 1.0** on Windows 11 using Python 3.10, Intel oneAPI compilers, and Visual Studio 2022.

---

## Repository Structure

```
SCRIPTS/
├── shakermaker.cfg               ← Central configuration file (edit this first)
├── 00_shakermaker_common.ps1     ← Shared functions used by all scripts (do not edit)
├── 01_shakermaker_setup.ps1      ← Step 1: Install prerequisites and create virtual environment
├── 02_shakermaker_junction.ps1   ← Step 2: Create junction to the ShakerMaker repository
├── 03_shakermaker_build.ps1      ← Step 3: Compile ShakerMaker and configure DLL loading
└── RUN_ME.bat                    ← Main launcher with interactive menu
```

---

## Requirements

Before running anything, make sure your machine has internet access and enough disk space. The full installation requires approximately 20 GB for Visual Studio 2022 and the Intel oneAPI toolkits.

| Tool | Version | Purpose |
|------|---------|---------|
| Windows 11 | Any | Operating system |
| Python | 3.10.x | Runtime and build environment |
| Git | 2.x | Repository access |
| Visual Studio 2022 Community | 17.x | C++ compiler (`cl.exe`) |
| Intel oneAPI Base Toolkit | 2025.x | Math Kernel Library (MKL) |
| Intel oneAPI HPC Toolkit | 2025.x | Fortran compiler (`ifx`) |
| NumPy | 1.26.4 | Build dependency — do NOT use 2.x |

---

## Quick Start

### Step 0 — Edit the configuration file

Open `shakermaker.cfg` and fill in your settings. This is the only file you need to edit. Everything else reads from here automatically.

```ini
# Python version
PYTHON_VERSION      = 3.10
PYTHON_FULL_VERSION = 3.10.9

# Name of the virtual environment to create
VENV_NAME           = shakermaker_venv

# Base folder for the virtual environment
# Result will be: C:\Users\your_username\shakermaker_venv
VENV_BASE           = C:\Users\%USERNAME%

# Folder where the junction will be created (must have no spaces)
COMPILER_DIR        = C:\shakermaker_compiler

# Full path to your ShakerMaker source repository
# This path may contain spaces — the junction script handles it
SHAKERMAKER_SOURCE  = C:\path\to\your\ShakerMaker_OP

# Python packages to install in the virtual environment
PYTHON_DEPS = numpy==1.26.4 setuptools wheel h5py mpi4py matplotlib scipy
```

> **Important:** Keep `VENV_NAME` short (e.g. `sm_venv`, `shaker`). Long names cause Windows command-line length errors during compilation.

---

### Step 0.1 — Unblock the scripts

Windows blocks scripts downloaded from the internet. Before running anything, open PowerShell and unblock all files in the SCRIPTS folder:

```powershell
Get-ChildItem "C:\path\to\SCRIPTS" | Unblock-File
```

You must do this every time you download or update the scripts from GitHub or Dropbox.

---

### Step 0.2 — Run the launcher

Double-click `RUN_ME.bat` or open it from PowerShell. You will see an interactive menu:

```
+==========================================================+
|         ShakerMaker - Windows Setup Menu                |
+==========================================================+
|                                                          |
|   [1]  Step 1 - Install Prerequisites                   |
|   [2]  Step 2 - Create Junction                         |
|   [3]  Step 3 - Build and Compile                       |
|   [4]  Run All Steps in Order (1 then 2 then 3)         |
|   [Q]  Quit                                             |
|                                                          |
+==========================================================+
```

For a fresh installation on a new machine, choose **[4] Run All Steps in Order**.

---

## What Each Script Does

### `shakermaker.cfg` — Central Configuration

The single source of truth for all scripts. All paths, version numbers, virtual environment names, and Python dependencies are defined here. No other file needs to be edited.

---

### `00_shakermaker_common.ps1` — Shared Functions

Loaded automatically by the other three scripts via dot-sourcing. Contains:

- `Read-ShakerConfig` — parses `shakermaker.cfg` into a hashtable, expands `%USERNAME%`, and validates required keys
- `Write-ShakerLog` — writes timestamped log entries
- `Print-Header`, `Print-OK`, `Print-SKIP`, `Print-FAIL`, `Print-INFO` — colored console output helpers

Do not run this file directly.

---

### `01_shakermaker_setup.ps1` — Prerequisites and Virtual Environment

**What it does:**

1. Reads configuration from `shakermaker.cfg`
2. Checks whether each required tool is already installed:
   - Git
   - Python 3.10
   - Visual Studio 2022 Community
   - Intel oneAPI Base Toolkit
   - Intel oneAPI HPC Toolkit
3. Shows a checklist with `[OK]` for installed tools and `[--]` for missing ones
4. Checks whether the virtual environment already exists — if it does, it reuses it; if not, it creates it
5. Checks which Python packages are already installed and at the correct version
6. Asks for confirmation before installing anything
7. Installs only the missing components via `winget`
8. Creates the virtual environment using `py -3.10 -m venv`
9. Upgrades `pip`
10. Installs missing Python packages
11. Verifies all packages are correctly installed

**Log file:** `C:\Users\your_username\shakermaker_setup.log`

> **Note:** Visual Studio and Intel oneAPI are large downloads (5–10 GB each). The script waits for each installer to finish before proceeding.

---

### `02_shakermaker_junction.ps1` — Junction Setup

**Why this is needed:**

The ShakerMaker source repository may live inside a path with spaces (for example inside Dropbox or OneDrive). The Windows compiler tools used during the build cannot handle spaces in paths and will fail silently or with cryptic errors. The solution is to create a Windows junction — a filesystem link — from a clean path with no spaces to the actual repository location.

**What it does:**

1. Reads `COMPILER_DIR` and `SHAKERMAKER_SOURCE` from `shakermaker.cfg`
2. Checks if a junction already exists at `COMPILER_DIR\ShakerMaker` — if it does, asks whether to replace it
3. If `SHAKERMAKER_SOURCE` is defined in the config, offers to use it directly; otherwise asks the user to enter the path interactively
4. Validates that the path exists and contains `setup.py`
5. Creates the `COMPILER_DIR` folder if it does not exist
6. Creates the junction using `mklink /J`
7. Writes the validated `SHAKERMAKER_SOURCE` back to `shakermaker.cfg` so the build script can find it
8. Verifies the junction is a valid reparse point and that `setup.py` is reachable through it

**Result:** `C:\shakermaker_compiler\ShakerMaker\` → your actual repository

**Log file:** `C:\Users\your_username\shakermaker_junction.log`

---

### `03_shakermaker_build.ps1` — Build and Compile

**What it does:**

1. Reads all paths and settings from `shakermaker.cfg`
2. Pre-build checks: verifies the junction, virtual environment, Visual Studio, and Intel `ifx` compiler are all present
3. Creates the `ifort.bat` wrapper — Intel oneAPI 2025.x ships `ifx.exe` but NumPy's f2py looks for `ifort.exe`; this wrapper redirects `ifort` calls to `ifx`
4. Launches a CMD subprocess (not PowerShell) with the full build environment initialized:
   - Calls `VsDevCmd.bat` to set up the MSVC compiler
   - Calls `setvars.bat` to initialize Intel oneAPI
   - Activates the virtual environment
   - Sets the Intel `LIB` and `PATH` manually (required because `setvars.bat` sometimes does not set them correctly)
   - Runs `python setup.py install` from the junction path
5. Creates `sitecustomize.py` in both the virtual environment site-packages and the Python base site-packages — this file is loaded automatically by Python at every startup and configures the Intel MPI environment and DLL paths so that ShakerMaker can be imported from Jupyter and VS Code without any manual setup
6. Runs a smoke test (`from shakermaker import core`) inside an initialized CMD to verify the build succeeded

**Why CMD and not PowerShell for the build:**

`VsDevCmd.bat` and `setvars.bat` modify the PATH and environment variables of the shell that calls them. This only works correctly in CMD. The script handles this by launching a temporary `.bat` file as a subprocess.

**Log file:** `C:\Users\your_username\shakermaker_build.log`

> **Note on first-time compilation:** If the virtual environment name is long, the first build attempt may fail with `WinError 206: The filename or extension is too long`. This is a Windows limitation on command-line length. Simply run Step 3 again — the second attempt uses the compilation cache and succeeds. To avoid this entirely, use a short virtual environment name in `shakermaker.cfg`.

---

## After Installation

Once the build completes successfully, ShakerMaker is ready to use. The `sitecustomize.py` created by Step 3 handles DLL loading automatically at every Python startup — no manual environment initialization is needed.

To use ShakerMaker in Jupyter or VS Code, select the virtual environment you created as your Python kernel. The imports will work directly:

```python
from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.stationlist import StationList
import numpy as np
import matplotlib.pyplot as plt
```

To run models with MPI from the command line, open a **fresh CMD** window (not PowerShell) and initialize the environment manually:

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
C:\Users\your_username\your_venv\Scripts\activate.bat
chcp 65001
cd "C:\path\to\your\model"
mpiexec -n 10 python your_script.py
```

---

## Troubleshooting

### Scripts are blocked and cannot run

Run this in PowerShell from the SCRIPTS folder:

```powershell
Get-ChildItem "C:\path\to\SCRIPTS" | Unblock-File
```

### Build fails with `WinError 206: The filename or extension is too long`

Your virtual environment name is too long. Use a short name in `shakermaker.cfg` such as `sm_venv` and run Step 3 again.

### `DLL load failed while importing core`

The `sitecustomize.py` was not created or was deleted. Run Step 3 again — it recreates the file automatically.

### `CompilerNotFound: intelvem`

The `ifort.bat` wrapper was not created or is not on PATH. Run Step 3 again.

### `The input line is too long` in CMD

The PATH variable overflowed from running `VsDevCmd.bat` or `setvars.bat` multiple times in the same CMD window. Close the window and open a fresh one.

### Build fails on first attempt but succeeds on second

This is a known Windows path-length issue with longer virtual environment names. Run Step 3 a second time — it will use the compilation cache and complete successfully.

---

## Reinstalling ShakerMaker

If you need to reinstall after modifying the source code, simply run Step 3 again. It will recompile and reinstall, and recreate `sitecustomize.py` automatically.

---

## Tested Configuration

| Component | Version |
|-----------|---------|
| Windows | 11 (26200.x) |
| Python | 3.10.11 |
| Visual Studio 2022 Community | 17.14.28 |
| Intel oneAPI Base Toolkit | 2025.x |
| Intel oneAPI HPC Toolkit | 2025.x |
| NumPy | 1.26.4 |
| ShakerMaker | 1.0 (branch: optimize_process) |