# ==============================================================================
#  ShakerMaker - Step 3: Build and Compile
#  Version : 2.0
# ==============================================================================
#  Initializes the compiler environment, builds ShakerMaker, creates
#  sitecustomize.py for DLL loading, and runs a smoke test.
#
#  IMPORTANT: This script must be run from CMD, not PowerShell, because
#  VsDevCmd.bat and setvars.bat modify the CMD environment. This script
#  launches a CMD subprocess automatically to handle this.
#
#  Run from RUN_ME.bat or manually:
#      PowerShell -ExecutionPolicy Bypass -File .\shakermaker_build.ps1
# ==============================================================================

. "$PSScriptRoot\00_shakermaker_common.ps1"

# --- Load config --------------------------------------------------------------
$cfg             = Read-ShakerConfig "$PSScriptRoot\shakermaker.cfg"
$PYTHON_VERSION      = $cfg["PYTHON_VERSION"]
$PYTHON_FULL_VERSION = $cfg["PYTHON_FULL_VERSION"]
$VENV_NAME           = $cfg["VENV_NAME"]
$VENV_BASE           = $cfg["VENV_BASE"]
$COMPILER_DIR        = $cfg["COMPILER_DIR"]
$VENV_PATH           = "$VENV_BASE\$VENV_NAME"
$JUNCTION_PATH       = "$COMPILER_DIR\ShakerMaker"
$LOG_FILE            = "$VENV_BASE\shakermaker_build.log"

$VS_DEV_CMD  = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
$SETVARS_BAT = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
$IFX_EXE     = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifx.exe"
$IFORT_BAT   = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifort.bat"
$INTEL_LIB   = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"
$INTEL_MPI   = "C:\Program Files (x86)\Intel\oneAPI\mpi\latest\bin"
$INTEL_BIN   = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
$SITE_PKG    = "$VENV_PATH\Lib\site-packages"

function Log { param([string]$m) Write-ShakerLog $LOG_FILE $m }

# --- Banner -------------------------------------------------------------------
Clear-Host
Write-Host ""
Write-Host "  +---------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "  |      ShakerMaker - Step 3: Build and Compile            |" -ForegroundColor Cyan
Write-Host "  |      Python $PYTHON_FULL_VERSION  |  Venv: $VENV_NAME" -ForegroundColor Cyan
Write-Host "  +---------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Log "Build started - user: $env:USERNAME - Python: $PYTHON_FULL_VERSION"

# ==============================================================================
Print-Header "STEP 1 - Pre-Build Checks"
# ==============================================================================

$abort = $false

# Junction
if (Test-Path "$JUNCTION_PATH\setup.py") {
    Print-OK "Junction found: $JUNCTION_PATH"
    Log "[OK] Junction OK"
} else {
    Print-FAIL "Junction not found or setup.py missing at $JUNCTION_PATH"
    Print-INFO "Run Step 2 (Junction Setup) first."
    Log "[!!] Junction missing"
    $abort = $true
}

# Venv
if (Test-Path "$VENV_PATH\Scripts\python.exe") {
    Print-OK "Virtual environment found: $VENV_PATH"
    Log "[OK] Venv OK"
} else {
    Print-FAIL "Virtual environment not found at $VENV_PATH"
    Print-INFO "Run Step 1 (Prerequisites Setup) first."
    Log "[!!] Venv missing"
    $abort = $true
}

# Visual Studio
if (Test-Path $VS_DEV_CMD) {
    Print-OK "Visual Studio 2022 found"
    Log "[OK] VS2022 OK"
} else {
    Print-FAIL "VsDevCmd.bat not found at $VS_DEV_CMD"
    Log "[!!] VS2022 missing"
    $abort = $true
}

# Intel ifx
if (Test-Path $IFX_EXE) {
    Print-OK "Intel ifx found"
    Log "[OK] ifx OK"
} else {
    Print-FAIL "ifx.exe not found at $IFX_EXE"
    Print-INFO "Run Step 1 and install Intel oneAPI HPC Toolkit."
    Log "[!!] ifx missing"
    $abort = $true
}

if ($abort) {
    Write-Host ""
    Print-FAIL "One or more required components are missing. Cannot proceed."
    Log "Build aborted - missing components"
    Read-Host "  Press Enter to exit"
    exit 1
}

# ==============================================================================
Print-Header "STEP 2 - Create ifort.bat Wrapper"
# ==============================================================================

# ifort.bat is needed because f2py looks for ifort but Intel 2025 only has ifx
if (Test-Path $IFORT_BAT) {
    Print-OK "ifort.bat wrapper already exists"
    Log "[OK] ifort.bat exists"
} else {
    try {
        "@ifx %*" | Set-Content $IFORT_BAT -Encoding ASCII
        Print-OK "ifort.bat wrapper created at $IFORT_BAT"
        Log "[OK] ifort.bat created"
    } catch {
        Print-FAIL "Failed to create ifort.bat - try running as Administrator"
        Print-INFO "Or manually run: echo @ifx %* > `"$IFORT_BAT`""
        Log "[!!] ifort.bat creation failed: $_"
        Read-Host "  Press Enter to exit"
        exit 1
    }
}

# ==============================================================================
Print-Header "STEP 3 - Run Build via CMD"
# ==============================================================================

Write-Host ""
Write-Host "  The build requires a CMD environment with VS and Intel compilers." -ForegroundColor White
Write-Host "  Launching build subprocess now..." -ForegroundColor Gray
Write-Host ""
Log "Launching CMD build subprocess"

# Build the CMD script as a temp file
$buildCmdScript = "$env:TEMP\shakermaker_build_cmd.bat"

$cmdContent = @"
@echo off
chcp 65001 >nul

echo.
echo  Initializing Visual Studio environment...
call "$VS_DEV_CMD" -arch=x64 -host_arch=x64
if errorlevel 1 (
    echo  [!!] VsDevCmd.bat failed
    exit /b 1
)

echo  Initializing Intel oneAPI environment...
call "$SETVARS_BAT" intel64
if errorlevel 1 (
    echo  [!!] setvars.bat failed
    exit /b 1
)

echo  Activating virtual environment...
call "$VENV_PATH\Scripts\activate.bat"

echo  Setting Intel LIB path...
set LIB=%LIB%;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib
echo  Setting Intel PATH...
set PATH=%PATH%;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin

echo  Running build...
cd /d "$JUNCTION_PATH"
python setup.py install
if errorlevel 1 (
    echo  [!!] Build failed
    exit /b 1
)

echo  [OK] Build completed successfully
exit /b 0
"@

$cmdContent | Set-Content $buildCmdScript -Encoding ASCII
Log "CMD build script written to $buildCmdScript"

# Run the CMD script and stream output
$process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$buildCmdScript`"" `
    -NoNewWindow -Wait -PassThru

if ($process.ExitCode -eq 0) {
    Print-OK "Build completed successfully"
    Log "[OK] Build subprocess exited 0"
} else {
    Print-FAIL "Build failed (exit code $($process.ExitCode))"
    Log "[!!] Build subprocess exited $($process.ExitCode)"
    Read-Host "  Press Enter to exit"
    exit 1
}

# Clean up temp file
Remove-Item $buildCmdScript -Force -ErrorAction SilentlyContinue

# ==============================================================================
Print-Header "STEP 4 - Create sitecustomize.py"
# ==============================================================================

# sitecustomize.py is auto-loaded by Python at every startup.
# It sets MPI environment variables and adds Intel DLL paths.
# Required in BOTH the venv site-packages AND the Python base site-packages
# so that Jupyter and VS Code find the DLLs regardless of how they launch.

$siteContent = @"
import os
os.environ["I_MPI_FABRICS"] = "shm"
os.environ["FI_PROVIDER"] = "sockets"
os.add_dll_directory(r"$INTEL_BIN")
os.add_dll_directory(r"$INTEL_MPI")
"@

# Find Python base site-packages
$pythonBase = & "$VENV_PATH\Scripts\python.exe" -c "import sys; print([p for p in sys.path if 'AppData' in p and 'site-packages' in p][0])" 2>&1
if (-not $pythonBase -or $pythonBase -notmatch "site-packages") {
    $pythonBase = & "$VENV_PATH\Scripts\python.exe" -c "import sysconfig; print(sysconfig.get_path('purelib', 'nt'))" 2>&1
}

# Write to venv site-packages
$siteCustomizeVenv = "$SITE_PKG\sitecustomize.py"
try {
    [System.IO.File]::WriteAllText($siteCustomizeVenv, $siteContent)
    Print-OK "sitecustomize.py created in venv: $siteCustomizeVenv"
    Log "[OK] sitecustomize.py venv created"
} catch {
    Print-FAIL "Failed to create sitecustomize.py in venv: $_"
    Log "[!!] sitecustomize.py venv failed: $_"
}

# Write to Python base site-packages
$pythonExeBase = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExeBase) {
    $pythonExeBase = & "$VENV_PATH\Scripts\python.exe" -c "import sys; print(sys._base_executable)" 2>&1
}
$pythonBaseLib = Split-Path $pythonExeBase
$siteCustomizeBase = "$pythonBaseLib\Lib\site-packages\sitecustomize.py"

try {
    [System.IO.File]::WriteAllText($siteCustomizeBase, $siteContent)
    Print-OK "sitecustomize.py created in Python base: $siteCustomizeBase"
    Log "[OK] sitecustomize.py base created"
} catch {
    Print-FAIL "Failed to create sitecustomize.py in Python base: $_"
    Log "[!!] sitecustomize.py base failed: $_"
}

# Verify venv
$written = Get-Content $siteCustomizeVenv -Raw
if ($written -match "I_MPI_FABRICS") {
    Print-OK "sitecustomize.py contents verified"
} else {
    Print-FAIL "sitecustomize.py contents look wrong - check manually"
    Log "[!!] sitecustomize.py verification failed"
}

# ==============================================================================
Print-Header "STEP 5 - Smoke Test"
# ==============================================================================

# Smoke test must run inside a CMD with VS + Intel + venv initialized.
# Running from PowerShell directly causes MPI to crash on init.

$smokeScript = "$env:TEMP\shakermaker_smoke.bat"

$smokeContent = @"
@echo off
chcp 65001 >nul
call "$VS_DEV_CMD" -arch=x64 -host_arch=x64 >nul 2>&1
call "$SETVARS_BAT" intel64 >nul 2>&1
call "$VENV_PATH\Scripts\activate.bat" >nul 2>&1
set LIB=%LIB%;$INTEL_LIB
set PATH=%PATH%;$INTEL_BIN
python -c "from shakermaker import core; print('core OK')"
if errorlevel 1 (
    echo [!!] core import failed
    exit /b 1
)
exit /b 0
"@

$smokeContent | Set-Content $smokeScript -Encoding ASCII
Write-Host "  Testing core import inside initialized CMD..." -ForegroundColor White

$smokeProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$smokeScript`"" `
    -NoNewWindow -Wait -PassThru

if ($smokeProcess.ExitCode -eq 0) {
    Print-OK "core import OK"
    Log "[OK] smoke test passed"
} else {
    Print-FAIL "core import failed - check DLL paths or MPI installation"
    Log "[!!] smoke test failed"
}
Remove-Item $smokeScript -Force -ErrorAction SilentlyContinue



# ==============================================================================
$line = "=" * 70
Write-Host ""
Write-Host $line -ForegroundColor Cyan
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "  Venv    : $VENV_PATH" -ForegroundColor White
Write-Host "  Build   : $JUNCTION_PATH" -ForegroundColor White
Write-Host "  Log     : $LOG_FILE" -ForegroundColor White
Write-Host ""
Write-Host "  IMPORTANT: Always activate from CMD, never from PowerShell." -ForegroundColor Red
Write-Host ""
Write-Host "  Open a fresh CMD window and run:" -ForegroundColor White
Write-Host "  call `"$VS_DEV_CMD`" -arch=x64 -host_arch=x64" -ForegroundColor Yellow
Write-Host "  call `"%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat`" intel64" -ForegroundColor Yellow
Write-Host "  $VENV_PATH\Scripts\activate.bat" -ForegroundColor Yellow
Write-Host $line -ForegroundColor Cyan
Write-Host ""
Log "Build script complete"
Read-Host "  Press Enter to exit"