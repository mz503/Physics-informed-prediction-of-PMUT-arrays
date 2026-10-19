param(
    [string]$PythonPath,
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPath = Join-Path $ProjectRoot ".venv-runall"
$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"

Set-Location $ProjectRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path $EnvironmentPython)) {
    if (-not $PythonPath) {
        $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($PythonLauncher) {
            & $PythonLauncher.Source -3.12 -m venv $EnvironmentPath
            if ($LASTEXITCODE -ne 0) {
                throw "Could not create the project environment with Python 3.12. Pass -PythonPath with a Python 3.12 executable."
            }
        }
        else {
            $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if (-not $PythonCommand) {
                throw "Python 3.12 was not found. Install Python 3.12 or pass -PythonPath with its executable."
            }
            $PythonPath = $PythonCommand.Source
        }

        if (-not (Test-Path $EnvironmentPython)) {
            if (-not $PythonPath) {
                $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
                if ($PythonLauncher) {
                    $PythonPath = (& $PythonLauncher.Source -3.12 -c "import sys; print(sys.executable)").Trim()
                }
            }
            if (-not $PythonPath) {
                throw "Could not resolve the Python 3.12 executable used to create the project environment."
            }
            & $PythonPath -m venv $EnvironmentPath
            if ($LASTEXITCODE -ne 0) {
                throw "Could not create the project environment. Check that -PythonPath points to Python 3.12."
            }
        }
    }
    else {
        & $PythonPath -m venv $EnvironmentPath
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the project environment. Check that -PythonPath points to Python 3.12."
        }
    }
}

if (-not (Test-Path $EnvironmentPython)) {
    throw "The project environment does not contain its Python executable: $EnvironmentPython"
}

$PythonVersion = & $EnvironmentPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($PythonVersion.Trim() -ne "3.12") {
    throw "The project environment uses Python $PythonVersion; the pinned dependencies require Python 3.12."
}

Write-Host "Installing pinned project dependencies..."
& $EnvironmentPython -m pip install --disable-pip-version-check -r $RequirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed. Resolve the pip error above and rerun this script."
}

& $EnvironmentPython -c "import numpy, pandas, sklearn, scipy, matplotlib, plotly, torch, openpyxl"
if ($LASTEXITCODE -ne 0) {
    throw "A required project dependency could not be imported after installation."
}

if ($SetupOnly) {
    Write-Host "Project environment and pinned dependencies are ready."
    exit 0
}

$Scripts = @(
    "00_export_data.py",
    "01_train_cache.py",
    "04_metrics_table.py",
    "06_progressive_training.py",
    "07_regime_aware_training.py",
    "08_progressive_images.py"
)

Write-Host "Using Python $PythonVersion at $EnvironmentPython"
foreach ($Script in $Scripts) {
    $ScriptPath = Join-Path $ProjectRoot "scripts\$Script"
    Write-Host "`n==> $Script"
    & $EnvironmentPython $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Script failed with exit code $LASTEXITCODE."
    }
}

Write-Host "`nAll data preparation, training, metrics, explorers, and figures completed."
