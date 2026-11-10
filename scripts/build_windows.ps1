$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$RequiredFiles = @(
    "data\dataset.zip",
    "third_party\DiffSinger\requirements.txt",
    "third_party\DiffSinger\checkpoints\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt"
)
foreach ($RelativePath in $RequiredFiles) {
    if (-not (Test-Path (Join-Path $Root $RelativePath) -PathType Leaf)) {
        throw "Required app asset is missing: $RelativePath. Prepare the project data, DiffSinger source, and vocoder before building."
    }
}
$HasCheckpointModel = Get-ChildItem (Join-Path $Root "third_party\DiffSinger\checkpoints") -Directory -ErrorAction SilentlyContinue |
    Where-Object {
        (Test-Path (Join-Path $_.FullName "config.yaml")) -and
        (Get-ChildItem $_.FullName -Filter "model_ckpt_steps_*.ckpt" -File -ErrorAction SilentlyContinue | Select-Object -First 1)
    } |
    Select-Object -First 1
$HasOnnxModel = @(
    Get-ChildItem (Join-Path $Root "third_party\DiffSinger\artifacts") -Filter "*.onnx" -File -Recurse -ErrorAction SilentlyContinue
    Get-ChildItem (Join-Path $Root "data\acoustic_models") -Filter "*.onnx" -File -Recurse -ErrorAction SilentlyContinue
) | Select-Object -First 1
if (-not $HasCheckpointModel -and -not $HasOnnxModel) {
    throw "No acoustic Checkpoint or ONNX model was found. Add at least one model before building."
}

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    & py -3.12 -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 is required. Install 64-bit Python 3.12 and retry."
    }
    $BasePython = "py"
    $BasePythonArgs = @("-3.12")
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Install 64-bit Python 3.12 and add it to PATH."
    }
    $PythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($PythonVersion -ne "3.12") {
        throw "Python 3.12 is required; found $PythonVersion."
    }
    $BasePython = "python"
    $BasePythonArgs = @()
}

$BuildEnvironment = Join-Path $Root ".build-windows-venv"
$EnvironmentPython = Join-Path $BuildEnvironment "Scripts\python.exe"
if (-not (Test-Path $EnvironmentPython)) {
    & $BasePython @BasePythonArgs -m venv $BuildEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Windows build environment."
    }
}

& $EnvironmentPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
& $EnvironmentPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Could not install CPU-only PyTorch." }
& $EnvironmentPython -m pip install -r (Join-Path $Root "third_party\DiffSinger\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install DiffSinger requirements." }
& $EnvironmentPython -m pip install onnxruntime "pyinstaller>=6.11,<7"
if ($LASTEXITCODE -ne 0) { throw "Could not install ONNX Runtime or PyInstaller." }

& $EnvironmentPython -c "import struct, onnxruntime as ort, torch; assert struct.calcsize('P') == 8; assert 'CPUExecutionProvider' in ort.get_available_providers(); assert torch.version.cuda is None; print(f'ONNX Runtime {ort.__version__}; PyTorch {torch.__version__}; 64-bit CPU only')"
if ($LASTEXITCODE -ne 0) { throw "The CPU inference runtime check failed." }

& $EnvironmentPython -m PyInstaller --noconfirm --clean (Join-Path $Root "scripts\windows_app.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

$App = Join-Path $Root "dist\DiffSingerGUI\DiffSingerGUI.exe"
if (-not (Test-Path $App)) {
    throw "The executable was not created: $App"
}
& $App --_self-test-runtime
if ($LASTEXITCODE -ne 0) { throw "The packaged ONNX Runtime self-test failed." }

Write-Host ""
Write-Host "Build complete: $App"
Write-Host "Keep the whole dist\DiffSingerGUI folder together when moving the app."
