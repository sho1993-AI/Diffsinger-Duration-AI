#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFSINGER_COMMIT="cb7869b8733a3882679c328994ee162845092cff"
DIFFSINGER_DIR="${DIFFSINGER_DIR:-$ROOT_DIR/third_party/DiffSinger}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-cpu312}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))'; then
    printf 'Use Python 3.10, 3.11, or 3.12 for this DiffSinger environment.\n' >&2
    exit 1
fi

if [[ -d "$DIFFSINGER_DIR/.git" ]]; then
    current_commit="$(git -C "$DIFFSINGER_DIR" rev-parse HEAD)"
    if [[ "$current_commit" != "$DIFFSINGER_COMMIT" ]]; then
        printf 'Existing DiffSinger checkout is at %s; expected %s.\n' \
            "$current_commit" "$DIFFSINGER_COMMIT" >&2
        exit 1
    fi
elif [[ -e "$DIFFSINGER_DIR" ]]; then
    printf 'DiffSinger destination exists but is not a git checkout: %s\n' \
        "$DIFFSINGER_DIR" >&2
    exit 1
else
    mkdir -p "$(dirname "$DIFFSINGER_DIR")"
    git clone https://github.com/openvpi/DiffSinger.git "$DIFFSINGER_DIR"
    git -C "$DIFFSINGER_DIR" checkout "$DIFFSINGER_COMMIT"
fi

if [[ -x "$VENV_DIR/bin/python" ]] && ! "$VENV_DIR/bin/python" -c \
    'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))'; then
    printf 'Existing environment uses an unsupported Python version: %s\n' "$VENV_DIR" >&2
    printf 'Set VENV_DIR to a new path and rerun this script.\n' >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if "$PYTHON_BIN" -c 'import ensurepip' >/dev/null 2>&1; then
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    elif command -v virtualenv >/dev/null 2>&1; then
        virtualenv --python "$PYTHON_BIN" "$VENV_DIR"
    else
        printf 'Python venv support is missing; install python3.12-venv or virtualenv.\n' >&2
        exit 1
    fi
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
"$VENV_DIR/bin/python" -m pip install -r "$DIFFSINGER_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m pip install onnxruntime tensorboard

"$VENV_DIR/bin/python" -c \
    'import lightning, onnxruntime, parselmouth, torch, yaml; print(f"PyTorch {torch.__version__}; ONNX Runtime {onnxruntime.__version__}; CUDA build: {torch.version.cuda}"); assert torch.version.cuda is None'
printf '\nCPU environment ready. Activate it with:\n  source "%s/bin/activate"\n' "$VENV_DIR"
printf 'DiffSinger source: %s\n' "$DIFFSINGER_DIR"