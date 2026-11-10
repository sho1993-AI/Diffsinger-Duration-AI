from pathlib import Path

from PyInstaller.utils.hooks import collect_all


ROOT_DIR = Path(SPECPATH).resolve().parent.parent


def collect_tree(source: Path, destination: str) -> list[tuple[str, str]]:
    entries = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in {".git", "__pycache__", ".pytest_cache", "lightning_logs"} for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        entries.append((str(path), str(Path(destination) / relative.parent)))
    return entries


datas = []
binaries = []
hiddenimports = ["duration_postprocess"]
for package in ("librosa", "lightning", "onnxruntime", "parselmouth", "pyworld", "torch", "torchaudio"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)

datas.extend(collect_tree(ROOT_DIR / "scripts", "scripts"))
datas.extend(collect_tree(ROOT_DIR / "data", "data"))
datas.extend(collect_tree(ROOT_DIR / "third_party" / "DiffSinger", "third_party/DiffSinger"))

analysis = Analysis(
    [str(ROOT_DIR / "scripts" / "phoneme_viewer.py")],
    pathex=[str(ROOT_DIR / "scripts"), str(ROOT_DIR / "third_party" / "DiffSinger")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="DiffSingerGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    contents_directory=".",
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="DiffSingerGUI",
)
