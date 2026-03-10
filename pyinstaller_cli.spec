# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

hidden = [
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends._tkagg",
]

datas = collect_data_files("app", includes=["*.json"])

a = Analysis(
    ["app/cli.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["build_hooks/pyi_rth_fitter_multiprocessing.py"],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="FitterCLI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    onefile=True,
)
