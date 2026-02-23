from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Iterable, List

from app.runtime_config import set_timing_env_from_config


_PKG_TO_MODULE = {
    "pyyaml": "yaml",
    "pillow": "PIL",
    "python-dateutil": "dateutil",
}

_SKIP_RUNTIME = {"pyinstaller"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _requirements_path() -> Path:
    return _repo_root() / "requirements.txt"


def _parse_requirement_name(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("-"):
        return None
    s = s.split("#", 1)[0].strip()
    if not s:
        return None
    s = s.split(";", 1)[0].strip()
    m = re.match(r"^([A-Za-z0-9_.-]+)", s)
    if not m:
        return None
    name = m.group(1).lower()
    if name in _SKIP_RUNTIME:
        return None
    return name


def iter_runtime_requirements() -> List[str]:
    path = _requirements_path()
    if not path.exists():
        return []
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = _parse_requirement_name(line)
        if not name:
            continue
        if name not in names:
            names.append(name)
    return names


def _module_name_for_package(pkg: str) -> str:
    return _PKG_TO_MODULE.get(pkg, pkg.replace("-", "_"))


def find_missing_runtime_dependencies(packages: Iterable[str] | None = None) -> List[str]:
    pkgs = list(packages) if packages is not None else iter_runtime_requirements()
    missing: List[str] = []
    for pkg in pkgs:
        module_name = _module_name_for_package(pkg)
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(pkg)
    return missing


def preflight_or_raise() -> None:
    missing = find_missing_runtime_dependencies()
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: {}. Install with `python -m pip install -r requirements.txt`.".format(
                ", ".join(missing)
            )
        )


def initialize_runtime(raise_on_missing: bool = True) -> str | None:
    set_timing_env_from_config()
    missing = find_missing_runtime_dependencies()
    if not missing:
        return None
    message = (
        "Missing runtime dependencies: {}. Install with `python -m pip install -r requirements.txt`.".format(
            ", ".join(missing)
        )
    )
    if raise_on_missing:
        raise RuntimeError(message)
    return message


def initialize_runtime_cli_or_exit() -> None:
    try:
        initialize_runtime(raise_on_missing=True)
    except Exception as exc:
        sys.stderr.write(str(exc) + "\n")
        raise SystemExit(2)


def initialize_runtime_gui_or_exit() -> None:
    message = initialize_runtime(raise_on_missing=False)
    if not message:
        return
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Fitter", message)
        root.destroy()
    except Exception:
        sys.stderr.write(message + "\n")
    raise SystemExit(2)
