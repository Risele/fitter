from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


_CACHE: Dict[str, Any] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return _repo_root()


def get_storage_root(create: bool = False) -> Path:
    if getattr(sys, "frozen", False):
        root = _exe_dir() / "fitter_data"
    else:
        root = _repo_root() / ".fitter_data"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "default_config.json"


def get_user_config_path(create_parent: bool = False) -> Path:
    root = get_storage_root(create=create_parent)
    cfg_name = str(get_config().get("storage", {}).get("config_file", "config.json"))
    return root / cfg_name


def get_history_path(create_parent: bool = False) -> Path:
    root = get_storage_root(create=create_parent)
    name = str(get_config().get("storage", {}).get("history_file", "history.json"))
    return root / name


def get_default_output_dir(create: bool = False) -> Path:
    root = get_storage_root(create=create)
    subdir = str(get_config().get("storage", {}).get("output_subdir", "outputs"))
    out_dir = root / subdir
    if create:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[index]
        else:
            out[k] = v
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def get_config(force_reload: bool = False) -> Dict[str, Any]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE

    defaults = _load_json(_default_config_path())
    user_cfg_path = get_storage_root(create=False) / str(defaults.get("storage", {}).get("config_file", "config.json"))
    user_cfg = _load_json(user_cfg_path) if user_cfg_path.exists() else {}
    _CACHE = _deep_merge(defaults, user_cfg)
    return _CACHE


def set_timing_env_from_config() -> None:
    enabled = bool(get_config().get("logging", {}).get("enabled", False))
    os.environ["FITTER_TIMING_LOG"] = "1" if enabled else "0"


def codegen_max_line_length() -> int:
    try:
        value = int(get_config().get("codegen", {}).get("max_line_length", 80))
    except Exception:
        value = 80
    return max(40, value)


def default_array_placement() -> str:
    value = str(get_config().get("codegen", {}).get("array_placement", "file_static"))
    return value if value in ("file_static", "function_static") else "file_static"


def code_preview_syntax_highlight_enabled() -> bool:
    return bool(get_config().get("ui", {}).get("code_preview", {}).get("syntax_highlight", True))
