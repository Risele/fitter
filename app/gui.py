from __future__ import annotations

import json
import os
import re
import multiprocessing
import tkinter as tk
from dataclasses import asdict
from datetime import datetime
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Dict, List, Optional, Tuple

from app.bootstrap import initialize_runtime_gui_or_exit

initialize_runtime_gui_or_exit()

from app.pipeline import (
    CodegenConfig,
    FitComputationBundle,
    FitConfig,
    FitExecutionConfig,
    FitResultBundle,
    RenderedOutputBundle,
    build_source,
    estimate_for_config,
    estimate_from_dict,
    estimate_to_dict,
    fit_result_from_dict,
    fit_result_to_dict,
    render_outputs,
    resolve_out_path,
    rows_from_list,
    rows_to_list,
    run_fit,
    run_fit_compute,
    split_fit_config,
)
from app.runtime_config import (
    code_preview_syntax_highlight_enabled,
    default_array_placement,
    get_default_output_dir,
    get_history_path,
)
from core.types import FitError
from app.plotting_tk import clear_children, embed_figure


HISTORY_FILENAME = "history.json"
HISTORY_MAX = 200


def _history_path() -> str:
    return str(get_history_path(create_parent=True))


def _load_history() -> List[Dict[str, object]]:
    path = _history_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _save_history(entries: List[Dict[str, object]]) -> None:
    path = _history_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries[:HISTORY_MAX], f, indent=2)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_translations() -> Dict[str, Dict[str, str]]:
    path = os.path.join(os.path.dirname(__file__), "i18n.json")
    if not os.path.exists(path):
        return {"en": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {"en": {}}
    return {"en": {}}


TRANSLATIONS = _load_translations()


def _t(lang: str, key: str) -> str:
    return TRANSLATIONS.get(lang, {}).get(key, TRANSLATIONS.get("en", {}).get(key, key))


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def update_text(self, text: str) -> None:
        self.text = text

    def _show(self, _event=None) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack(ipadx=4, ipady=2)

    def _hide(self, _event=None) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def _auto_relax_fit(
    config: FitConfig,
    relax_settings: Dict[str, object],
    cancel_event,
    source=None,
    estimate=None,
    progress_callback=None,
) -> Dict[str, object]:
    if not relax_settings.get("enabled", False):
        bundle = run_fit(
            config,
            cancel_check=cancel_event.is_set if cancel_event else None,
            source=source,
            estimate=estimate,
            progress_callback=progress_callback,
        )
        return {
            "bundle": bundle,
            "relax_info": None,
            "final_abs": config.abs_err,
            "final_sym": config.sym_err,
        }

    step_pct = float(relax_settings.get("step_pct", 25.0))
    max_retries = int(relax_settings.get("max_retries", 8))
    cap_mult = float(relax_settings.get("cap_mult", 5.0))
    step_factor = 1.0 + step_pct / 100.0

    init_abs = float(config.abs_err)
    init_sym = float(config.sym_err)
    cur_abs = float(config.abs_err)
    cur_sym = float(config.sym_err)

    retries = 0
    relaxed_abs = False
    relaxed_sym = False

    while True:
        trial = FitConfig(**config.to_dict())
        trial.abs_err = cur_abs
        trial.sym_err = cur_sym
        try:
            is_initial_trial = (retries == 0 and cur_abs == init_abs and cur_sym == init_sym)
            bundle = run_fit(
                trial,
                cancel_check=cancel_event.is_set if cancel_event else None,
                source=source,
                estimate=(estimate if is_initial_trial else None),
                progress_callback=progress_callback,
            )
            relax_info = None
            if retries > 0:
                relax_info = {
                    "retries": retries,
                    "step_pct": step_pct,
                    "cap_mult": cap_mult,
                    "init_abs": init_abs,
                    "init_sym": init_sym,
                    "final_abs": cur_abs,
                    "final_sym": cur_sym,
                    "relaxed_abs": relaxed_abs,
                    "relaxed_sym": relaxed_sym,
                }
            return {
                "bundle": bundle,
                "relax_info": relax_info,
                "final_abs": cur_abs,
                "final_sym": cur_sym,
            }
        except FitError as exc:
            if exc.code != "constraints_unsatisfied":
                raise
            max_abs = float(exc.data.get("max_abs", 0.0))
            mean_signed = float(exc.data.get("mean_signed", 0.0))
            need_abs = max_abs > cur_abs
            need_sym = abs(mean_signed) > cur_sym

            if not need_abs and not need_sym:
                raise

            retries += 1
            if retries > max_retries:
                raise

            if need_abs:
                cur_abs = cur_abs * step_factor
                relaxed_abs = True
            if need_sym:
                if cur_sym == 0.0:
                    cur_sym = abs(mean_signed) * step_factor
                else:
                    cur_sym = cur_sym * step_factor
                relaxed_sym = True

            if cur_abs > init_abs * cap_mult or cur_sym > init_sym * cap_mult:
                raise


def _fit_worker(config_dict: Dict[str, object], relax_settings: Dict[str, object], cancel_event, queue) -> None:
    try:
        config = FitConfig(**config_dict)
        source = build_source(config)
        estimate = estimate_for_config(config, source=source)
        queue.put(
            {
                "status": "progress_estimate",
                "estimate": {
                    "N_est": int(estimate.n_segments),
                    "h_max": float(estimate.h_max),
                    "M": float(estimate.m_used),
                },
            }
        )
        def _progress_cb(payload: Dict[str, object]) -> None:
            queue.put({"status": "progress_fit", "progress": dict(payload)})

        result = _auto_relax_fit(
            config,
            relax_settings,
            cancel_event,
            source=source,
            estimate=estimate,
            progress_callback=_progress_cb,
        )
        bundle = result["bundle"]
        queue.put(
            {
                "status": "ok",
                "fit": bundle.fit,
                "rows": bundle.rows,
                "summary": bundle.summary,
                "c_code": bundle.c_code,
                "report_text": bundle.report_text,
                "relax_info": result["relax_info"],
                "final_abs": result["final_abs"],
                "final_sym": result["final_sym"],
                "estimate_full": estimate_to_dict(bundle.estimate if bundle.estimate is not None else estimate),
                "estimate": {
                    "N_est": int(bundle.estimate.n_segments) if bundle.estimate is not None else int(estimate.n_segments),
                    "h_max": float(bundle.estimate.h_max) if bundle.estimate is not None else float(estimate.h_max),
                    "M": float(bundle.estimate.m_used) if bundle.estimate is not None else float(estimate.m_used),
                },
            }
        )
    except FitError as exc:
        queue.put(
            {
                "status": "fit_error",
                "code": exc.code,
                "data": exc.data,
                "message": str(exc),
            }
        )
    except Exception as exc:
        queue.put({"status": "error", "message": str(exc), "type": type(exc).__name__})


class FitterGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.history = _load_history()
        self.last_bundle = None
        self.last_compute_bundle: Optional[FitComputationBundle] = None
        self.last_rendered_bundle: Optional[RenderedOutputBundle] = None
        self._active_canvas = None
        self._running = False
        self._cancel_event = None
        self._process = None
        self._queue = None
        self._run_estimate = None
        self._run_fit_progress: Optional[Dict[str, object]] = None
        self._codegen_autoupdate_after = None
        self._suspend_codegen_autoupdate = False
        self._mp_ctx = multiprocessing.get_context("spawn")

        self._i18n_widgets: List[Tuple[tk.Widget, str]] = []
        self._tooltips: List[Tuple[ToolTip, str]] = []

        self._build_vars()
        self._build_ui()
        self._refresh_history_list()
        self._sync_source_fields()
        self._sync_analytic_mode_fields()
        self._sync_mode_fields()
        self._apply_language()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_vars(self) -> None:
        self.language_var = tk.StringVar(value="en")
        self.source_type_var = tk.StringVar(value="table")
        self.analytic_mode_var = tk.StringVar(value="expr")

        self.table_path_var = tk.StringVar(value="")
        self.expr_var = tk.StringVar(value="")
        self.func_file_var = tk.StringVar(value="")
        self.xmin_var = tk.StringVar(value="-2")
        self.xmax_var = tk.StringVar(value="5")

        self.interp_var = tk.StringVar(value="linear")
        self.mode_var = tk.StringVar(value="float")
        self.abs_err_var = tk.StringVar(value="1e-2")
        self.sym_err_var = tk.StringVar(value="1e-2")
        self.auto_relax_var = tk.BooleanVar(value=False)
        self.relax_step_var = tk.StringVar(value="25")
        self.relax_retries_var = tk.StringVar(value="8")
        self.relax_cap_var = tk.StringVar(value="5")
        self.n_max_var = tk.StringVar(value="16384")
        self.with_slopes_var = tk.BooleanVar(value=False)
        self.test_count_var = tk.StringVar(value="64")
        self.xmin_shift_var = tk.BooleanVar(value=False)
        self.q_frac_var = tk.StringVar(value="20")
        self.x_shift_var = tk.StringVar(value="12")
        self.y_scale_var = tk.StringVar(value="100000")

        self.func_name_var = tk.StringVar(value="f_approx")
        self.array_name_var = tk.StringVar(value="tbl_f")
        self.array_placement_var = tk.StringVar(value=default_array_placement())
        self.array_placement_display_var = tk.StringVar(value="")
        self.out_path_var = tk.StringVar(value="")

        self.status_var = tk.StringVar(value="")
        self.run_info_var = tk.StringVar(value="")

        self.mode_var.trace_add("write", lambda *_: self._sync_mode_fields())
        self.source_type_var.trace_add("write", lambda *_: self._sync_source_fields())
        self.analytic_mode_var.trace_add("write", lambda *_: self._sync_analytic_mode_fields())
        self.language_var.trace_add("write", lambda *_: self._apply_language())
        self.auto_relax_var.trace_add("write", lambda *_: self._sync_relax_fields())
        self.array_placement_var.trace_add("write", lambda *_: self._sync_array_placement_combo())
        for var in (
            self.mode_var,
            self.with_slopes_var,
            self.test_count_var,
            self.q_frac_var,
            self.x_shift_var,
            self.y_scale_var,
            self.func_name_var,
            self.array_name_var,
            self.array_placement_var,
        ):
            var.trace_add("write", lambda *_: self._on_codegen_setting_changed())

    def _build_ui(self) -> None:
        self.root.geometry("1100x820")

        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=6, pady=4)

        self.run_btn = ttk.Button(toolbar, command=self.on_run)
        self.run_btn.pack(side="left")
        self._register_i18n(self.run_btn, "run")
        self.save_btn = ttk.Button(toolbar, command=self.on_save_report)
        self.save_btn.pack(side="left", padx=(6, 0))
        self._register_i18n(self.save_btn, "save_report")
        self.cancel_btn = ttk.Button(toolbar, command=self.on_cancel)
        self.cancel_btn.pack(side="left", padx=(6, 0))
        self._register_i18n(self.cancel_btn, "cancel")
        self.regen_btn = ttk.Button(toolbar, command=self.on_regenerate_code)
        self.regen_btn.pack(side="left", padx=(6, 0))
        self._register_i18n(self.regen_btn, "regen_code")

        self.progress = ttk.Progressbar(toolbar, mode="indeterminate", length=160)
        self._progress_visible = False
        self.run_info_label = ttk.Label(toolbar, textvariable=self.run_info_var, anchor="w")
        self.run_info_label.pack(side="left", padx=(8, 0))

        lang_label = ttk.Label(toolbar)
        lang_label.pack(side="right", padx=(6, 0))
        self._register_i18n(lang_label, "language")

        self.lang_combo = ttk.Combobox(
            toolbar,
            values=[],
            state="readonly",
            width=10,
        )
        self.lang_combo.pack(side="right")
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language_selected)
        self.lang_combo.set("")

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=4)

        self.main_tab = ttk.Frame(nb)
        self.plot_tab = ttk.Frame(nb)
        self.history_tab = ttk.Frame(nb)

        nb.add(self.main_tab, text="Main")
        nb.add(self.plot_tab, text="Plot")
        nb.add(self.history_tab, text="History")

        self._build_main_tab()
        self._build_plot_tab()
        self._build_history_tab()

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=6, pady=(0, 4))

        self._sync_relax_fields()
        self._sync_run_state()

    def _build_main_tab(self) -> None:
        frame = self.main_tab
        content, canvas = self._create_scrollable(frame)

        src_box = ttk.LabelFrame(content)
        src_box.pack(fill="x", padx=6, pady=4)
        self._register_i18n(src_box, "source_type")

        self.table_radio = ttk.Radiobutton(src_box, variable=self.source_type_var, value="table")
        self.table_radio.pack(side="left", padx=6, pady=6)
        self._register_i18n(self.table_radio, "source_table")

        self.analytic_radio = ttk.Radiobutton(src_box, variable=self.source_type_var, value="analytic")
        self.analytic_radio.pack(side="left", padx=6, pady=6)
        self._register_i18n(self.analytic_radio, "source_analytic")

        table_box = ttk.LabelFrame(content)
        table_box.pack(fill="x", padx=6, pady=4)
        self._register_i18n(table_box, "table_source")

        table_label = ttk.Label(table_box)
        table_label.grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self._register_i18n(table_label, "table_file")

        self.table_entry = ttk.Entry(table_box, textvariable=self.table_path_var, width=70)
        self.table_entry.grid(row=0, column=1, sticky="we", padx=6, pady=6)

        self.table_browse_btn = ttk.Button(table_box, command=self.on_browse_table)
        self.table_browse_btn.grid(row=0, column=2, sticky="w", padx=6, pady=6)
        self._register_i18n(self.table_browse_btn, "browse")
        interp_table_label = ttk.Label(table_box)
        interp_table_label.grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(interp_table_label, "interp")
        self.interp_combo = ttk.Combobox(table_box, textvariable=self.interp_var, values=["linear", "spline"], state="readonly")
        self.interp_combo.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        table_box.columnconfigure(1, weight=1)

        analytic_box = ttk.LabelFrame(content)
        analytic_box.pack(fill="x", padx=6, pady=4)
        self._register_i18n(analytic_box, "analytic_source")

        self.expr_radio = ttk.Radiobutton(analytic_box, variable=self.analytic_mode_var, value="expr")
        self.expr_radio.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(self.expr_radio, "analytic_mode_expr")

        self.expr_entry = ttk.Entry(analytic_box, textvariable=self.expr_var, width=50)
        self.expr_entry.grid(row=0, column=1, sticky="we", padx=6, pady=4)

        self.file_radio = ttk.Radiobutton(analytic_box, variable=self.analytic_mode_var, value="file")
        self.file_radio.grid(row=0, column=2, sticky="w", padx=6, pady=4)
        self._register_i18n(self.file_radio, "analytic_mode_file")

        self.file_entry = ttk.Entry(analytic_box, textvariable=self.func_file_var, width=40)
        self.file_entry.grid(row=0, column=3, sticky="we", padx=6, pady=4)
        self.file_browse_btn = ttk.Button(analytic_box, command=self.on_browse_expr)
        self.file_browse_btn.grid(row=0, column=4, sticky="w", padx=6, pady=4)
        self._register_i18n(self.file_browse_btn, "browse")

        domain_row = ttk.Frame(analytic_box)
        domain_row.grid(row=1, column=0, columnspan=5, sticky="w", padx=6, pady=4)

        xmin_label = ttk.Label(domain_row)
        xmin_label.pack(side="left")
        self._register_i18n(xmin_label, "xmin")
        self.xmin_entry = ttk.Entry(domain_row, textvariable=self.xmin_var, width=12)
        self.xmin_entry.pack(side="left", padx=(6, 12))

        xmax_label = ttk.Label(domain_row)
        xmax_label.pack(side="left")
        self._register_i18n(xmax_label, "xmax")
        self.xmax_entry = ttk.Entry(domain_row, textvariable=self.xmax_var, width=12)
        self.xmax_entry.pack(side="left", padx=(6, 0))

        analytic_box.columnconfigure(1, weight=1)
        analytic_box.columnconfigure(3, weight=1)

        self.table_box = table_box
        self.analytic_box = analytic_box

        params_box = ttk.LabelFrame(content)
        params_box.pack(fill="x", padx=6, pady=4)
        self._register_i18n(params_box, "parameters")

        abs_label = ttk.Label(params_box)
        abs_label.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(abs_label, "abs_err")
        self.abs_entry = ttk.Entry(params_box, textvariable=self.abs_err_var, width=12)
        self.abs_entry.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        sym_label = ttk.Label(params_box)
        sym_label.grid(row=0, column=2, sticky="w", padx=6, pady=4)
        self._register_i18n(sym_label, "sym_err")
        self.sym_entry = ttk.Entry(params_box, textvariable=self.sym_err_var, width=12)
        self.sym_entry.grid(row=0, column=3, sticky="w", padx=6, pady=4)

        self.xmin_shift_check = ttk.Checkbutton(params_box, variable=self.xmin_shift_var)
        self.xmin_shift_check.grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(self.xmin_shift_check, "xmin_shift")

        self.auto_relax_check = ttk.Checkbutton(params_box, variable=self.auto_relax_var)
        self.auto_relax_check.grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(self.auto_relax_check, "auto_relax_enable")

        step_label = ttk.Label(params_box)
        step_label.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        self._register_i18n(step_label, "auto_relax_step")
        self.relax_step_entry = ttk.Entry(params_box, textvariable=self.relax_step_var, width=8)
        self.relax_step_entry.grid(row=2, column=2, sticky="w", padx=6, pady=4)

        retry_label = ttk.Label(params_box)
        retry_label.grid(row=2, column=3, sticky="w", padx=6, pady=4)
        self._register_i18n(retry_label, "auto_relax_retries")
        self.relax_retries_entry = ttk.Entry(params_box, textvariable=self.relax_retries_var, width=8)
        self.relax_retries_entry.grid(row=2, column=4, sticky="w", padx=6, pady=4)

        cap_label = ttk.Label(params_box)
        cap_label.grid(row=2, column=5, sticky="w", padx=6, pady=4)
        self._register_i18n(cap_label, "auto_relax_cap")
        self.relax_cap_entry = ttk.Entry(params_box, textvariable=self.relax_cap_var, width=8)
        self.relax_cap_entry.grid(row=2, column=6, sticky="w", padx=6, pady=4)

        fit_tabs = ttk.Notebook(content)
        fit_tabs.pack(fill="x", padx=6, pady=4)
        fit_basic_tab = ttk.Frame(fit_tabs)
        fit_adv_tab = ttk.Frame(fit_tabs)
        fit_alg_tab = ttk.Frame(fit_tabs)
        fit_tabs.add(fit_basic_tab, text="")
        fit_tabs.add(fit_adv_tab, text="")
        fit_tabs.add(fit_alg_tab, text="")
        self.fit_settings_tabs = fit_tabs
        self.fit_basic_tab = fit_basic_tab
        self.fit_adv_tab = fit_adv_tab
        self.fit_alg_tab = fit_alg_tab
        lut_box = ttk.LabelFrame(fit_basic_tab)
        lut_box.pack(fill="x", padx=6, pady=4)
        self._register_i18n(lut_box, "uniform_lut_method")
        nmax_label = ttk.Label(lut_box)
        nmax_label.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(nmax_label, "n_max")
        self.nmax_entry = ttk.Entry(lut_box, textvariable=self.n_max_var, width=12)
        self.nmax_entry.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        self.fit_adv_placeholder = ttk.Label(fit_adv_tab)
        self.fit_adv_placeholder.pack(anchor="w", padx=8, pady=8)
        self._register_i18n(self.fit_adv_placeholder, "placeholder_new_release")
        self.fit_alg_placeholder = ttk.Label(fit_alg_tab)
        self.fit_alg_placeholder.pack(anchor="w", padx=8, pady=8)
        self._register_i18n(self.fit_alg_placeholder, "placeholder_new_release")

        codegen_box = ttk.LabelFrame(content)
        codegen_box.pack(fill="x", padx=6, pady=4)
        self._register_i18n(codegen_box, "codegen_opts")

        fn_label = ttk.Label(codegen_box)
        fn_label.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(fn_label, "func_name")
        self.func_name_entry = ttk.Entry(codegen_box, textvariable=self.func_name_var, width=24)
        self.func_name_entry.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        an_label = ttk.Label(codegen_box)
        an_label.grid(row=0, column=2, sticky="w", padx=6, pady=4)
        self._register_i18n(an_label, "array_name")
        self.array_name_entry = ttk.Entry(codegen_box, textvariable=self.array_name_var, width=24)
        self.array_name_entry.grid(row=0, column=3, sticky="w", padx=6, pady=4)

        mode_label = ttk.Label(codegen_box)
        mode_label.grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(mode_label, "mode")
        self.mode_combo = ttk.Combobox(
            codegen_box,
            textvariable=self.mode_var,
            values=["float", "double", "int16", "int32"],
            state="readonly",
            width=12,
        )
        self.mode_combo.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        self.slopes_check = ttk.Checkbutton(codegen_box, variable=self.with_slopes_var)
        self.slopes_check.grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(self.slopes_check, "with_slopes")

        test_label = ttk.Label(codegen_box)
        test_label.grid(row=2, column=1, sticky="e", padx=(6, 2), pady=4)
        self._register_i18n(test_label, "test_count")
        self.test_entry = ttk.Entry(codegen_box, textvariable=self.test_count_var, width=10)
        self.test_entry.grid(row=2, column=2, sticky="w", padx=(2, 6), pady=4)

        placement_label = ttk.Label(codegen_box)
        placement_label.grid(row=1, column=2, sticky="w", padx=6, pady=4)
        self._register_i18n(placement_label, "array_placement")
        self.array_placement_label = placement_label
        self.array_placement_combo = ttk.Combobox(
            codegen_box,
            textvariable=self.array_placement_display_var,
            values=[],
            state="readonly",
            width=18,
        )
        self.array_placement_combo.grid(row=1, column=3, sticky="w", padx=6, pady=4)
        self.array_placement_combo.bind("<<ComboboxSelected>>", self._on_array_placement_selected)

        fp_box = ttk.LabelFrame(codegen_box)
        fp_box.grid(row=3, column=0, columnspan=4, sticky="we", padx=6, pady=4)
        self._register_i18n(fp_box, "fixed_point")

        q_label = ttk.Label(fp_box)
        q_label.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(q_label, "q_frac")
        self.q_frac_entry = ttk.Entry(fp_box, textvariable=self.q_frac_var, width=12)
        self.q_frac_entry.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        xshift_label = ttk.Label(fp_box)
        xshift_label.grid(row=0, column=2, sticky="w", padx=6, pady=4)
        self._register_i18n(xshift_label, "x_shift")
        self.x_shift_entry = ttk.Entry(fp_box, textvariable=self.x_shift_var, width=12)
        self.x_shift_entry.grid(row=0, column=3, sticky="w", padx=6, pady=4)

        yscale_label = ttk.Label(fp_box)
        yscale_label.grid(row=0, column=4, sticky="w", padx=6, pady=4)
        self._register_i18n(yscale_label, "y_scale")
        self.y_scale_entry = ttk.Entry(fp_box, textvariable=self.y_scale_var, width=12)
        self.y_scale_entry.grid(row=0, column=5, sticky="w", padx=6, pady=4)

        out_label = ttk.Label(codegen_box)
        out_label.grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self._register_i18n(out_label, "output_path")
        self.out_entry = ttk.Entry(codegen_box, textvariable=self.out_path_var, width=60)
        self.out_entry.grid(row=4, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        self.out_browse_btn = ttk.Button(codegen_box, command=self.on_browse_out)
        self.out_browse_btn.grid(row=4, column=3, sticky="w", padx=6, pady=4)
        self._register_i18n(self.out_browse_btn, "browse")
        codegen_box.columnconfigure(1, weight=1)

        result_box = ttk.LabelFrame(content)
        result_box.pack(fill="both", expand=True, padx=6, pady=4)
        self._register_i18n(result_box, "results")

        summary_label = ttk.Label(result_box)
        summary_label.pack(anchor="w", padx=6, pady=(4, 0))
        self._register_i18n(summary_label, "summary")
        summary_wrap = ttk.Frame(result_box)
        summary_wrap.pack(fill="x", padx=6, pady=4)
        self.summary_text = tk.Text(summary_wrap, height=7, wrap="none")
        self.summary_text.pack(side="left", fill="x", expand=True)
        summary_y = ttk.Scrollbar(summary_wrap, orient="vertical", command=self.summary_text.yview)
        summary_y.pack(side="right", fill="y")
        self.summary_text.configure(yscrollcommand=summary_y.set)

        code_label = ttk.Label(result_box)
        code_label.pack(anchor="w", padx=6, pady=(4, 0))
        self._register_i18n(code_label, "c_code")
        code_wrap = ttk.Frame(result_box)
        code_wrap.pack(fill="both", expand=True, padx=6, pady=4)
        self.code_text = tk.Text(code_wrap, height=12, wrap="none")
        self.code_text.pack(side="left", fill="both", expand=True)
        self.code_text.configure(font=tkfont.nametofont("TkFixedFont"))
        code_y = ttk.Scrollbar(code_wrap, orient="vertical", command=self.code_text.yview)
        code_y.pack(side="right", fill="y")
        code_x = ttk.Scrollbar(result_box, orient="horizontal", command=self.code_text.xview)
        code_x.pack(fill="x", padx=6, pady=(0, 4))
        self.code_text.configure(yscrollcommand=code_y.set, xscrollcommand=code_x.set)

        action_row = ttk.Frame(result_box)
        action_row.pack(fill="x", padx=6, pady=(0, 4))
        self.copy_code_btn = ttk.Button(action_row, command=self.on_copy_code)
        self.copy_code_btn.pack(side="left")
        self._register_i18n(self.copy_code_btn, "copy_code")
        self.copy_summary_btn = ttk.Button(action_row, command=self.on_copy_summary)
        self.copy_summary_btn.pack(side="left", padx=(6, 0))
        self._register_i18n(self.copy_summary_btn, "copy_summary")
        self.regen_code_btn = ttk.Button(action_row, command=self.on_regenerate_code)
        self.regen_code_btn.pack(side="left", padx=(6, 0))
        self._register_i18n(self.regen_code_btn, "regen_code")

        self._add_tooltips()
        self._bind_mousewheel(canvas)

    def _create_scrollable(self, parent: tk.Widget) -> Tuple[ttk.Frame, tk.Canvas]:
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _on_frame_configure(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        return content, canvas

    def _bind_mousewheel(self, canvas: tk.Canvas) -> None:
        def _enter(_event=None) -> None:
            self._active_canvas = canvas
            canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        def _leave(_event=None) -> None:
            if self._active_canvas is canvas:
                canvas.unbind_all("<MouseWheel>")
                self._active_canvas = None

        canvas.bind("<Enter>", _enter)
        canvas.bind("<Leave>", _leave)

    def _on_mousewheel(self, event) -> None:
        if self._active_canvas is None:
            return
        delta = int(-1 * (event.delta / 120))
        self._active_canvas.yview_scroll(delta, "units")

    def _build_plot_tab(self) -> None:
        frame = self.plot_tab
        self.plot_info = ttk.Label(frame)
        self.plot_info.pack(anchor="w", padx=10, pady=(10, 0))
        self._register_i18n(self.plot_info, "plot_info")

        self.plot_container = ttk.Frame(frame)
        self.plot_container.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_history_tab(self) -> None:
        frame = self.history_tab

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True, padx=10, pady=8)

        self.history_list = tk.Listbox(list_frame, height=10)
        self.history_list.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.history_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.history_list.configure(yscrollcommand=scrollbar.set)

        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=10, pady=6)
        self.load_btn = ttk.Button(btns, command=self.on_history_load)
        self.load_btn.pack(side="left")
        self._register_i18n(self.load_btn, "load_config")
        self.rerun_btn = ttk.Button(btns, command=self.on_history_rerun)
        self.rerun_btn.pack(side="left", padx=(6, 0))
        self._register_i18n(self.rerun_btn, "rerun")
        self.export_btn = ttk.Button(btns, command=self.on_history_export)
        self.export_btn.pack(side="left", padx=(6, 0))
        self._register_i18n(self.export_btn, "export_report")
        self.delete_btn = ttk.Button(btns, command=self.on_history_delete)
        self.delete_btn.pack(side="left", padx=(6, 0))
        self._register_i18n(self.delete_btn, "delete")

        batch_box = ttk.LabelFrame(frame)
        batch_box.pack(fill="x", padx=10, pady=8)
        self._register_i18n(batch_box, "batch")

        self.batch_csv_btn = ttk.Button(batch_box, command=self.on_batch_csv)
        self.batch_csv_btn.grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self._register_i18n(self.batch_csv_btn, "batch_csv")

        self.batch_label = ttk.Label(batch_box)
        self.batch_label.grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self._register_i18n(self.batch_label, "batch_analytic_label")
        self.batch_expr_text = tk.Text(batch_box, height=6)
        self.batch_expr_text.grid(row=2, column=0, columnspan=2, sticky="we", padx=6, pady=6)
        self.batch_run_btn = ttk.Button(batch_box, command=self.on_batch_analytic)
        self.batch_run_btn.grid(row=3, column=0, sticky="w", padx=6, pady=6)
        self._register_i18n(self.batch_run_btn, "run_analytic_batch")
        batch_box.columnconfigure(1, weight=1)

        self._add_history_tooltips()

    def _register_i18n(self, widget: tk.Widget, key: str) -> None:
        self._i18n_widgets.append((widget, key))

    def _register_tooltip(self, widget: tk.Widget, key: str) -> None:
        tip = ToolTip(widget, "")
        self._tooltips.append((tip, key))

    def _add_tooltips(self) -> None:
        self._register_tooltip(self.run_btn, "run_tip")
        self._register_tooltip(self.save_btn, "save_tip")
        self._register_tooltip(self.cancel_btn, "cancel_tip")
        self._register_tooltip(self.regen_btn, "regen_code_tip")
        self._register_tooltip(self.lang_combo, "language_tip")
        self._register_tooltip(self.table_radio, "source_table_tip")
        self._register_tooltip(self.analytic_radio, "source_analytic_tip")
        self._register_tooltip(self.table_entry, "table_file_tip")
        self._register_tooltip(self.table_browse_btn, "browse_tip")
        self._register_tooltip(self.expr_radio, "analytic_mode_expr_tip")
        self._register_tooltip(self.file_radio, "analytic_mode_file_tip")
        self._register_tooltip(self.expr_entry, "analytic_expr_tip")
        self._register_tooltip(self.file_entry, "analytic_file_tip")
        self._register_tooltip(self.file_browse_btn, "browse_tip")
        self._register_tooltip(self.xmin_entry, "xmin_tip")
        self._register_tooltip(self.xmax_entry, "xmax_tip")
        self._register_tooltip(self.interp_combo, "interp_tip")
        self._register_tooltip(self.mode_combo, "mode_tip")
        self._register_tooltip(self.abs_entry, "abs_err_tip")
        self._register_tooltip(self.sym_entry, "sym_err_tip")
        self._register_tooltip(self.auto_relax_check, "auto_relax_enable_tip")
        self._register_tooltip(self.relax_step_entry, "auto_relax_step_tip")
        self._register_tooltip(self.relax_retries_entry, "auto_relax_retries_tip")
        self._register_tooltip(self.relax_cap_entry, "auto_relax_cap_tip")
        self._register_tooltip(self.nmax_entry, "n_max_tip")
        self._register_tooltip(self.test_entry, "test_count_tip")
        self._register_tooltip(self.slopes_check, "with_slopes_tip")
        self._register_tooltip(self.xmin_shift_check, "xmin_shift_tip")
        self._register_tooltip(self.q_frac_entry, "q_frac_tip")
        self._register_tooltip(self.x_shift_entry, "x_shift_tip")
        self._register_tooltip(self.y_scale_entry, "y_scale_tip")
        self._register_tooltip(self.func_name_entry, "func_name_tip")
        self._register_tooltip(self.array_name_entry, "array_name_tip")
        self._register_tooltip(self.array_placement_combo, "array_placement_tip")
        self._register_tooltip(self.out_entry, "output_path_tip")
        self._register_tooltip(self.out_browse_btn, "browse_tip")

    def _add_history_tooltips(self) -> None:
        self._register_tooltip(self.history_list, "history_list_tip")
        self._register_tooltip(self.load_btn, "load_config_tip")
        self._register_tooltip(self.rerun_btn, "rerun_tip")
        self._register_tooltip(self.export_btn, "export_report_tip")
        self._register_tooltip(self.delete_btn, "delete_tip")
        self._register_tooltip(self.batch_csv_btn, "batch_csv_tip")
        self._register_tooltip(self.batch_expr_text, "batch_analytic_tip")
        self._register_tooltip(self.batch_run_btn, "run_analytic_batch_tip")

    def _apply_language(self) -> None:
        lang = self.language_var.get()
        self.root.title(_t(lang, "window_title"))
        for widget, key in self._i18n_widgets:
            widget.configure(text=_t(lang, key))
        for tip, key in self._tooltips:
            tip.update_text(_t(lang, key))
        self.status_var.set(_t(lang, "status_ready"))
        self._update_tab_titles()
        self._update_language_combo(lang)
        self._update_run_info_label()

    def _on_close(self) -> None:
        if self._process is not None and self._process.is_alive():
            if self._cancel_event is not None:
                self._cancel_event.set()
            try:
                self._process.terminate()
            except Exception:
                pass
        self.root.quit()
        self.root.destroy()

    def _update_tab_titles(self) -> None:
        lang = self.language_var.get()
        nb = self.main_tab.master
        nb.tab(self.main_tab, text=_t(lang, "tab_main"))
        nb.tab(self.plot_tab, text=_t(lang, "tab_plot"))
        nb.tab(self.history_tab, text=_t(lang, "tab_history"))
        if hasattr(self, "fit_settings_tabs"):
            self.fit_settings_tabs.tab(self.fit_basic_tab, text=_t(lang, "fit_tab_basic"))
            self.fit_settings_tabs.tab(self.fit_adv_tab, text=_t(lang, "fit_tab_advanced"))
            self.fit_settings_tabs.tab(self.fit_alg_tab, text=_t(lang, "fit_tab_algorithms"))

    def _update_language_combo(self, lang: str) -> None:
        labels = [_t(lang, "language_en_name"), _t(lang, "language_ru_name")]
        self.lang_combo.configure(values=labels)
        if lang == "ru":
            self.lang_combo.set(_t(lang, "language_ru_name"))
        else:
            self.lang_combo.set(_t(lang, "language_en_name"))
        self._sync_array_placement_combo()

    def _on_language_selected(self, _event=None) -> None:
        val = self.lang_combo.get()
        lang = self.language_var.get()
        ru_label = _t(lang, "language_ru_name")
        self.language_var.set("ru" if val == ru_label else "en")

    def _array_placement_items(self) -> List[Tuple[str, str]]:
        lang = self.language_var.get()
        return [
            (_t(lang, "array_placement_file_static"), "file_static"),
            (_t(lang, "array_placement_function_static"), "function_static"),
        ]

    def _sync_array_placement_combo(self) -> None:
        if not hasattr(self, "array_placement_combo"):
            return
        items = self._array_placement_items()
        labels = [label for label, _ in items]
        self.array_placement_combo.configure(values=labels)
        current_value = self.array_placement_var.get() or default_array_placement()
        label_by_value = {value: label for label, value in items}
        self.array_placement_display_var.set(label_by_value.get(current_value, labels[0] if labels else ""))

    def _on_array_placement_selected(self, _event=None) -> None:
        items = self._array_placement_items()
        value_by_label = {label: value for label, value in items}
        selected = self.array_placement_display_var.get()
        self.array_placement_var.set(value_by_label.get(selected, default_array_placement()))
        self._on_codegen_setting_changed()

    def _on_codegen_setting_changed(self) -> None:
        if self._suspend_codegen_autoupdate:
            return
        if self._running or self.last_compute_bundle is None:
            return
        if self._codegen_autoupdate_after is not None:
            try:
                self.root.after_cancel(self._codegen_autoupdate_after)
            except Exception:
                pass
            self._codegen_autoupdate_after = None
        self._codegen_autoupdate_after = self.root.after(200, self._run_codegen_autoupdate)

    def _run_codegen_autoupdate(self) -> None:
        self._codegen_autoupdate_after = None
        self._regenerate_code_from_cache(show_error=False, set_status=False)

    def _sync_source_fields(self) -> None:
        is_table = self.source_type_var.get() == "table"
        state_table = "normal" if is_table else "disabled"
        state_analytic = "normal" if not is_table else "disabled"

        for child in self.table_box.winfo_children():
            try:
                child.configure(state=state_table)
            except Exception:
                pass
        for child in self.analytic_box.winfo_children():
            try:
                child.configure(state=state_analytic)
            except Exception:
                pass
        for w in (getattr(self, "xmin_entry", None), getattr(self, "xmax_entry", None)):
            if w is not None:
                try:
                    w.configure(state=state_analytic)
                except Exception:
                    pass
        self._sync_analytic_mode_fields()

    def _sync_analytic_mode_fields(self) -> None:
        if self.source_type_var.get() != "analytic":
            self.expr_entry.configure(state="disabled")
            self.file_entry.configure(state="disabled")
            self.file_browse_btn.configure(state="disabled")
            return
        use_expr = self.analytic_mode_var.get() == "expr"
        self.expr_entry.configure(state="normal" if use_expr else "disabled")
        self.file_entry.configure(state="disabled" if use_expr else "normal")
        self.file_browse_btn.configure(state="disabled" if use_expr else "normal")

    def _sync_mode_fields(self) -> None:
        is_int = self.mode_var.get() in ("int16", "int32")
        state = "normal" if is_int else "disabled"
        for entry in (self.q_frac_entry, self.x_shift_entry, self.y_scale_entry):
            entry.configure(state=state)

    def _sync_relax_fields(self) -> None:
        state = "normal" if self.auto_relax_var.get() else "disabled"
        for entry in (self.relax_step_entry, self.relax_retries_entry, self.relax_cap_entry):
            entry.configure(state=state)

    def _sync_run_state(self) -> None:
        state_run = "disabled" if self._running else "normal"
        state_cancel = "normal" if self._running else "disabled"
        self.run_btn.configure(state=state_run)
        self.cancel_btn.configure(state=state_cancel)
        self.save_btn.configure(state="normal" if not self._running else "disabled")
        regen_state = "normal" if (not self._running and self.last_compute_bundle is not None) else "disabled"
        self.regen_btn.configure(state=regen_state)
        self.regen_code_btn.configure(state=regen_state)
        if self._running:
            if not self._progress_visible:
                self.progress.pack(side="left", padx=(10, 0))
                self._progress_visible = True
            self.progress.start(10)
        else:
            self.progress.stop()
            self.progress.configure(value=0)
            if self._progress_visible:
                self.progress.pack_forget()
                self._progress_visible = False
        self._update_run_info_label()

    def _set_status_key(self, key: str) -> None:
        self.status_var.set(_t(self.language_var.get(), key))

    def _update_run_info_label(self) -> None:
        lang = self.language_var.get()
        if not self._running:
            self.run_info_var.set("")
            return
        est = self._run_estimate if isinstance(self._run_estimate, dict) else None
        prog = self._run_fit_progress if isinstance(self._run_fit_progress, dict) else None
        n_est = est.get("N_est") if est else None
        checked_n = prog.get("checked_n") if prog else None
        best_n = prog.get("best_n") if prog else None
        best_err = prog.get("best_max_abs_err") if prog else None
        parts = [_t(lang, "run_info_prefix")]
        if n_est is not None:
            parts.append(_t(lang, "run_info_n_est").format(n=int(n_est)))
        if checked_n is not None:
            parts.append(_t(lang, "run_info_n_check").format(n=int(checked_n)))
        if best_n is not None and best_err is not None:
            try:
                parts.append(_t(lang, "run_info_best").format(n=int(best_n), err=self._format_sci(best_err)))
            except Exception:
                parts.append("best N={}, err={}".format(best_n, best_err))
        self.run_info_var.set(" | ".join(parts))

    def _format_estimate_short(self, estimate: Optional[Dict[str, object]]) -> str:
        if not isinstance(estimate, dict):
            return ""
        try:
            n_est = int(estimate.get("N_est"))
            h_max = float(estimate.get("h_max"))
            m_used = float(estimate.get("M"))
        except Exception:
            return ""
        return "N_est={}, h<={:.6g}, M={:.6g}".format(n_est, h_max, m_used)

    def _set_status_with_estimate(self, key: str, estimate: Optional[Dict[str, object]]) -> None:
        lang = self.language_var.get()
        template = _t(lang, key)
        est_short = self._format_estimate_short(estimate)
        if not est_short:
            self.status_var.set(template)
            return
        try:
            self.status_var.set(template.format(estimate=est_short))
        except Exception:
            self.status_var.set("{} | {}".format(template, est_short))
        if key.startswith("status_running"):
            self._update_run_info_label()

    def _format_error(self, key: str, detail: str, **kwargs) -> str:
        lang = self.language_var.get()
        template = _t(lang, key)
        try:
            return template.format(detail=detail, **kwargs)
        except Exception:
            return "{}: {}".format(template, detail)

    def _format_exception(self, exc: Exception, fallback_key: str, **kwargs) -> str:
        lang = self.language_var.get()
        if isinstance(exc, FitError):
            if exc.code == "constraints_unsatisfied":
                template = _t(lang, "error_constraints_detail")
                try:
                    max_abs = exc.data.get("max_abs")
                    mean = exc.data.get("mean_signed")
                    return template.format(
                        n_max=exc.data.get("n_max", "?"),
                        max_abs=self._format_sci(max_abs),
                        mean=self._format_sci(mean),
                    )
                except Exception:
                    return template
        if isinstance(exc, RuntimeError):
            mapped = self._map_core_error(str(exc), lang)
            if mapped is not None:
                return mapped
        return self._format_error(fallback_key, str(exc), **kwargs)

    def _format_sci(self, value) -> str:
        try:
            return "{:.2e}".format(float(value))
        except Exception:
            return str(value)

    def _map_core_error(self, detail: str, lang: str) -> Optional[str]:
        pattern = (
            r"Cannot satisfy error constraints up to n_max=(?P<nmax>\\d+)\\."
            r" Last max_abs=(?P<max_abs>[-+eE0-9\\.]+), mean=(?P<mean>[-+eE0-9\\.]+)"
        )
        match = re.search(pattern, detail)
        if not match:
            return None
        data = match.groupdict()
        template = _t(lang, "error_constraints_detail")
        try:
            return template.format(
                n_max=data.get("nmax", "?"),
                max_abs=data.get("max_abs", "?"),
                mean=data.get("mean", "?"),
            )
        except Exception:
            return None

    def _get_float(self, value: str, name: str) -> float:
        try:
            return float(value)
        except Exception:
            raise ValueError(_t(self.language_var.get(), "invalid_value").format(name))

    def _get_int(self, value: str, name: str) -> int:
        try:
            return int(value)
        except Exception:
            raise ValueError(_t(self.language_var.get(), "invalid_value").format(name))

    def _build_config(self) -> FitConfig:
        if self.source_type_var.get() == "table":
            path = self.table_path_var.get().strip()
            if not path:
                raise ValueError(_t(self.language_var.get(), "table_path_required"))
            func = None
            func_file = None
        else:
            path = None
            if self.analytic_mode_var.get() == "file":
                func_file = self.func_file_var.get().strip() or None
                if not func_file:
                    raise ValueError(_t(self.language_var.get(), "expr_file_required"))
                func = None
            else:
                func_file = None
                expr = self.expr_var.get().strip()
                if not expr:
                    raise ValueError(_t(self.language_var.get(), "expr_required"))
                func = expr

        config = FitConfig(
            path=path,
            func=func,
            func_file=func_file,
            xmin=self._get_float(self.xmin_var.get(), "xmin") if path is None else None,
            xmax=self._get_float(self.xmax_var.get(), "xmax") if path is None else None,
            interp=self.interp_var.get(),
            mode=self.mode_var.get(),
            abs_err=self._get_float(self.abs_err_var.get(), "abs_err"),
            sym_err=self._get_float(self.sym_err_var.get(), "sym_err"),
            n_max=self._get_int(self.n_max_var.get(), "n_max"),
            with_slopes=self.with_slopes_var.get(),
            test_count=self._get_int(self.test_count_var.get(), "test_count"),
            xmin_shift=self.xmin_shift_var.get(),
            q_frac=self._get_int(self.q_frac_var.get(), "q_frac"),
            x_shift=self._get_int(self.x_shift_var.get(), "x_shift"),
            y_scale=self._get_int(self.y_scale_var.get(), "y_scale"),
            func_name=self.func_name_var.get().strip() or "f_approx",
            array_name=self.array_name_var.get().strip() or "tbl_f",
            array_placement=self.array_placement_var.get() or default_array_placement(),
        )
        return config

    def _build_codegen_config(self) -> CodegenConfig:
        return CodegenConfig(
            mode=self.mode_var.get(),
            with_slopes=self.with_slopes_var.get(),
            test_count=self._get_int(self.test_count_var.get(), "test_count"),
            q_frac=self._get_int(self.q_frac_var.get(), "q_frac"),
            x_shift=self._get_int(self.x_shift_var.get(), "x_shift"),
            y_scale=self._get_int(self.y_scale_var.get(), "y_scale"),
            func_name=self.func_name_var.get().strip() or "f_approx",
            array_name=self.array_name_var.get().strip() or "tbl_f",
            array_placement=self.array_placement_var.get() or default_array_placement(),
        )

    def _format_relax_report(
        self,
        retries: int,
        step_pct: float,
        cap_mult: float,
        init_abs: float,
        init_sym: float,
        final_abs: float,
        final_sym: float,
        relaxed_abs: bool,
        relaxed_sym: bool,
    ) -> str:
        lang = self.language_var.get()
        template = _t(lang, "auto_relax_report")
        changed = []
        if relaxed_abs:
            changed.append(_t(lang, "auto_relax_abs"))
        if relaxed_sym:
            changed.append(_t(lang, "auto_relax_sym"))
        return template.format(
            retries=retries,
            step_pct=step_pct,
            cap_mult=cap_mult,
            changed=", ".join(changed) if changed else _t(lang, "auto_relax_none"),
            init_abs=self._format_sci(init_abs),
            init_sym=self._format_sci(init_sym),
            final_abs=self._format_sci(final_abs),
            final_sym=self._format_sci(final_sym),
        )

    def _config_from_dict(self, cfg: Dict[str, object]) -> FitConfig:
        return FitConfig(
            path=cfg.get("path"),
            func=cfg.get("func"),
            func_file=cfg.get("func_file"),
            xmin=cfg.get("xmin"),
            xmax=cfg.get("xmax"),
            a=cfg.get("a", 1.0),
            b=cfg.get("b", 0.0),
            interp=cfg.get("interp", "linear"),
            mode=cfg.get("mode", "float"),
            abs_err=cfg.get("abs_err", 1e-2),
            sym_err=cfg.get("sym_err", 1e-2),
            n_max=cfg.get("n_max", 16384),
            with_slopes=cfg.get("with_slopes", False),
            test_count=cfg.get("test_count", 64),
            xmin_shift=cfg.get("xmin_shift", False),
            q_frac=cfg.get("q_frac", 20),
            x_shift=cfg.get("x_shift", 12),
            y_scale=cfg.get("y_scale", 100000),
            func_name=cfg.get("func_name", "f_approx"),
            array_name=cfg.get("array_name", "tbl_f"),
            array_placement=cfg.get("array_placement", default_array_placement()),
        )

    def _set_code_text(self, text: str) -> None:
        self.code_text.configure(state="normal")
        self.code_text.delete("1.0", tk.END)
        self.code_text.insert(tk.END, text)
        self._highlight_code()
        self.code_text.configure(state="disabled")

    def _highlight_code(self) -> None:
        if not code_preview_syntax_highlight_enabled():
            return
        try:
            self.code_text.tag_delete("comment", "pp", "kw")
        except Exception:
            pass
        self.code_text.tag_configure("comment", foreground="#2f6f44")
        self.code_text.tag_configure("pp", foreground="#5b2a86")
        self.code_text.tag_configure("kw", foreground="#0b5394")
        for idx, line in enumerate(self.code_text.get("1.0", tk.END).splitlines(), start=1):
            if line.strip().startswith("/*") or line.strip().startswith("*") or line.strip().startswith("*/"):
                self.code_text.tag_add("comment", f"{idx}.0", f"{idx}.end")
            if line.lstrip().startswith("#"):
                self.code_text.tag_add("pp", f"{idx}.0", f"{idx}.end")
            for token in ("static", "const", "inline", "return", "int32_t", "uint32_t", "float", "double"):
                start = 0
                while True:
                    pos = line.find(token, start)
                    if pos < 0:
                        break
                    self.code_text.tag_add("kw", f"{idx}.{pos}", f"{idx}.{pos + len(token)}")
                    start = pos + len(token)

    def _update_results(self, summary: str, c_code: str) -> None:
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)
        self._set_code_text(c_code)

    def _add_history_entry(
        self,
        config: FitConfig,
        summary: str,
        c_code: str,
        report_text: str,
        gui_settings: Optional[Dict[str, object]] = None,
        fit_data: Optional[Dict[str, object]] = None,
        rows: Optional[List[Dict[str, object]]] = None,
        estimate: Optional[Dict[str, object]] = None,
        render_config: Optional[Dict[str, object]] = None,
    ) -> None:
        entry = {
            "timestamp": _now_stamp(),
            "config": config.to_dict(),
            "summary": summary,
            "c_code": c_code,
            "report_text": report_text,
        }
        if gui_settings:
            entry["gui"] = gui_settings
        if fit_data is not None:
            entry["fit"] = fit_data
        if rows is not None:
            entry["rows"] = rows
        if estimate is not None:
            entry["estimate"] = estimate
        if render_config is not None:
            entry["render_config"] = render_config
        self.history.insert(0, entry)
        _save_history(self.history)
        self._refresh_history_list()

    def on_regenerate_code(self) -> None:
        if not self._regenerate_code_from_cache(show_error=True, set_status=True):
            self._set_status_key("status_run_failed")

    def _regenerate_code_from_cache(self, show_error: bool, set_status: bool) -> bool:
        if self.last_compute_bundle is None:
            return False
        try:
            rendered = render_outputs(self.last_compute_bundle, self._build_codegen_config())
        except Exception as exc:
            if show_error:
                msg = self._format_exception(exc, "error_run_detail")
                messagebox.showerror(_t(self.language_var.get(), "run_failed"), msg)
            return False
        self.last_rendered_bundle = rendered
        if self.last_bundle is not None:
            self.last_bundle.summary = rendered.summary
            self.last_bundle.c_code = rendered.c_code
            self.last_bundle.report_text = rendered.report_text
        self._update_results(rendered.summary, rendered.c_code)
        self._sync_run_state()
        if set_status:
            self._set_status_key("status_code_regenerated")
        return True

    def _refresh_history_list(self) -> None:
        self.history_list.delete(0, tk.END)
        for entry in self.history:
            label = "{} | {}".format(entry.get("timestamp", ""), self._history_label(entry))
            self.history_list.insert(tk.END, label)

    def _history_label(self, entry: Dict[str, object]) -> str:
        cfg = entry.get("config", {})
        if not isinstance(cfg, dict):
            return _t(self.language_var.get(), "history_item_run")
        if cfg.get("path"):
            return os.path.basename(str(cfg.get("path")))
        if cfg.get("func"):
            return str(cfg.get("func"))
        if cfg.get("func_file"):
            return os.path.basename(str(cfg.get("func_file")))
        return _t(self.language_var.get(), "history_item_run")

    def _selected_history_entry(self) -> Optional[Dict[str, object]]:
        sel = self.history_list.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx < 0 or idx >= len(self.history):
            return None
        return self.history[idx]

    def _update_plot(self) -> None:
        if self.last_bundle is None:
            return
        try:
            from app.plotting import build_fit_figure

            fig = build_fit_figure(self.last_bundle.source, self.last_bundle.fit, sample_points=1000)
        except Exception as exc:
            msg = self._format_error("error_plot_detail", str(exc))
            messagebox.showerror(_t(self.language_var.get(), "plot_failed"), msg)
            return

        clear_children(self.plot_container)
        embed_figure(self.plot_container, fig, with_toolbar=True)

    def on_browse_table(self) -> None:
        path = filedialog.askopenfilename(
            title=_t(self.language_var.get(), "select_csv"),
            filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All", "*.*")],
        )
        if path:
            self.table_path_var.set(path)

    def on_browse_expr(self) -> None:
        path = filedialog.askopenfilename(
            title=_t(self.language_var.get(), "select_expr"),
            filetypes=[("Text files", "*.txt"), ("All", "*.*")],
        )
        if path:
            self.func_file_var.set(path)

    def on_browse_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title=_t(self.language_var.get(), "save_report"),
            defaultextension=".txt",
            initialdir=str(get_default_output_dir(create=True)),
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
        )
        if path:
            self.out_path_var.set(path)

    def on_run(self) -> None:
        if self._running:
            return
        try:
            config = self._build_config()
        except Exception as exc:
            msg = self._format_exception(exc, "error_run_detail")
            messagebox.showerror(_t(self.language_var.get(), "run_failed"), msg)
            self._set_status_key("status_run_failed")
            return

        self._running = True
        self._cancel_event = self._mp_ctx.Event()
        self._run_estimate = None
        self._run_fit_progress = None
        self._set_status_key("status_running")
        self._sync_run_state()

        relax_settings = {
            "enabled": self.auto_relax_var.get(),
            "step_pct": self._get_float(self.relax_step_var.get(), "step_pct"),
            "max_retries": self._get_int(self.relax_retries_var.get(), "max_retries"),
            "cap_mult": self._get_float(self.relax_cap_var.get(), "cap_mult"),
        }

        self._queue = self._mp_ctx.Queue()
        self._process = self._mp_ctx.Process(
            target=_fit_worker,
            args=(config.to_dict(), relax_settings, self._cancel_event, self._queue),
        )
        self._process.daemon = True
        self._process.start()
        self._current_config = config
        self._poll_worker()

    def _poll_worker(self) -> None:
        if not self._running:
            return
        if self._process is None or self._queue is None:
            return

        terminal_msg = None
        while True:
            try:
                msg = self._queue.get_nowait()
            except Exception:
                msg = None
            if msg is None:
                break
            status = msg.get("status")
            if status == "progress_estimate":
                estimate = msg.get("estimate")
                if isinstance(estimate, dict):
                    self._run_estimate = estimate
                    self._update_run_info_label()
                    self._set_status_with_estimate("status_running_estimate", estimate)
                continue
            if status == "progress_fit":
                progress = msg.get("progress")
                if isinstance(progress, dict):
                    self._run_fit_progress = progress
                    self._update_run_info_label()
                continue
            terminal_msg = msg

        if terminal_msg is None:
            if not self._process.is_alive():
                terminal_msg = {"status": "error", "message": _t(self.language_var.get(), "worker_failed")}
            else:
                self.root.after(100, self._poll_worker)
                return

        self._process.join(timeout=0.1)
        self._process = None
        self._queue = None
        self._on_run_complete(self._current_config, terminal_msg)

    def _on_run_complete(self, config: FitConfig, msg) -> None:
        self._running = False
        self._sync_run_state()
        self._cancel_event = None

        status = msg.get("status")
        estimate = msg.get("estimate") if isinstance(msg.get("estimate"), dict) else self._run_estimate
        self._run_estimate = None
        self._run_fit_progress = None
        if status in ("error", "fit_error"):
            if status == "fit_error":
                code = msg.get("code")
                if code == "cancelled":
                    messagebox.showinfo(
                        _t(self.language_var.get(), "run_cancelled"),
                        _t(self.language_var.get(), "run_cancelled_detail"),
                    )
                    self._set_status_key("status_cancelled")
                    return
                exc = FitError(code=code or "error", data=msg.get("data", {}), message=msg.get("message", ""))
                err_msg = self._format_exception(exc, "error_run_detail")
            else:
                err_msg = self._format_error("error_run_detail", msg.get("message", ""))
            messagebox.showerror(_t(self.language_var.get(), "run_failed"), err_msg)
            if estimate is not None:
                self._set_status_with_estimate("status_run_failed_estimate", estimate)
            else:
                self._set_status_key("status_run_failed")
            return

        fit = msg.get("fit")
        rows = rows_from_list(msg.get("rows"))
        summary = msg.get("summary", "")
        c_code = msg.get("c_code", "")
        report_text = msg.get("report_text", "")
        relax_info = msg.get("relax_info")
        final_abs = msg.get("final_abs")
        final_sym = msg.get("final_sym")
        relax_used = relax_info is not None
        estimate_full = estimate_from_dict(msg.get("estimate_full") if isinstance(msg.get("estimate_full"), dict) else None)

        source = build_source(config)
        bundle = FitResultBundle(
            source=source,
            fit=fit,
            summary=summary,
            c_code=c_code,
            report_text=report_text,
            rows=rows,
            estimate=estimate_full,
        )
        self.last_bundle = bundle
        self.last_compute_bundle = FitComputationBundle(source=source, fit=fit, rows=rows, estimate=estimate_full)
        self.last_rendered_bundle = RenderedOutputBundle(summary=summary, c_code=c_code, report_text=report_text)
        if relax_info:
            relax_report = self._format_relax_report(
                retries=relax_info["retries"],
                step_pct=relax_info["step_pct"],
                cap_mult=relax_info["cap_mult"],
                init_abs=relax_info["init_abs"],
                init_sym=relax_info["init_sym"],
                final_abs=relax_info["final_abs"],
                final_sym=relax_info["final_sym"],
                relaxed_abs=relax_info["relaxed_abs"],
                relaxed_sym=relax_info["relaxed_sym"],
            )
            summary = relax_report + "\n\n" + summary
            report_text = relax_report + "\n\n" + report_text
            bundle.summary = summary
            bundle.report_text = report_text
            if self.last_rendered_bundle is not None:
                self.last_rendered_bundle.summary = summary
                self.last_rendered_bundle.report_text = report_text

        self._update_results(summary, c_code)
        self._sync_run_state()
        gui_settings = {
            "auto_relax": self.auto_relax_var.get(),
            "step_pct": self._get_float(self.relax_step_var.get(), "step_pct"),
            "max_retries": self._get_int(self.relax_retries_var.get(), "max_retries"),
            "cap_mult": self._get_float(self.relax_cap_var.get(), "cap_mult"),
            "final_abs_err": final_abs,
            "final_sym_err": final_sym,
            "relax_used": relax_used,
        }
        try:
            render_cfg_dict = asdict(self._build_codegen_config())
        except Exception:
            render_cfg_dict = None
        self._add_history_entry(
            config,
            summary,
            c_code,
            report_text,
            gui_settings=gui_settings,
            fit_data=fit_result_to_dict(fit) if fit is not None else None,
            rows=rows_to_list(rows),
            estimate=estimate_to_dict(estimate_full),
            render_config=render_cfg_dict,
        )
        if estimate is not None:
            self._set_status_with_estimate("status_run_complete_estimate", estimate)
        else:
            self._set_status_key("status_run_complete")

        out_path = self.out_path_var.get().strip()
        if out_path:
            try:
                out_path = resolve_out_path(out_path)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(report_text + "\n")
            except Exception as exc:
                msg = self._format_error("error_save_detail", str(exc))
                messagebox.showwarning(_t(self.language_var.get(), "save_failed"), msg)

        self._update_plot()

    def on_cancel(self) -> None:
        if not self._running or self._cancel_event is None:
            return
        self._cancel_event.set()
        self._set_status_key("status_cancelling")

    def on_save_report(self) -> None:
        if self.last_bundle is None:
            messagebox.showinfo(_t(self.language_var.get(), "save_report"), _t(self.language_var.get(), "run_first"))
            return
        path = self.out_path_var.get().strip()
        if not path:
            path = filedialog.asksaveasfilename(
                title=_t(self.language_var.get(), "save_report"),
                defaultextension=".txt",
                initialdir=str(get_default_output_dir(create=True)),
                filetypes=[("Text", "*.txt"), ("All", "*.*")],
            )
        if not path:
            return
        try:
            path = resolve_out_path(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.last_bundle.report_text + "\n")
            self._set_status_key("status_saved")
        except Exception as exc:
            msg = self._format_error("error_save_detail", str(exc))
            messagebox.showerror(_t(self.language_var.get(), "save_failed"), msg)

    def on_copy_code(self) -> None:
        text = self.code_text.get("1.0", tk.END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status_key("status_code_copied")

    def on_copy_summary(self) -> None:
        text = self.summary_text.get("1.0", tk.END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status_key("status_summary_copied")

    def on_history_load(self) -> None:
        entry = self._selected_history_entry()
        if not entry:
            return
        cfg = entry.get("config", {})
        if not isinstance(cfg, dict):
            return
        gui_cfg = entry.get("gui") if isinstance(entry, dict) else None
        self._apply_config(cfg, gui_cfg if isinstance(gui_cfg, dict) else None)
        summary = entry.get("summary")
        c_code = entry.get("c_code")
        if isinstance(summary, str) and isinstance(c_code, str):
            self._update_results(summary, c_code)
        fit_data = entry.get("fit")
        rows_data = entry.get("rows")
        est_data = entry.get("estimate")
        if isinstance(fit_data, dict):
            try:
                fit = fit_result_from_dict(fit_data)
                rows = rows_from_list(rows_data)
                est = estimate_from_dict(est_data if isinstance(est_data, dict) else None)
                source = build_source(self._config_from_dict(cfg))
                self.last_compute_bundle = FitComputationBundle(source=source, fit=fit, rows=rows, estimate=est)
                rendered = RenderedOutputBundle(
                    summary=str(summary or ""),
                    c_code=str(c_code or ""),
                    report_text=str(entry.get("report_text", "")),
                )
                self.last_rendered_bundle = rendered
                self.last_bundle = FitResultBundle(
                    source=source,
                    fit=fit,
                    summary=rendered.summary,
                    c_code=rendered.c_code,
                    report_text=rendered.report_text,
                    rows=rows,
                    estimate=est,
                )
                self._update_plot()
                self._sync_run_state()
                self._set_status_key("status_loaded")
                return
            except Exception as exc:
                self.last_compute_bundle = None
                self.last_rendered_bundle = None
                self.last_bundle = None
                self._sync_run_state()
                self.status_var.set(_t(self.language_var.get(), "status_loaded_plot_restore_failed").format(detail=str(exc)))
                return
        self.last_compute_bundle = None
        self.last_rendered_bundle = None
        self.last_bundle = None
        self._sync_run_state()
        self._set_status_key("status_loaded_legacy")

    def _apply_config(self, cfg: Dict[str, object], gui_cfg: Optional[Dict[str, object]] = None) -> None:
        prev_suspend = self._suspend_codegen_autoupdate
        self._suspend_codegen_autoupdate = True
        try:
            if cfg.get("path"):
                self.source_type_var.set("table")
                self.table_path_var.set(str(cfg.get("path")))
            else:
                self.source_type_var.set("analytic")
                if cfg.get("func_file"):
                    self.analytic_mode_var.set("file")
                    self.func_file_var.set(str(cfg.get("func_file") or ""))
                else:
                    self.analytic_mode_var.set("expr")
                    self.expr_var.set(str(cfg.get("func") or ""))
                self.xmin_var.set(str(cfg.get("xmin", "")))
                self.xmax_var.set(str(cfg.get("xmax", "")))

            self.interp_var.set(str(cfg.get("interp", "linear")))
            self.mode_var.set(str(cfg.get("mode", "float")))
            self.abs_err_var.set(str(cfg.get("abs_err", "1e-2")))
            self.sym_err_var.set(str(cfg.get("sym_err", "1e-2")))
            self.n_max_var.set(str(cfg.get("n_max", "16384")))
            self.with_slopes_var.set(bool(cfg.get("with_slopes", False)))
            self.test_count_var.set(str(cfg.get("test_count", "64")))
            self.xmin_shift_var.set(bool(cfg.get("xmin_shift", False)))
            self.q_frac_var.set(str(cfg.get("q_frac", "20")))
            self.x_shift_var.set(str(cfg.get("x_shift", "12")))
            self.y_scale_var.set(str(cfg.get("y_scale", "100000")))

            self.func_name_var.set(str(cfg.get("func_name", "f_approx")))
            self.array_name_var.set(str(cfg.get("array_name", "tbl_f")))
            self.array_placement_var.set(str(cfg.get("array_placement", default_array_placement())))

            if isinstance(gui_cfg, dict):
                self.auto_relax_var.set(bool(gui_cfg.get("auto_relax", False)))
                if gui_cfg.get("step_pct") is not None:
                    self.relax_step_var.set(str(gui_cfg.get("step_pct")))
                if gui_cfg.get("max_retries") is not None:
                    self.relax_retries_var.set(str(gui_cfg.get("max_retries")))
                if gui_cfg.get("cap_mult") is not None:
                    self.relax_cap_var.set(str(gui_cfg.get("cap_mult")))
        finally:
            self._suspend_codegen_autoupdate = prev_suspend

    def on_history_rerun(self) -> None:
        entry = self._selected_history_entry()
        if not entry:
            return
        cfg = entry.get("config", {})
        if not isinstance(cfg, dict):
            return
        try:
            config = self._config_from_dict(cfg)
            bundle = run_fit(config)
        except Exception as exc:
            msg = self._format_exception(exc, "error_run_detail")
            messagebox.showerror(_t(self.language_var.get(), "run_failed"), msg)
            return
        self.last_bundle = bundle
        self.last_compute_bundle = FitComputationBundle(source=bundle.source, fit=bundle.fit, rows=bundle.rows, estimate=bundle.estimate)
        self.last_rendered_bundle = RenderedOutputBundle(summary=bundle.summary, c_code=bundle.c_code, report_text=bundle.report_text)
        self._update_results(bundle.summary, bundle.c_code)
        self._sync_run_state()
        self._add_history_entry(
            config,
            bundle.summary,
            bundle.c_code,
            bundle.report_text,
            fit_data=fit_result_to_dict(bundle.fit),
            rows=rows_to_list(bundle.rows),
            estimate=estimate_to_dict(bundle.estimate),
            render_config=asdict(self._build_codegen_config()),
        )
        self._set_status_key("status_rerun_complete")
        self._update_plot()

    def on_history_export(self) -> None:
        entry = self._selected_history_entry()
        if not entry:
            return
        report_text = entry.get("report_text")
        if not isinstance(report_text, str):
            return
        path = filedialog.asksaveasfilename(
            title=_t(self.language_var.get(), "save_report"),
            defaultextension=".txt",
            initialdir=str(get_default_output_dir(create=True)),
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            path = resolve_out_path(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report_text + "\n")
        except Exception as exc:
            msg = self._format_error("error_save_detail", str(exc))
            messagebox.showerror(_t(self.language_var.get(), "save_failed"), msg)

    def on_history_delete(self) -> None:
        sel = self.history_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < 0 or idx >= len(self.history):
            return
        del self.history[idx]
        _save_history(self.history)
        self._refresh_history_list()

    def on_batch_csv(self) -> None:
        paths = filedialog.askopenfilenames(
            title=_t(self.language_var.get(), "select_csv"),
            filetypes=[("Data files", "*.csv *.tsv *.txt"), ("All", "*.*")],
        )
        if not paths:
            return
        self._run_batch_csv(list(paths))

    def _run_batch_csv(self, paths: List[str]) -> None:
        ok = 0
        for path in paths:
            try:
                config = FitConfig(
                    path=path,
                    interp=self.interp_var.get(),
                    mode=self.mode_var.get(),
                    abs_err=self._get_float(self.abs_err_var.get(), "abs_err"),
                    sym_err=self._get_float(self.sym_err_var.get(), "sym_err"),
                    n_max=self._get_int(self.n_max_var.get(), "n_max"),
                    with_slopes=self.with_slopes_var.get(),
                    test_count=self._get_int(self.test_count_var.get(), "test_count"),
                    q_frac=self._get_int(self.q_frac_var.get(), "q_frac"),
                    x_shift=self._get_int(self.x_shift_var.get(), "x_shift"),
                    y_scale=self._get_int(self.y_scale_var.get(), "y_scale"),
                    func_name=self.func_name_var.get().strip() or "f_approx",
                    array_name=self.array_name_var.get().strip() or "tbl_f",
                )
                bundle = run_fit(config)
                self._add_history_entry(
                    config,
                    bundle.summary,
                    bundle.c_code,
                    bundle.report_text,
                    fit_data=fit_result_to_dict(bundle.fit),
                    rows=rows_to_list(bundle.rows),
                    estimate=estimate_to_dict(bundle.estimate),
                    render_config=asdict(self._build_codegen_config()),
                )
                ok += 1
            except Exception as exc:
                msg = self._format_exception(exc, "error_batch_detail", item=path)
                messagebox.showwarning(_t(self.language_var.get(), "batch_failed"), msg)
        self.status_var.set(_t(self.language_var.get(), "status_batch_complete").format(ok))

    def on_batch_analytic(self) -> None:
        lines = [line.strip() for line in self.batch_expr_text.get("1.0", tk.END).splitlines()]
        exprs = [line for line in lines if line]
        if not exprs:
            return
        ok = 0
        for expr in exprs:
            try:
                config = FitConfig(
                    func=expr,
                    xmin=self._get_float(self.xmin_var.get(), "xmin"),
                    xmax=self._get_float(self.xmax_var.get(), "xmax"),
                    interp=self.interp_var.get(),
                    mode=self.mode_var.get(),
                    abs_err=self._get_float(self.abs_err_var.get(), "abs_err"),
                    sym_err=self._get_float(self.sym_err_var.get(), "sym_err"),
                    n_max=self._get_int(self.n_max_var.get(), "n_max"),
                    with_slopes=self.with_slopes_var.get(),
                    test_count=self._get_int(self.test_count_var.get(), "test_count"),
                    q_frac=self._get_int(self.q_frac_var.get(), "q_frac"),
                    x_shift=self._get_int(self.x_shift_var.get(), "x_shift"),
                    y_scale=self._get_int(self.y_scale_var.get(), "y_scale"),
                    func_name=self.func_name_var.get().strip() or "f_approx",
                    array_name=self.array_name_var.get().strip() or "tbl_f",
                )
                bundle = run_fit(config)
                self._add_history_entry(
                    config,
                    bundle.summary,
                    bundle.c_code,
                    bundle.report_text,
                    fit_data=fit_result_to_dict(bundle.fit),
                    rows=rows_to_list(bundle.rows),
                    estimate=estimate_to_dict(bundle.estimate),
                    render_config=asdict(self._build_codegen_config()),
                )
                ok += 1
            except Exception as exc:
                msg = self._format_exception(exc, "error_batch_detail", item=expr)
                messagebox.showwarning(_t(self.language_var.get(), "batch_failed"), msg)
        self.status_var.set(_t(self.language_var.get(), "status_batch_analytic_complete").format(ok))


def main() -> None:
    root = tk.Tk()
    app = FitterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
