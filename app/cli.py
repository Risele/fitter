from __future__ import annotations

import sys

from app.bootstrap import initialize_runtime_cli_or_exit
initialize_runtime_cli_or_exit()

from app.pipeline import FitConfig, resolve_out_path, run_fit
from app.plotting import plot_fit
from app.reporting import emit_report


def _usage() -> str:
    return "\n".join(
        [
            "Usage: python -m app.cli <path> [flags]  |  python -m app.cli --func <expr|name> --xmin <x0> --xmax <x1> [flags]",
            "Flags:",
            "  --linear | --spline (default: --linear)",
            "  --float | --double | --int16 | --int32 (default: --float)",
            "  --abs-err <float> (default: 1e-2)",
            "  --sym-err <float> (default: 1e-2)",
            "  --n-max <int> (default: 16384)",
            "  --with-slopes (store slopes)",
            "  --test-count <int> (default: 64; 0 disables test vectors)",
            "  --plot (show plot of source and fit)",
            "  --xmin-shift (optimize xmin with left shift)",
            "  --out <path> (write output to file; append .txt if no extension)",
            "  --func <expr|name> (analytic source; name: sin|cos|exp|log|linear; expr uses x and supports ^)",
            "  --func-file <path> (load analytic expression from file)",
            "  --xmin <float> --xmax <float> (analytic source domain)",
            "  --a <float> --b <float> (linear function: a*x + b; defaults 1, 0)",
            "  --q-frac <int> (int modes only; default: 20)",
            "  --x-shift <int> (int modes only; default: 12)",
            "  --y-scale <int> (int modes only; default: 100000)",
            "  --func-name <str> (default: f_approx)",
            "  --array-name <str> (default: tbl_f)",
        ]
    )


def _parse_args(argv: list) -> dict:
    if len(argv) < 2:
        raise ValueError(_usage())

    opts = {
        "path": None,
        "func": None,
        "func_file": None,
        "out": None,
        "xmin": None,
        "xmax": None,
        "a": 1.0,
        "b": 0.0,
        "interp": "linear",
        "mode": None,
        "abs_err": 1e-2,
        "sym_err": 1e-2,
        "n_max": 16384,
        "with_slopes": False,
        "test_count": 64,
        "plot": False,
        "xmin_shift": False,
        "q_frac": 20,
        "x_shift": 12,
        "y_scale": 100000,
        "func_name": "f_approx",
        "array_name": "tbl_f",
    }

    mode_flags = {"--float": "float", "--double": "double", "--int16": "int16", "--int32": "int32"}

    i = 1
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--"):
            if opts["path"] is None:
                opts["path"] = arg
                i += 1
                continue
            raise ValueError("Unexpected positional arg: {}".format(arg))

        if arg in ("--linear", "--spline"):
            opts["interp"] = "linear" if arg == "--linear" else "spline"
            i += 1
            continue

        if arg in mode_flags:
            if opts["mode"] is not None:
                raise ValueError("Only one numeric mode may be selected")
            opts["mode"] = mode_flags[arg]
            i += 1
            continue

        if arg == "--with-slopes":
            opts["with_slopes"] = True
            i += 1
            continue
        if arg == "--plot":
            opts["plot"] = True
            i += 1
            continue
        if arg == "--xmin-shift":
            opts["xmin_shift"] = True
            i += 1
            continue

        def require_value() -> str:
            if i + 1 >= len(argv):
                raise ValueError("Missing value for {}".format(arg))
            return argv[i + 1]

        if arg == "--abs-err":
            opts["abs_err"] = float(require_value())
            i += 2
            continue
        if arg == "--sym-err":
            opts["sym_err"] = float(require_value())
            i += 2
            continue
        if arg == "--n-max":
            opts["n_max"] = int(require_value())
            i += 2
            continue
        if arg == "--test-count":
            opts["test_count"] = int(require_value())
            i += 2
            continue
        if arg == "--func":
            opts["func"] = require_value()
            i += 2
            continue
        if arg == "--func-file":
            opts["func_file"] = require_value()
            i += 2
            continue
        if arg == "--out":
            opts["out"] = require_value()
            i += 2
            continue
        if arg == "--xmin":
            opts["xmin"] = float(require_value())
            i += 2
            continue
        if arg == "--xmax":
            opts["xmax"] = float(require_value())
            i += 2
            continue
        if arg == "--a":
            opts["a"] = float(require_value())
            i += 2
            continue
        if arg == "--b":
            opts["b"] = float(require_value())
            i += 2
            continue
        if arg == "--q-frac":
            opts["q_frac"] = int(require_value())
            i += 2
            continue
        if arg == "--x-shift":
            opts["x_shift"] = int(require_value())
            i += 2
            continue
        if arg == "--y-scale":
            opts["y_scale"] = int(require_value())
            i += 2
            continue
        if arg == "--func-name":
            opts["func_name"] = require_value()
            i += 2
            continue
        if arg == "--array-name":
            opts["array_name"] = require_value()
            i += 2
            continue

        raise ValueError("Unknown flag: {}".format(arg))

    if opts["func"] is None and opts["func_file"] is None and opts["path"] is None:
        raise ValueError(_usage())
    if opts["path"] is not None and (opts["func"] is not None or opts["func_file"] is not None):
        raise ValueError("Provide either a path or an analytic --func/--func-file, not both")
    if opts["func"] is not None and opts["func_file"] is not None:
        raise ValueError("Provide either --func or --func-file, not both")
    if opts["func"] is not None or opts["func_file"] is not None:
        if opts["xmin"] is None or opts["xmax"] is None:
            raise ValueError("Analytic source requires --xmin and --xmax")
        if not (opts["xmax"] > opts["xmin"]):
            raise ValueError("--xmax must be > --xmin")

    if opts["mode"] is None:
        opts["mode"] = "float"

    return opts


def main(argv: list) -> int:
    try:
        opts = _parse_args(argv)
    except Exception as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2

    config = FitConfig(
        path=opts["path"],
        func=opts["func"],
        func_file=opts["func_file"],
        xmin=opts["xmin"],
        xmax=opts["xmax"],
        a=opts["a"],
        b=opts["b"],
        interp=opts["interp"],
        mode=opts["mode"],
        abs_err=opts["abs_err"],
        sym_err=opts["sym_err"],
        n_max=opts["n_max"],
        with_slopes=opts["with_slopes"],
        test_count=opts["test_count"],
        xmin_shift=opts["xmin_shift"],
        q_frac=opts["q_frac"],
        x_shift=opts["x_shift"],
        y_scale=opts["y_scale"],
        func_name=opts["func_name"],
        array_name=opts["array_name"],
    )

    bundle = run_fit(config)

    if opts["out"] is None:
        emit_report(bundle.summary, bundle.c_code, out=sys.stdout)
    else:
        out_path = resolve_out_path(opts["out"])
        with open(out_path, "w", encoding="utf-8") as f:
            emit_report(bundle.summary, bundle.c_code, out=f)

    if opts["plot"]:
        plot_fit(bundle.source, bundle.fit, sample_points=1000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
