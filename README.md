# Fitter

Uniform LUT fitting and C code generation for analytic functions and table data, with GUI and CLI workflows.

## Current Project State

The project now includes:

- a Tkinter GUI with fit controls, plotting, history, batch runs, and multilingual labels/tooltips
- a CLI for scripted use
- a split pipeline (`fit compute` + `render/codegen`) so code output can be regenerated without re-running the fit
- startup dependency preflight from `requirements.txt`
- portable runtime storage/config for source and compiled builds
- codegen formatting features (wrapped arrays, array placement options)
- future-ready algorithm scaffolding (current shipped method: uniform LUT)

## Project Layout

- `core/`: fitting/evaluation math and shared types
- `ios/`: source loading/parsing (tables + analytic expressions)
- `codegen/`: C generators, formatting, memory estimates
- `app/`: GUI, CLI, pipeline, reporting, runtime config/bootstrap
- `build/`: PyInstaller specs (GUI + CLI)
- `tests/`: tests and sample data
- `fitter.py`: source-mode dispatcher entry point

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The app performs a startup preflight check for runtime dependencies and exits early with a clear message if something is missing.

## Running (Source / Python)

### Default dispatcher (GUI if no args, CLI if args are present)

```bash
python fitter.py
python -m app
```

Examples using the dispatcher in CLI mode:

```bash
python fitter.py tests/test.csv --float --test-count 0
python -m app --func "sin(x) + 0.1*x*x" --xmin -2 --xmax 5
```

### Explicit GUI

```bash
python -m app.gui
```

### Explicit CLI

```bash
python -m app.cli <path> [flags]
python -m app.cli --func <expr|name> --xmin <x0> --xmax <x1> [flags]
```

## CLI Usage Examples

### Table source (CSV / TSV / space-delimited)

```bash
python -m app.cli tests/test.csv --linear --float --test-count 0
python -m app.cli tests/test.csv --spline --double --with-slopes
```

### Analytic source

```bash
python -m app.cli --func sin --xmin -2 --xmax 5
python -m app.cli --func linear --a 0.7 --b 0.1 --xmin -2 --xmax 5
python -m app.cli --func "sin(x) + 0.1*x*x" --xmin -2 --xmax 5
python -m app.cli --func-file expr.txt --xmin -2 --xmax 5
```

Built-in analytic names:

- `sin`, `cos`, `exp`, `log`, `linear`

### Output file

```bash
python -m app.cli tests/test.csv --out out/fit_report
python -m app.cli tests/test.csv --out out/fit_report.txt
```

- If `--out` has no extension, `.txt` is appended.
- When `--out` is set, the report is written to file instead of stdout.

## CLI Flags (Current)

### Source / interpolation

- `--linear` / `--spline`
- `<path>` for table data
- `--func <expr|name>`
- `--func-file <path>`
- `--xmin <float> --xmax <float>`
- `--a <float> --b <float>` (for `linear`)

### Error / fit controls

- `--abs-err <float>`
- `--sym-err <float>`
- `--n-max <int>`
- `--xmin-shift`

### Code generation controls

- `--float` (default)
- `--double`
- `--int16`
- `--int32`
- `--with-slopes`
- `--test-count <int>`
- `--func-name <str>`
- `--array-name <str>`
- `--q-frac <int>` / `--x-shift <int>` / `--y-scale <int>` (int modes)

### Misc

- `--plot` (requires `matplotlib`)
- `--out <path>`

## GUI Highlights (Current)

- Table and analytic sources
- Interpolation mode near table-source controls
- Parameter section for error limits, auto-relax, and `Optimize xmin`
- Fitting tabs (`Basic`, `Advanced`, `Algorithms`) with current uniform LUT controls in `Basic`
- Codegen section with numeric mode, slopes, test count, names, array placement, fixed-point settings
- `Update Code` behavior:
  - code/summary are regenerated from cached fit data
  - changing codegen parameters auto-updates output without re-running fitting
- Run progress info shown near the progress bar (estimated `N`, current `N`, best-so-far)
- Plot tab with separate X scales for fit/error subplots
- History restore:
  - restores config and rendered outputs immediately
  - restores fit/plot too for newer history entries with cached fit payloads

## Runtime Config and Storage

The app uses a bundled default config (`app/default_config.json`) plus an optional user override file.

### Source (Python) runs

- storage root: `.fitter_data/` in the repo

### Frozen / compiled runs

- storage root: `fitter_data/` next to the executable

### Stored items

- `config.json` (user overrides)
- `history.json`
- `outputs/` (default export directory)

### Config-driven options (current)

- timing logging on/off (`logging.enabled`)
- codegen max line length for wrapped arrays
- default array placement
- code preview syntax highlighting toggle

## Expression Syntax

- Variable: `x`
- Operators: `+ - * / ^` (`^` means power)
- Parentheses: `(` `)`
- Constants: `pi`, `e`
- Functions include:
  - `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `pow`, `min`, `max`
  - `asin`, `acos`, `atan`, `floor`, `ceil`, `sinh`, `cosh`, `tanh`

## Build (PyInstaller)

Preferred specs (current):

- `pyinstaller_gui.spec` -> `FitterGUI`
- `pyinstaller_cli.spec` -> `FitterCLI`

Example:

```bash
pyinstaller pyinstaller_gui.spec
pyinstaller pyinstaller_cli.spec
```

## Tests

Useful test commands:

```bash
python -m unittest tests.test_pipeline tests.test_ls_fit tests.test_segment_estimator
python -m unittest tests.test_codegen_formatting tests.test_runtime_bootstrap
```

## Notes

- Current fitting method shipped in the GUI/CLI is uniform LUT.
- The codebase already includes planning documents for future expansion (`Plan.md`, `BenchmarkSpec.md`).
