# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of `open-ephys-python-tools` that has grown into the analysis codebase for H-reflex EMG experiments recorded by the **H-Reflex Behavior App** (`hreflex_txbdc`, a PySide6 app that streams from the Open Ephys GUI). The upstream README still describes `control` and `streaming` modules, but this fork only has `analysis`. Almost all active work happens in `src/open_ephys/analysis/helpers.py` and the Jupyter notebooks next to it.

## Flag every assumption

Whenever you make an assumption to get a task done, tell the user right away in a message. Don't save it for the end. This applies to every kind of assumption, including:
- what an ambiguous request means
- how a file format field is laid out or what it means
- thresholds, windows, and other parameter values
- which recording, stage, or notebook was meant
- how app behaviour maps onto code you can't verify
- any fallback or default you chose

For each one, say exactly where it lives (file and line, function, or notebook cell), what you assumed, and why. Ask first when the assumption would change scientific results. In your final message, list all the assumptions you made during the task.

## Environment & commands

- Windows, PowerShell/Git Bash. The virtualenv is `.venv` at the repo root. Use `.venv/Scripts/python` and `.venv/Scripts/pip`. There is no `python` on PATH in Git Bash.
- Install for development: `.venv/Scripts/pip install -e .`. CI uses `uv sync --extra dev` + `uv run pytest tests` on Python 3.12.
- Run tests: `.venv/Scripts/python -m pytest tests`
- Run a single test: `.venv/Scripts/python -m pytest tests/test_h_reflex_comparison_plot.py::test_plot_h_reflex_comparison_shows_mean_sd_and_stimulation_std`
- `tests/test_{binary,nwb,openephys}_format.py` cover the upstream Open Ephys `Session` loaders against `tests/data/v0.6.7_*`. Only `test_h_reflex_comparison_plot.py` touches `helpers.py`, so most H-reflex code has no tests. Verify it by running the notebooks, or by loading real recordings in a script, against the data folders under `src/open_ephys/analysis/` (e.g. `HRPilot-17_*`, `Up-Conditioning_VNS/HRPilot-36`).
- Execute a notebook headlessly: `.venv/Scripts/jupyter nbconvert --to notebook --execute <nb>.ipynb`. Use `--to html` to export.
- Formatter in dev deps: `black`. No lint config is enforced.
- Notebook JSON contains non-cp1252 characters. When reading `.ipynb` from Python, open with `encoding='utf-8'` and set `PYTHONIOENCODING=utf-8` before printing.

## Architecture

### Two unrelated data paths
1. **Upstream Open Ephys loaders**: `session.py` → `recordnode.py` → `recording.py` → `formats/{Binary,Nwb,OpenEphys}Recording.py`. `from open_ephys.analysis import Session`. These read raw GUI recordings and are rarely changed here.
2. **H-Reflex app binary files** (`.hrs1`–`.hrs6`, `.hrft`), all handled in `helpers.py`. This is the main code.

### `helpers.py` (~17k lines, ~200 functions, one flat module)
Notebooks import it as a top-level module (`from helpers import ...`), with the notebook's cwd set to `src/open_ephys/analysis/`. They don't use the package path. Keep it importable that way. Its sections, in file order:
- **Constants**: sample rate (5000 Hz), stim ADC thresholds, block-type IDs, `PERI_STIM_BG_*` (the single standardized −55 to −5 ms pre-stim background window that every viewer must use), `FILTERING_PROTOCOL_*`, `FT_CONDITION_*`.
- **Binary reader primitives** (`hrs_read_val/_string/_datetime`, little-endian, length-prefixed strings) and **data classes** (`MhRecHeader`, `MhRecTrial` and its stage subclasses `DcpTrial`, `FrequencyTestTrial`, `UpCondPelletTrial`, …, plus `EmgDataBlock`).
- **File readers**: `read_hrs1` … `read_hrs6`, `read_hrs_ft`, `find_hrs_files` (returns a 7-tuple of paths), `detect_app_version` (V1/V2/V3).
- Signal helpers, analysis, simulation engine, EEG/respiration pipeline, post-hoc bin analysis.
- **Plotly/ipywidgets viewers**: `make_viewer`, `plot_hrs2_analysis`, `plot_hrs2_trials`, `make_ft_viewer`/`make_ft_sync_viewer`/`make_ft_avg_viewer`, `make_failed_trials_viewer`, `make_filtering_viewer`.
- **Binary writers** (`hrs_write_*`) at the end. These mirror the readers and are used by `Modify_H-reflex_Data.ipynb`.

### File-format versioning is the main hazard
- **Extension meaning changed between app versions.** V1: `.hrs1` = EMG characterization, `.hrs2` = MH recruitment. V2+: `.hrs1` = MH recruitment, `.hrs2` = Control Mode, `.hrs3` = Down-Condition Pellet. V3 adds `.hrs4`/`.hrs5`/`.hrs6` (conditioning stages) and `.hrft` (Frequency Test). Notebooks say `.hrsft`, but the code globs `*.hrft`. Always branch on `detect_app_version()`.
- Each file begins with an `int32 file_version`. Readers gate fields on `header.file_version >= N`. When the app adds fields, add a new version gate and never change the existing ones, because old recordings must still parse. Some trailers have changed length without a version bump. The readers resync by scanning for the next EMG block rather than trusting fixed byte counts. Keep that approach.
- `src/open_ephys/Hreflex_app_files/` is a **reference copy of the app's own source** (PySide6/pyqtgraph; its relative imports don't resolve here). Read `*_data_file.py` / `*_stage.py` there to see what the app actually serializes. Don't import it or try to run it.
- A `settings.json` sidecar may sit next to the binary files. Readers attach it as `header.settings`.

### Notebook architecture
- Main notebooks: `Read_H-Reflex_App.ipynb` and `Read_H-Reflex_App_Simplified.ipynb`. Keep them in sync when changing shared viewer behaviour. Others: `Frequency_Train_Analysis`, `Filtering_Analysis`, `Post_Hoc_Global_Windowing`, `EMG_Trial_Initiation_Simulator`, `Modify_H-reflex_Data`.
- Shared pattern: a `RECORDING_DIRS = [(label, dir, sample_rate_or_None), ...]` list is passed to `load_all_recordings()`, which returns `{label: {stage_map, sample_rate, hrs1_header, ft_trials, ft_header, ft_files, ft_trial_hz, app_version}}`. Viewers are built with `make_viewer(all_recordings, label, render_fn, stage_filter)`, which provides Recording + Stage dropdowns. Put new plotting logic in `helpers.py` as a function and keep notebook cells thin.
- Plotting is Plotly (not matplotlib) for interactive viewers. If Plotly raises nbformat errors in a kernel started before install, restart the kernel.
- Peri-stimulus viewers share `draw_peristim_decorations()` for the grid, stim box, M/H annotations, and DIGIN overlay. Change it there, not per viewer.
- `Old_Analysis_Codes/`, `PythonCommands/`, `Notes/`, `Todo_List.txt`, and the root `test.py` are scratch or legacy files, not part of the package.
