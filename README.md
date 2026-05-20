# Roman Target Visibility Tool (RTVT)

RTVT is a prototype Roman target visibility tool inspired by the JWST General
Target Visibility Tool (GTVT). It computes quick-look visibility windows for
fixed sky targets using Roman field-of-regard constraints, then exposes the
results through a command-line interface, importable Python API, and interactive
notebook visualizations.

Current capabilities include:

- Sun-target separation and in/out-of-field-of-regard sampling
- Nominal roll and focal-plane position-angle quantities
- Static command-line summaries, CSV output, and visibility plots
- A desktop pop-up GUI for coordinate entry, summaries, plots, tables, and exports
- Equatorial RA/Dec and Galactic l/b input modes
- An interactive notebook sky selector with cumulative Gantt-style visibility windows
  and multi-target Sun-separation comparison
- A Plotly 3D Sun-track helper for geometry exploration

RTVT is an alpha-stage research/development package. Validate science outputs
against current Roman observatory requirements before operational use.

## Installation and Usage

### Quick Local Demo with `venv`

From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all,test]"
```

Verify the install:

```bash
rtvt --version
rtvt --help
rtvt --ra 90.0 --dec -1.0 --start-date 2024-01-01 --duration-days 5
rtvt-gui
```

Test file outputs:

```bash
rtvt --ra 90.0 --dec -1.0 \
  --start-date 2024-01-01 \
  --duration-days 5 \
  --write-csv visibility.csv \
  --write-plot visibility.png \
  --write-report visibility_report.html
```

That creates `visibility.csv`, `visibility.png`, and `visibility_report.html`
in the current directory.

To open the plot from the command line as well:

```bash
rtvt --ra 90.0 --dec -1.0 \
  --start-date 2024-01-01 \
  --duration-days 365 \
  --show-plot
```

To save and show the same plot:

```bash
rtvt --ra 90.0 --dec -1.0 \
  --start-date 2024-01-01 \
  --duration-days 365 \
  --write-plot visibility.png \
  --show-plot
```

When `--show-plot` opens a Matplotlib window, close the window to return to the
terminal prompt. On systems without an interactive Matplotlib backend, RTVT will
still save plots with `--write-plot`.

### Install Directly from GitHub

Once the repository is public, install the latest version from `main`:

```bash
python -m pip install "git+https://github.com/shahbandeh/RTVT.git"
```

For notebook widgets and the 3D Plotly helper:

```bash
python -m pip install "rtvt[all] @ git+https://github.com/shahbandeh/RTVT.git"
```

### Conda Environment Option

```bash
conda create -n rtvt python=3.12 -y
conda activate rtvt
python -m pip install --upgrade pip
python -m pip install "rtvt[all] @ git+https://github.com/shahbandeh/RTVT.git"
```

To use the environment from Jupyter:

```bash
python -m ipykernel install --user --name rtvt --display-name "Python (rtvt)"
```

## Command-Line Examples

Run a fixed-target visibility calculation:

```bash
rtvt --ra 253.2458 --dec 2.4008
```

Restrict the date range and sampling:

```bash
rtvt --ra 253.2458 --dec 2.4008 \
  --start-date 2026-01-01 \
  --duration-days 365 \
  --sampling-days 1
```

Use Galactic coordinates instead of equatorial RA/Dec:

```bash
rtvt --coordinate-system galactic \
  --lon 0.0 \
  --lat 0.0 \
  --start-date 2026-01-01 \
  --duration-days 365 \
  --sampling-days 1
```

Write products:

```bash
rtvt --ra 253.2458 --dec 2.4008 \
  --start-date 2026-01-01 \
  --duration-days 365 \
  --write-csv visibility.csv \
  --write-plot visibility.png \
  --write-report visibility_report.html
```

RA may be provided in decimal degrees or sexagesimal hour angle. Dec may be
provided in decimal degrees or sexagesimal degrees. In Galactic mode, longitude
and latitude are entered as decimal-degree `l` and `b`.

## GUI Usage

RTVT includes a desktop pop-up GUI for demos and exploratory target checks.
Launch it from an environment where RTVT is installed:

```bash
rtvt-gui
```

The GUI provides:

- an Equatorial/Galactic coordinate selector
- RA/Dec, start date, duration, and sampling inputs
- terminal-style summary output
- visibility and Sun-target separation plots
- observable-window table
- sampled-data preview
- save buttons for CSV, PNG/PDF plot output, and an HTML report
- a button to copy the equivalent CLI command

The GUI opens as a normal desktop window. Close the window when you are done.

## Python API

The package keeps the original module names available for notebook
compatibility:

```python
from tgt_vis import compute_visibility
from interactive_visibility_gantt import launch_interactive_sky_gantt
from roman_visibility_3d import roman_suntrack_3d
```

The packaged wrapper also exposes the core calculator:

```python
from rtvt import compute_visibility
```

## Notebook Usage

In a notebook using the RTVT environment:

```python
%matplotlib widget
from interactive_visibility_gantt import launch_interactive_sky_gantt_with_controls

viewer = launch_interactive_sky_gantt_with_controls(
    grid_step_deg=10,
    duration_days=365,
    sampling_days=1,
)
```

The notebook [Run_viz_tool_interactive_gantt.ipynb](Run_viz_tool_interactive_gantt.ipynb)
is the current demo entry point. It includes an Equatorial/Galactic selector,
an exact coordinate-entry panel next to the all-sky map, the latest target
visibility plot, a cumulative Gantt chart with the latest target highlighted,
a multi-target Sun-separation comparison plot, and an HTML report button for
the current selected-target state.

## Relationship to JWST GTVT

JWST GTVT provides an installable command-line visibility tool for JWST targets.
RTVT follows the same general product direction for Roman:

- installable Python package
- command-line entry point for repeatable calculations
- text/CSV and plot outputs for quick inspection
- importable Python functions for notebooks and higher-level workflows

The mission geometry and field-of-regard rules are Roman-specific, so RTVT does
not reuse JWST GTVT calculations directly.

## Development Checks

```bash
python -m pytest
python -m pip wheel . --no-deps -w dist
```
