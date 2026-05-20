# Roman Target Visibility Tool (RTVT)

RTVT is a Python tool for quick-look Roman target visibility calculations. It
computes Sun-target separation, field-of-regard visibility, nominal roll, and
related position-angle quantities for fixed sky targets.

The current package keeps the original module names available:

```python
from tgt_vis import compute_visibility
from interactive_visibility_gantt import launch_interactive_sky_gantt
from roman_visibility_3d import roman_suntrack_3d
```

It also installs console commands:

```bash
rtvt --help
roman_tvt --help
```

## Installation

For local development:

```bash
conda create -n rtvt python=3.12
conda activate rtvt
pip install -e .
```

After the private GitHub repository is available, you can test a direct GitHub
install with:

```bash
pip install "git+https://github.com/shahbandeh/RTVT.git"
```

For notebook interactivity and the Plotly 3D helper:

```bash
pip install -e ".[all]"
```

## Command-line usage

Run a fixed-target visibility calculation:

```bash
rtvt --ra 253.2458 --dec 2.4008
```

Restrict the date range and write outputs:

```bash
rtvt --ra 253.2458 --dec 2.4008 \
  --start-date 2026-01-01 \
  --duration-days 365 \
  --sampling-days 1 \
  --write-csv visibility.csv \
  --write-plot visibility.png
```

RA may be given in decimal degrees or sexagesimal hour angle. Dec may be given
in decimal degrees or sexagesimal degrees.

## Notebook usage

In a notebook with the widget backend enabled:

```python
%matplotlib widget
from interactive_visibility_gantt import launch_interactive_sky_gantt

viewer = launch_interactive_sky_gantt(
    grid_step_deg=10,
    duration_days=365,
    sampling_days=1,
)
```

## Development checks

```bash
python -m pytest
python -m build
```

RTVT is currently an alpha-stage research/development package. Validate science
outputs against Roman observatory requirements before operational use.
