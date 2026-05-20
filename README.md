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

## Install from GitHub in a Clean Environment

Because this repository is private, make sure your terminal can access private
GitHub repositories first. The easiest route is the GitHub CLI:

```bash
gh auth status
gh auth login
gh auth setup-git
```

Then create a fresh environment and install RTVT directly from GitHub:

```bash
conda create -n rtvt-test python=3.12 -y
conda activate rtvt-test
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/shahbandeh/RTVT.git"
```

If you prefer SSH and already have SSH keys configured with GitHub, this should
also work:

```bash
python -m pip install "git+ssh://git@github.com/shahbandeh/RTVT.git"
```

Verify the install:

```bash
rtvt --version
rtvt --help
rtvt --ra 90.0 --dec -1.0 --start-date 2024-01-01 --duration-days 5
```

You should see a terminal summary headed `Roman Target Visibility Tool`.

To test file outputs:

```bash
rtvt --ra 90.0 --dec -1.0 \
  --start-date 2024-01-01 \
  --duration-days 5 \
  --write-csv visibility.csv \
  --write-plot visibility.png
```

That should create `visibility.csv` and `visibility.png` in your current
directory.

## Optional Notebook Install

The base install is enough for the command-line tool and core visibility API.
For notebook interactivity and the Plotly 3D helper, install the optional
extras:

```bash
python -m pip install "rtvt[all] @ git+https://github.com/shahbandeh/RTVT.git"
```

Then install a Jupyter kernel for the environment if needed:

```bash
python -m ipykernel install --user --name rtvt-test --display-name "Python (rtvt-test)"
```

## Local Development Install

If you have cloned this repository and want an editable development install:

```bash
conda create -n rtvt-dev python=3.12 -y
conda activate rtvt-dev
python -m pip install --upgrade pip
python -m pip install -e ".[all,test]"
```

## Command-line Usage

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
