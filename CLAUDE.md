# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

nnvis is a multipage Streamlit app with interactive neural network visualizations. Inspired by [Colah's Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/).

## Setup & Running

This project uses [uv](https://docs.astral.sh/uv/) for local development.

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
streamlit run app.py
```

There are no tests or linting configured. Verify changes with `python -c "import ast; ast.parse(open('file.py').read())"`.

## Deployment

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io/).

**Dependencies are declared in two places — keep them in sync:**
- `pyproject.toml` — used by `uv` for local development
- `requirements.txt` — used by Streamlit Community Cloud for deployment

Community Cloud resolves dependencies in this order: `requirements.txt` > `Pipfile` > `environment.yml` > `pyproject.toml`. When it falls through to `pyproject.toml`, it uses Poetry, which fails because this isn't a packaged project. The `requirements.txt` file ensures it uses plain pip instead.

**When adding or changing a dependency, update both files.**

## Architecture

The app uses `st.navigation` in `app.py` (thin entrypoint) to register pages from `pages/`:

- **`app.py`** — entrypoint with `st.set_page_config` + `st.navigation`
- **`pages/0_Neural_Net_Visualizer.py`** (~850 lines) — main space-folding visualizer
- **`pages/1_Autograd_Visualizer.py`** (~570 lines) — autograd step-through visualizer

### Neural Net Visualizer

The page is structured as a top-to-bottom Streamlit script:

1. **Sidebar controls** — all user-configurable parameters (network shape, activation, training, visualization)
2. **Activation functions + derivatives** — pure numpy implementations with a derivatives dict for backprop
3. **Dataset generators** — Two Spirals, Concentric Circles, XOR (each returns `(X, labels)`)
4. **Network setup** — random weight initialization based on sidebar params
5. **`render_frame()`** — the core visualization function; renders layer-by-layer space transformation + decision boundary directly to the current Streamlit context
6. **`train_step()`** — manual SGD backprop that mutates weight arrays in-place
7. **Training + visualization orchestration** — handles both animated and non-animated paths

### Key design decisions

- **No ML framework** — forward pass, backprop, and softmax are all hand-written numpy for transparency and zero dependencies
- **`render_frame()` renders directly to Streamlit** (via `st.*` calls) rather than returning figures. This lets it work inside `st.empty().container()` for animation and mix matplotlib + Plotly in the same render
- **Visualization strategy by hidden width:**
  - `width=2` → exact 2D pcolormesh with gouraud shading (grid topology preserved)
  - `width=3` → interactive Plotly 3D scatter + grid lines (static view) or matplotlib 3D (during animation, for performance)
  - `width>3` → PCA projection to 2D with scatter plot
- **`st.sidebar.empty()` pattern** — used to reserve sidebar slots for content rendered later (e.g., activation plot rendered after `ACTIVATIONS` dict is defined)
- **Axis limits use per-stage lerp** — blends between previous and current tight range controlled by "View adaptation" slider, with clamping to never clip data
- **Dark mode styling** — matplotlib rcParams set transparent backgrounds and GitHub-dark-inspired colors; Plotly uses matching `rgba(0,0,0,0)` backgrounds

### Rendering paths

`render_frame(animating=False)` has two branches:
- **`use_plotly_3d=True`** (has 3D stages, not animating): per-stage rendering via `st.columns()` mixing `_render_stage_3d()` (Plotly) and `_render_stage_2d()` (matplotlib)
- **`use_plotly_3d=False`** (all 2D, or animating): single matplotlib figure with subplots, including `projection='3d'` subplots via `ax.remove()` + `fig.add_subplot()`

### Autograd Visualizer

Step-through visualization of reverse-mode autodiff on small expression graphs:

1. **`Value` class** — tape-based scalar autograd with `_backward` closures for VJP rules
2. **Preset expressions** — 4 presets (`x1*x2+sin(x1)`, `(x1+x2)^2`, etc.) as lambda + LaTeX
3. **`_build_steps()`** — builds forward/backward step sequence with per-step grad snapshots (handles partial accumulation for shared nodes)
4. **Graphviz rendering** — `_build_dot()` generates DOT strings with color-coded nodes (gray/blue/green)
5. **Stepper UI** — slider + prev/next buttons, phase indicator, LaTeX VJP formulas, summary table
