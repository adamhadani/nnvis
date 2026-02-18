# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

nnvis is a single-file Streamlit app that visualizes how feed-forward neural networks "fold" 2D space through successive layers. Inspired by [Colah's Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/).

## Setup & Running

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
streamlit run app.py
```

There are no tests or linting configured. The app is a single `app.py` file with no modules to import. Verify changes with `python -c "import ast; ast.parse(open('app.py').read())"`.

## Architecture

The entire app lives in `app.py` (~860 lines), structured as a top-to-bottom Streamlit script:

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
