# nnvis

Interactive visualization of how neural networks "fold" space. Watch a feed-forward network transform a 2D grid layer by layer — each linear transformation followed by a nonlinearity progressively warps the input space until the classes become separable.

Inspired by [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) by Chris Olah.

![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)

<p align="center">
  <img src="spacefolding-example.png" alt="nnvis screenshot showing space folding visualization with training metrics and layer-by-layer transformation" width="900">
</p>

## Features

| | Feature | Details |
|---|---|---|
| 🧩 | **Layer-by-layer space deformation** | See how each `Wx + b` and activation transforms the 2D grid, with gouraud-shaded pcolormesh colored by class probability |
| 🎯 | **Decision boundary heatmap** | Dense forward pass over input space showing the learned boundary with contour lines |
| 🎬 | **Live training animation** | Watch the network learn in real time as decision boundaries and space folding evolve during SGD |
| 🏗️ | **Configurable architecture** | Adjust depth (1–8 layers), width (2–32 neurons), activation (ReLU, Tanh, Sigmoid, Leaky ReLU, ELU), weight/bias scales, and learning rate |
| 🌐 | **Interactive 3D visualization** | For 3-wide hidden layers, drag-to-rotate Plotly 3D plots let you explore the transformed space from any angle |
| 📉 | **PCA projection** | For layers wider than 3, intermediate representations are projected to 2D via PCA |
| 🌀 | **Toy dataset overlays** | Two Spirals, Concentric Circles, and XOR — see how the network untangles each one |
| 🧮 | **Pure numpy** | No ML framework — forward pass, backprop, and softmax are hand-written for full transparency |

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/adamhadani/nnvis.git
cd nnvis
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```bash
source .venv/bin/activate
streamlit run app.py
```

This opens the app in your browser. Use the sidebar to configure the network and visualization:

1. Set the number of hidden layers and their width
2. Pick an activation function
3. Select a dataset overlay (e.g., Two Spirals)
4. Increase training steps and click "Animate training" to watch the network learn

## How It Works

The app builds a feed-forward network with numpy (no ML framework) and visualizes every intermediate representation:

- A 2D grid of points is passed through each layer
- At each stage (pre-activation `Wx + b` and post-activation), the deformed grid is plotted with points colored by the final softmax probability
- Grid lines overlay shows how the original grid structure warps through the network
- Training uses manual backpropagation with SGD, updating weights in-place

For **width=2** layers, the intermediate space is directly plottable as a 2D mesh. For **width=3**, interactive 3D scatter plots preserve all information. For wider layers, PCA projects to 2D for approximate visualization.

## License

MIT
