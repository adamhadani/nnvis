"""
nnvis - Neural Network Space Folding Visualizer

Interactive visualization of how feed-forward neural networks transform 2D space
through successive layers. Inspired by Colah's "Neural Networks, Manifolds, and
Topology" (https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers '3d' projection

# ── Dark-mode-friendly matplotlib styling ────────────────────
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
})

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="nnvis - Neural Network Space Folding",
    layout="wide",
)

st.title("Neural Network Space Folding")
st.markdown(
    "Watch how a feed-forward network transforms 2D space layer by layer. "
    "Each hidden layer applies a linear transformation (Wx + b) followed by a "
    "nonlinearity, progressively *folding* the space. Points are colored by the "
    "final softmax output probability.\n\n"
    "Inspired by [Colah's Neural Networks, Manifolds, and Topology]"
    "(https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)."
)


# ── Sidebar controls ────────────────────────────────────────
st.sidebar.header("Network Architecture")
num_hidden = st.sidebar.slider("Hidden layers", 1, 8, 3)
hidden_width = st.sidebar.slider(
    "Hidden layer width", 2, 32, 2,
    help="Number of neurons per hidden layer. "
    "Width 2: exact 2D visualization. "
    "Width 3: exact 3D visualization (use elevation/azimuth to rotate). "
    "Width > 3: PCA projection to 2D (approximate).",
)
activation_name = st.sidebar.selectbox(
    "Activation function", ["ReLU", "Tanh", "Sigmoid", "Leaky ReLU", "ELU"]
)
weight_scale = st.sidebar.slider("Weight scale", 0.1, 3.0, 1.2, 0.1)
bias_scale = st.sidebar.slider("Bias scale", 0.0, 2.0, 0.3, 0.05)

st.sidebar.header("Visualization")
grid_n = st.sidebar.slider("Grid resolution", 8, 60, 25)
show_gridlines = st.sidebar.checkbox("Show grid lines", True)
view_adapt = st.sidebar.slider(
    "View adaptation", 0.0, 1.0, 0.5, 0.05,
    help="How quickly axis limits adapt to each layer's range. "
    "0 = fixed global view, 1 = fully auto-scaled, 0.5 = gradual transition.",
)
if hidden_width == 3:
    view_elev = st.sidebar.slider("3D elevation", 0, 90, 30, 5)
    view_azim = st.sidebar.slider("3D azimuth", 0, 360, 45, 5)
else:
    view_elev, view_azim = 30, 45
seed = st.sidebar.number_input("Random seed", 0, 99999, 42, step=1)

st.sidebar.header("Dataset Overlay")
dataset_name = st.sidebar.selectbox(
    "Dataset", ["None", "Two Spirals", "Concentric Circles", "XOR"]
)

st.sidebar.header("Training")
if dataset_name != "None":
    train_steps = st.sidebar.slider("Training steps", 0, 5000, 0, step=50)
    learning_rate = st.sidebar.select_slider(
        "Learning rate",
        options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        value=0.1,
    )
    animate = train_steps > 0 and st.sidebar.checkbox("Animate training")
    if animate:
        steps_per_frame = st.sidebar.slider("Steps per frame", 10, 200, 50, 10)
else:
    train_steps = 0
    learning_rate = 0.1
    animate = False
    st.sidebar.caption("Select a dataset to enable training.")


# ── Activation functions ────────────────────────────────────
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh_act(x):
    return np.tanh(x)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def elu(x):
    return np.where(x > 0, x, np.exp(np.clip(x, -500, 500)) - 1)


ACTIVATIONS = {
    "ReLU": relu,
    "Tanh": tanh_act,
    "Sigmoid": sigmoid,
    "Leaky ReLU": leaky_relu,
    "ELU": elu,
}

ACTIVATION_DERIVS = {
    "ReLU": lambda z: (z > 0).astype(float),
    "Tanh": lambda z: 1 - np.tanh(z) ** 2,
    "Sigmoid": lambda z: sigmoid(z) * (1 - sigmoid(z)),
    "Leaky ReLU": lambda z: np.where(z > 0, 1.0, 0.01),
    "ELU": lambda z: np.where(z > 0, 1.0, np.exp(np.clip(z, -500, 500))),
}


# ── Dataset generators ──────────────────────────────────────
def make_spirals(n=300, noise=0.08, rng=None):
    t = np.linspace(0.5, 3 * np.pi, n)
    r = t / (3 * np.pi) * 1.3
    x1 = r * np.cos(t) + rng.normal(0, noise, n)
    y1 = r * np.sin(t) + rng.normal(0, noise, n)
    x2 = r * np.cos(t + np.pi) + rng.normal(0, noise, n)
    y2 = r * np.sin(t + np.pi) + rng.normal(0, noise, n)
    X = np.vstack([np.c_[x1, y1], np.c_[x2, y2]])
    labels = np.hstack([np.zeros(n), np.ones(n)])
    return X, labels


def make_circles(n=300, noise=0.05, rng=None):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x1 = 0.5 * np.cos(t) + rng.normal(0, noise, n)
    y1 = 0.5 * np.sin(t) + rng.normal(0, noise, n)
    x2 = 1.2 * np.cos(t) + rng.normal(0, noise, n)
    y2 = 1.2 * np.sin(t) + rng.normal(0, noise, n)
    X = np.vstack([np.c_[x1, y1], np.c_[x2, y2]])
    labels = np.hstack([np.zeros(n), np.ones(n)])
    return X, labels


def make_xor(n=400, noise=0.15, rng=None):
    X = rng.uniform(-1.3, 1.3, (n, 2))
    labels = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    X += rng.normal(0, noise, X.shape)
    return X, labels


DATASETS = {
    "Two Spirals": make_spirals,
    "Concentric Circles": make_circles,
    "XOR": make_xor,
}


# ── Network setup ───────────────────────────────────────────
rng = np.random.default_rng(seed)
act_fn = ACTIVATIONS[activation_name]

# Build layers: 2 -> hidden_width -> ... -> hidden_width -> 2 (softmax)
layers = []
for i in range(num_hidden):
    fan_in = 2 if i == 0 else hidden_width
    W = rng.normal(0, weight_scale, (hidden_width, fan_in))
    b = rng.normal(0, bias_scale, (hidden_width,))
    layers.append((W, b))

W_out = rng.normal(0, weight_scale, (2, hidden_width))
b_out = rng.normal(0, bias_scale, (2,))

# Display network metadata near top of page
_total_params = sum(
    W.shape[0] * W.shape[1] + b.shape[0] for W, b in layers
) + W_out.shape[0] * W_out.shape[1] + b_out.shape[0]
_arch_str = " \u2192 ".join(
    ["2"] + [str(hidden_width)] * num_hidden + ["2 (softmax)"]
)
st.caption(
    f"Architecture: {_arch_str}  |  "
    f"{_total_params} trainable parameters ({activation_name})"
)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Generate dataset ────────────────────────────────────────
ds_points = ds_labels = None
if dataset_name != "None":
    ds_points, ds_labels = DATASETS[dataset_name](
        rng=np.random.default_rng(seed + 1)
    )


# ── Pre-compute visualization grids ─────────────────────────
lin = np.linspace(-1.5, 1.5, grid_n)
xx, yy = np.meshgrid(lin, lin)
grid_flat = np.column_stack([xx.ravel(), yy.ravel()])

dense_n = 200
dense_lin = np.linspace(-1.8, 1.8, dense_n)
dx, dy = np.meshgrid(dense_lin, dense_lin)
dense_flat = np.column_stack([dx.ravel(), dy.ravel()])

stage_names = ["Input"]
for i in range(num_hidden):
    stage_names.append(f"Layer {i + 1}\nWx + b")
    stage_names.append(f"Layer {i + 1}\n{activation_name}")
stage_names.append("Output\n(logits)")

cmap = "coolwarm"
vis_norm = mcolors.Normalize(0, 1)


def project_pca(points, n_components=2):
    """Project high-dimensional points to 2D via PCA.

    Returns (projected, mean, basis) so dataset points can be
    projected using the same transformation.
    """
    mean = points.mean(axis=0)
    centered = points - mean
    cov = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; take the top n_components
    top_idx = np.argsort(eigenvalues)[::-1][:n_components]
    basis = eigenvectors[:, top_idx]
    return centered @ basis, mean, basis


# ── Rendering function ──────────────────────────────────────
def render_frame(ws, bs, w_o, b_o, step_info=None):
    """Render both visualizations with the given weights.

    Returns (grid_fig, boundary_fig).
    """
    # Forward pass on visualization grid (with intermediate states)
    states = [grid_flat.copy()]
    for W, b in zip(ws, bs):
        z = states[-1] @ W.T + b
        states.append(z)
        states.append(act_fn(z))
    g_logits = states[-1] @ w_o.T + b_o
    states.append(g_logits)
    pc0 = softmax(g_logits)[:, 0]

    # Forward pass on dataset overlay
    ds_st = None
    if ds_points is not None:
        ds_s = [ds_points.copy()]
        for W, b in zip(ws, bs):
            z = ds_s[-1] @ W.T + b
            ds_s.append(z)
            ds_s.append(act_fn(z))
        ds_s.append(ds_s[-1] @ w_o.T + b_o)
        ds_st = ds_s

    # ── Classify each stage's dimensionality ───────────────
    # dim=2: native 2D (pcolormesh), dim=3: native 3D, dim>3: PCA→2D
    stage_dim = []     # "2d", "3d", or "pca"
    states_vis = []    # projected coords for plotting
    pca_info = []      # (mean, basis) or None per stage
    for pts in states:
        d = pts.shape[1]
        if d == 2:
            stage_dim.append("2d")
            states_vis.append(pts)
            pca_info.append(None)
        elif d == 3:
            stage_dim.append("3d")
            states_vis.append(pts)
            pca_info.append(None)
        else:
            proj, mean, basis = project_pca(pts)
            stage_dim.append("pca")
            states_vis.append(proj)
            pca_info.append((mean, basis))

    # ── Layer-by-layer figure ────────────────────────────────
    n_stages = len(states)
    max_cols = min(5, n_stages)
    n_rows = (n_stages + max_cols - 1) // max_cols

    # Compute axis limits (on the 2D projection for pca/2d stages,
    # and on all 3 axes for 3D stages)
    PAD = 0.05
    stage_lims = []  # (xmin, xmax, ymin, ymax[, zmin, zmax])
    for idx, pts in enumerate(states_vis):
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        xspan = xmax - xmin or 1.0
        yspan = ymax - ymin or 1.0
        tight = [
            xmin - PAD * xspan, xmax + PAD * xspan,
            ymin - PAD * yspan, ymax + PAD * yspan,
        ]
        if stage_dim[idx] == "3d":
            zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
            zspan = zmax - zmin or 1.0
            tight.extend([zmin - PAD * zspan, zmax + PAD * zspan])

        if idx == 0:
            stage_lims.append(tuple(tight))
        else:
            prev = stage_lims[-1]
            # Blend x/y limits (first 4 values)
            bl = [prev[i] + view_adapt * (tight[i] - prev[i]) for i in range(4)]
            bl[0] = min(bl[0], tight[0])
            bl[1] = max(bl[1], tight[1])
            bl[2] = min(bl[2], tight[2])
            bl[3] = max(bl[3], tight[3])
            # For z limits on 3D stages, just use tight bounds
            if stage_dim[idx] == "3d":
                bl.extend(tight[4:6])
            stage_lims.append(tuple(bl))

    fig1, axes = plt.subplots(
        n_rows, max_cols,
        figsize=(4.2 * max_cols, 4.2 * n_rows),
        squeeze=False,
    )

    prob_2d = pc0.reshape(grid_n, grid_n)
    for idx in range(n_stages):
        row, col = divmod(idx, max_cols)
        ax = axes[row][col]
        pts = states_vis[idx]
        sdim = stage_dim[idx]

        if sdim == "3d":
            # Replace 2D axis with a 3D one at the same grid position
            ax.remove()
            ax = fig1.add_subplot(
                n_rows, max_cols, row * max_cols + col + 1,
                projection="3d",
            )
            ax.set_facecolor("none")
            pts_3d = pts.reshape(grid_n, grid_n, 3)
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=pc0, cmap=cmap, s=3, alpha=0.5, vmin=0, vmax=1,
                depthshade=True,
            )
            if show_gridlines:
                for i in range(grid_n):
                    ax.plot(pts_3d[i, :, 0], pts_3d[i, :, 1], pts_3d[i, :, 2],
                            color="w", alpha=0.12, lw=0.3)
                for j in range(grid_n):
                    ax.plot(pts_3d[:, j, 0], pts_3d[:, j, 1], pts_3d[:, j, 2],
                            color="w", alpha=0.12, lw=0.3)
            if ds_st is not None:
                dp = ds_st[idx]
                ax.scatter(dp[ds_labels == 0, 0], dp[ds_labels == 0, 1],
                           dp[ds_labels == 0, 2],
                           c="tab:red", s=8, alpha=0.7, zorder=5)
                ax.scatter(dp[ds_labels == 1, 0], dp[ds_labels == 1, 1],
                           dp[ds_labels == 1, 2],
                           c="tab:blue", s=8, alpha=0.7, zorder=5)
            lim = stage_lims[idx]
            ax.set_xlim(lim[0], lim[1])
            ax.set_ylim(lim[2], lim[3])
            ax.set_zlim(lim[4], lim[5])
            ax.view_init(elev=view_elev, azim=view_azim)
            ax.tick_params(labelsize=5)
            title = stage_names[idx] + "\n(3D)"

        elif sdim == "pca":
            # PCA-projected: use scatter (grid topology lost in projection)
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=pc0, cmap=cmap, s=4, alpha=0.6, vmin=0, vmax=1,
            )
            if ds_st is not None:
                dp = ds_st[idx]
                mean, basis = pca_info[idx]
                dp = (dp - mean) @ basis
                ax.scatter(dp[ds_labels == 0, 0], dp[ds_labels == 0, 1],
                           c="tab:red", s=5, alpha=0.7,
                           edgecolors="#30363d", linewidths=0.15, zorder=5)
                ax.scatter(dp[ds_labels == 1, 0], dp[ds_labels == 1, 1],
                           c="tab:blue", s=5, alpha=0.7,
                           edgecolors="#30363d", linewidths=0.15, zorder=5)
            lim = stage_lims[idx]
            ax.set_xlim(lim[0], lim[1])
            ax.set_ylim(lim[2], lim[3])
            orig_dim = states[idx].shape[1]
            title = stage_names[idx] + f"\n(PCA: {orig_dim}D \u2192 2D)"
            ax.set_aspect("equal")

        else:
            # Native 2D: use pcolormesh with grid structure
            pts_2d = pts.reshape(grid_n, grid_n, 2)
            try:
                ax.pcolormesh(
                    pts_2d[:, :, 0], pts_2d[:, :, 1], prob_2d,
                    cmap=cmap, shading="gouraud", alpha=0.85, vmin=0, vmax=1,
                )
            except Exception:
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    c=pc0, cmap=cmap, s=4, alpha=0.6, vmin=0, vmax=1,
                )
            if show_gridlines:
                for i in range(grid_n):
                    ax.plot(pts_2d[i, :, 0], pts_2d[i, :, 1],
                            color="w", alpha=0.15, lw=0.4)
                for j in range(grid_n):
                    ax.plot(pts_2d[:, j, 0], pts_2d[:, j, 1],
                            color="w", alpha=0.15, lw=0.4)
            if ds_st is not None:
                dp = ds_st[idx]
                ax.scatter(dp[ds_labels == 0, 0], dp[ds_labels == 0, 1],
                           c="tab:red", s=5, alpha=0.7,
                           edgecolors="#30363d", linewidths=0.15, zorder=5)
                ax.scatter(dp[ds_labels == 1, 0], dp[ds_labels == 1, 1],
                           c="tab:blue", s=5, alpha=0.7,
                           edgecolors="#30363d", linewidths=0.15, zorder=5)
            lim = stage_lims[idx]
            ax.set_xlim(lim[0], lim[1])
            ax.set_ylim(lim[2], lim[3])
            title = stage_names[idx]
            ax.set_aspect("equal")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=6)

    for idx in range(n_stages, n_rows * max_cols):
        row, col = divmod(idx, max_cols)
        axes[row][col].set_visible(False)

    fig1.subplots_adjust(right=0.92)
    cbar_ax = fig1.add_axes([0.935, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=vis_norm)
    sm.set_array([])
    fig1.colorbar(sm, cax=cbar_ax, label="P(class 0)")
    plt.tight_layout(rect=[0, 0, 0.92, 1])

    # ── Decision boundary figure ─────────────────────────────
    h = dense_flat.copy()
    for W, b in zip(ws, bs):
        h = act_fn(h @ W.T + b)
    d_prob = softmax(h @ w_o.T + b_o)[:, 0].reshape(dense_n, dense_n)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = ax2.imshow(
        d_prob, extent=[-1.8, 1.8, -1.8, 1.8], origin="lower",
        cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="bilinear",
    )
    ax2.contour(dx, dy, d_prob,
                levels=np.arange(0.1, 1.0, 0.1),
                colors="#8b949e", linewidths=0.5, alpha=0.6)
    ax2.contour(dx, dy, d_prob,
                levels=[0.5], colors="#c9d1d9", linewidths=1.5, linestyles="--")

    if ds_points is not None:
        ax2.scatter(ds_points[ds_labels == 0, 0], ds_points[ds_labels == 0, 1],
                    c="tab:red", s=14, alpha=0.8,
                    edgecolors="#30363d", linewidths=0.3, label="Class 0")
        ax2.scatter(ds_points[ds_labels == 1, 0], ds_points[ds_labels == 1, 1],
                    c="tab:blue", s=14, alpha=0.8,
                    edgecolors="#30363d", linewidths=0.3, label="Class 1")
        ax2.legend(fontsize=8, loc="upper right")

    title = "Softmax Decision Boundary"
    if step_info:
        title = (
            f"Step {step_info['step']} | "
            f"Loss: {step_info['loss']:.4f} | "
            f"Acc: {step_info['acc']:.1%}"
        )
    ax2.set_title(title, fontsize=12, fontweight="bold")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig2.colorbar(im, ax=ax2, label="P(class 0)", shrink=0.8)
    plt.tight_layout()

    return fig1, fig2


# ── Training helper ──────────────────────────────────────────
def train_step(Ws, bs, W_o, b_o, X, one_hot):
    """Run one SGD step. Mutates weight arrays in-place."""
    act_deriv = ACTIVATION_DERIVS[activation_name]
    n = len(one_hot)

    acts = [X]
    pre = []
    for W, b in zip(Ws, bs):
        z = acts[-1] @ W.T + b
        pre.append(z)
        acts.append(act_fn(z))
    logits = acts[-1] @ W_o.T + b_o
    probs = softmax(logits)

    dL = (probs - one_hot) / n
    dW_o = dL.T @ acts[-1]
    db_o = dL.sum(axis=0)
    d = dL @ W_o
    for i in range(len(Ws) - 1, -1, -1):
        dz = d * act_deriv(pre[i])
        dW = dz.T @ acts[i]
        db = dz.sum(axis=0)
        if i > 0:
            d = dz @ Ws[i]
        Ws[i] -= learning_rate * dW
        bs[i] -= learning_rate * db
    W_o -= learning_rate * dW_o
    b_o -= learning_rate * db_o

    return probs


def compute_metrics(probs, one_hot, labels):
    loss = -np.mean(
        np.sum(one_hot * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    )
    acc = np.mean(np.argmax(probs, axis=1) == labels.astype(int))
    return loss, acc


# ── Training + Visualization ────────────────────────────────
train_loss = None
train_acc = None
loss_history = []

if train_steps > 0 and ds_points is not None:
    n_samples = len(ds_labels)
    one_hot = np.zeros((n_samples, 2))
    one_hot[np.arange(n_samples), ds_labels.astype(int)] = 1.0

    Ws = [W.copy() for W, b in layers]
    bs = [b.copy() for W, b in layers]
    W_o = W_out.copy()
    b_o = b_out.copy()

if animate:
    # ── Animated training ────────────────────────────────────
    frame_slot = st.empty()
    progress_bar = st.progress(0.0)

    for step in range(train_steps):
        probs_t = train_step(Ws, bs, W_o, b_o, ds_points, one_hot)

        is_frame = (step + 1) % steps_per_frame == 0 or step == train_steps - 1
        if is_frame:
            loss, acc = compute_metrics(probs_t, one_hot, ds_labels)
            loss_history.append({"step": step + 1, "loss": loss})

            fig1, fig2 = render_frame(
                Ws, bs, W_o, b_o,
                step_info={"step": step + 1, "loss": loss, "acc": acc},
            )
            with frame_slot.container():
                st.subheader("Layer-by-Layer Space Transformation")
                st.pyplot(fig1)
                st.subheader("Decision Boundary in Input Space")
                st.pyplot(fig2)
            plt.close(fig1)
            plt.close(fig2)

            progress_bar.progress((step + 1) / train_steps)

    progress_bar.empty()

    # Final metrics
    _acts = [ds_points]
    for W, b in zip(Ws, bs):
        _acts.append(act_fn(_acts[-1] @ W.T + b))
    _probs = softmax(_acts[-1] @ W_o.T + b_o)
    train_loss, train_acc = compute_metrics(_probs, one_hot, ds_labels)

    layers = list(zip(Ws, bs))
    W_out = W_o
    b_out = b_o

else:
    # ── Non-animated training (if any) then single render ────
    if train_steps > 0 and ds_points is not None:
        record_every = max(1, train_steps // 200)
        for step in range(train_steps):
            probs_t = train_step(Ws, bs, W_o, b_o, ds_points, one_hot)
            if step % record_every == 0:
                loss, _ = compute_metrics(probs_t, one_hot, ds_labels)
                loss_history.append({"step": step, "loss": loss})

        _acts = [ds_points]
        for W, b in zip(Ws, bs):
            _acts.append(act_fn(_acts[-1] @ W.T + b))
        _probs = softmax(_acts[-1] @ W_o.T + b_o)
        train_loss, train_acc = compute_metrics(_probs, one_hot, ds_labels)
        loss_history.append({"step": train_steps, "loss": train_loss})

        layers = list(zip(Ws, bs))
        W_out = W_o
        b_out = b_o

    # Training metrics
    if train_loss is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Training Loss", f"{train_loss:.4f}")
        m2.metric("Accuracy", f"{train_acc:.1%}")
        m3.metric("Steps", train_steps)
        st.line_chart(
            pd.DataFrame(loss_history).set_index("step"),
            height=200,
        )

    # Render visualizations once
    ws_cur = [W for W, b in layers]
    bs_cur = [b for W, b in layers]
    fig1, fig2 = render_frame(ws_cur, bs_cur, W_out, b_out)
    st.subheader("Layer-by-Layer Space Transformation")
    st.pyplot(fig1)
    plt.close(fig1)
    st.subheader("Decision Boundary in Input Space")
    st.pyplot(fig2)
    plt.close(fig2)


with st.expander("Layer Weights & Biases"):
    for i, (W, b) in enumerate(layers):
        st.markdown(f"**Hidden Layer {i + 1}**")
        c1, c2 = st.columns(2)
        c1.code(f"W = {np.array2string(W, precision=3)}")
        c2.code(f"b = {np.array2string(b, precision=3)}")
    st.markdown("**Output Layer**")
    c1, c2 = st.columns(2)
    c1.code(f"W = {np.array2string(W_out, precision=3)}")
    c2.code(f"b = {np.array2string(b_out, precision=3)}")
