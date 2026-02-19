"""
Autograd Visualizer — step through forward & backward passes on small
expression graphs, seeing values and gradients propagate at each node.
"""

import math
import streamlit as st

# ── Colors (match main app dark-mode palette) ───────────────
CLR_TEXT = "#c9d1d9"
CLR_SECONDARY = "#8b949e"
CLR_BORDER = "#30363d"
CLR_ACCENT = "#58a6ff"
CLR_BG = "#21262d"
CLR_GREEN = "#3fb950"
CLR_NODE_DEFAULT = "#30363d"
CLR_NODE_ACTIVE = "#58a6ff"
CLR_NODE_DONE = "#3fb950"


# ═══════════════════════════════════════════════════════════════
# 1. Tape-based Value class
# ═══════════════════════════════════════════════════════════════

class Value:
    """Scalar value with autograd support."""

    def __init__(self, data, _children=(), _op="", name=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(_children)
        self._op = _op
        self.name = name
        # Extended metadata for pedagogical display
        self._backward_src = "None (leaf)"  # human-readable VJP source
        self._captured = {}  # closure captured variables {name: description}

    def __repr__(self):
        return f"Value({self.name}={self.data:.4f}, grad={self.grad:.4f})"

    # ── arithmetic ops ──────────────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, name=str(other))
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        out._backward_src = f"{self.name}.grad += out.grad\n{other.name}.grad += out.grad"
        out._captured = {"out.grad": "upstream gradient"}
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, name=str(other))
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        out._backward_src = (
            f"{self.name}.grad += {other.name}.data * out.grad\n"
            f"{other.name}.grad += {self.name}.data * out.grad"
        )
        out._captured = {
            f"{other.name}.data": f"{other.data:.4f} (swap rule)",
            f"{self.name}.data": f"{self.data:.4f} (swap rule)",
            "out.grad": "upstream gradient",
        }
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        out._backward_src = (
            f"{self.name}.grad += {exponent} * {self.name}.data^({exponent-1}) * out.grad"
        )
        out._captured = {
            "exponent": str(exponent),
            f"{self.name}.data": f"{self.data:.4f}",
            "out.grad": "upstream gradient",
        }
        return out

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other, name=str(other))
        return self * other ** (-1)

    def __rtruediv__(self, other):
        return Value(other, name=str(other)) * self ** (-1)

    # ── transcendental ops ──────────────────────────────────

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += math.cos(self.data) * out.grad

        out._backward = _backward
        out._backward_src = f"{self.name}.grad += cos({self.name}.data) * out.grad"
        out._captured = {
            f"cos({self.name}.data)": f"{math.cos(self.data):.4f}",
            "out.grad": "upstream gradient",
        }
        return out

    def cos(self):
        out = Value(math.cos(self.data), (self,), "cos")

        def _backward():
            self.grad += -math.sin(self.data) * out.grad

        out._backward = _backward
        out._backward_src = f"{self.name}.grad += -sin({self.name}.data) * out.grad"
        out._captured = {
            f"-sin({self.name}.data)": f"{-math.sin(self.data):.4f}",
            "out.grad": "upstream gradient",
        }
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        out._backward_src = f"{self.name}.grad += out.data * out.grad"
        out._captured = {
            "out.data": f"{out.data:.4f} (reuses forward result)",
            "out.grad": "upstream gradient",
        }
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        out._backward_src = f"{self.name}.grad += (1 / {self.name}.data) * out.grad"
        out._captured = {
            f"1/{self.name}.data": f"{1.0/self.data:.4f}",
            "out.grad": "upstream gradient",
        }
        return out

    # ── backward ────────────────────────────────────────────

    def backward(self):
        """Topological sort then reverse-order VJP accumulation."""
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
        return topo


# Module-level helpers for cleaner expression syntax
def sin(v):
    return v.sin()

def cos(v):
    return v.cos()

def exp(v):
    return v.exp()

def log(v):
    return v.log()


# ═══════════════════════════════════════════════════════════════
# 2. Preset expressions
# ═══════════════════════════════════════════════════════════════

def _build_topo_order(root):
    """Return nodes in topological order (leaves first)."""
    topo = []
    visited = set()

    def walk(v):
        if id(v) not in visited:
            visited.add(id(v))
            for c in v._children:
                walk(c)
            topo.append(v)

    walk(root)
    return topo


def _auto_name(topo):
    """Assign readable names to unnamed intermediate nodes."""
    counter = 1
    for v in topo:
        if not v.name:
            v.name = f"a{counter}"
            counter += 1


def _compute_parents(topo):
    """Return dict mapping node id -> list of parent Value nodes."""
    parents = {id(v): [] for v in topo}
    for v in topo:
        for c in v._children:
            if id(c) in parents:
                parents[id(c)].append(v)
    return parents


def _refresh_metadata(topo):
    """Regenerate _backward_src and _captured after names are assigned."""
    for v in topo:
        children = list(v._children)
        op = v._op

        if not op:
            v._backward_src = "None (leaf node)"
            v._captured = {}
        elif op == "+":
            a, b = children[0], children[1]
            v._backward_src = f"{a.name}.grad += out.grad\n{b.name}.grad += out.grad"
            v._captured = {"out.grad": "upstream gradient"}
        elif op == "*":
            a, b = children[0], children[1]
            v._backward_src = (
                f"{a.name}.grad += {b.name}.data * out.grad\n"
                f"{b.name}.grad += {a.name}.data * out.grad"
            )
            v._captured = {
                f"{b.name}.data": f"{b.data:.4f} (swap rule)",
                f"{a.name}.data": f"{a.data:.4f} (swap rule)",
                "out.grad": "upstream gradient",
            }
        elif op.startswith("**"):
            exp_val = float(op[2:])
            c = children[0]
            v._backward_src = (
                f"{c.name}.grad += {exp_val} * {c.name}.data^({exp_val-1}) * out.grad"
            )
            v._captured = {
                "exponent": str(exp_val),
                f"{c.name}.data": f"{c.data:.4f}",
                "out.grad": "upstream gradient",
            }
        elif op == "sin":
            c = children[0]
            v._backward_src = f"{c.name}.grad += cos({c.name}.data) * out.grad"
            v._captured = {
                f"cos({c.name}.data)": f"{math.cos(c.data):.4f}",
                "out.grad": "upstream gradient",
            }
        elif op == "cos":
            c = children[0]
            v._backward_src = f"{c.name}.grad += -sin({c.name}.data) * out.grad"
            v._captured = {
                f"-sin({c.name}.data)": f"{-math.sin(c.data):.4f}",
                "out.grad": "upstream gradient",
            }
        elif op == "exp":
            c = children[0]
            v._backward_src = f"{c.name}.grad += out.data * out.grad"
            v._captured = {
                "out.data": f"{v.data:.4f} (reuses forward result)",
                "out.grad": "upstream gradient",
            }
        elif op == "log":
            c = children[0]
            v._backward_src = f"{c.name}.grad += (1 / {c.name}.data) * out.grad"
            v._captured = {
                f"1/{c.name}.data": f"{1.0/c.data:.4f}",
                "out.grad": "upstream gradient",
            }


PRESETS = {
    "x1 * x2 + sin(x1)": {
        "fn": lambda x1, x2: x1 * x2 + sin(x1),
        "latex": r"f(x_1, x_2) = x_1 \cdot x_2 + \sin(x_1)",
    },
    "(x1 + x2)^2": {
        "fn": lambda x1, x2: (x1 + x2) ** 2,
        "latex": r"f(x_1, x_2) = (x_1 + x_2)^2",
    },
    "exp(-x1^2) + x2 * cos(x1)": {
        "fn": lambda x1, x2: exp(-(x1 ** 2)) + x2 * cos(x1),
        "latex": r"f(x_1, x_2) = e^{-x_1^2} + x_2 \cdot \cos(x_1)",
    },
    "log(x1^2 + 1) - x2 * x1": {
        "fn": lambda x1, x2: log(x1 ** 2 + 1) - x2 * x1,
        "latex": r"f(x_1, x_2) = \ln(x_1^2 + 1) - x_2 \cdot x_1",
    },
}


# ═══════════════════════════════════════════════════════════════
# 3. Graphviz DOT rendering
# ═══════════════════════════════════════════════════════════════

def _node_label(v, show_value, show_grad, grad_val=0.0):
    """Build an HTML-like label for a graphviz node."""
    parts = [v.name]
    if show_value:
        parts.append(f"val = {v.data:.4f}")
    if show_grad:
        parts.append(f"grad = {grad_val:.4f}")
    return "\\n".join(parts)


def _build_dot(topo, forward_done, backward_done, active_node_id=None, grad_snapshot=None):
    """
    Build a Graphviz DOT string.

    Parameters
    ----------
    topo : list[Value]
        Nodes in topological order (leaves first).
    forward_done : set[int]
        IDs of nodes whose forward value has been revealed.
    backward_done : set[int]
        IDs of nodes whose backward grad has been revealed.
    active_node_id : int | None
        ID of the currently active node (highlighted blue).
    """
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        f'  bgcolor="transparent";',
        f'  node [style=filled, fontcolor="{CLR_TEXT}", fontsize=12, '
        f'fontname="monospace", shape=record, penwidth=1.2];',
        f'  edge [color="{CLR_SECONDARY}", fontcolor="{CLR_SECONDARY}", '
        f'fontsize=10, fontname="monospace"];',
    ]

    for v in topo:
        vid = id(v)
        show_val = vid in forward_done
        show_grad = vid in backward_done

        if vid == active_node_id:
            fill = CLR_NODE_ACTIVE
            font = "#ffffff"
        elif show_grad:
            fill = CLR_NODE_DONE
            font = "#ffffff"
        elif show_val:
            fill = CLR_BG
            font = CLR_TEXT
        else:
            fill = CLR_NODE_DEFAULT
            font = CLR_SECONDARY

        grad_val = grad_snapshot.get(vid, 0.0) if grad_snapshot else 0.0
        label = _node_label(v, show_val, show_grad, grad_val)
        op_suffix = f" [{v._op}]" if v._op else ""
        lines.append(
            f'  n{vid} [label="{label}{op_suffix}", '
            f'fillcolor="{fill}", fontcolor="{font}", color="{CLR_BORDER}"];'
        )

    for v in topo:
        for c in v._children:
            lines.append(f"  n{id(c)} -> n{id(v)};")

    lines.append("}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 4. Build steps for the stepper UI
# ═══════════════════════════════════════════════════════════════

def _build_steps(topo, root):
    """
    Return a list of step dicts:
      {"phase": str, "description": str, "latex": str | None,
       "numeric": str | None, "active_id": int,
       "forward_done": set, "backward_done": set,
       "grad_snapshot": dict[int, float]}

    grad_snapshot maps node id -> grad value at this point in the backward
    pass, so shared nodes show partial accumulation correctly.
    """
    steps = []
    forward_done = set()
    empty_grads = {id(v): 0.0 for v in topo}

    # -- Forward pass steps --
    for v in topo:
        if not v._op:
            desc = f"Input **{v.name}** = {v.data:.4f}"
        else:
            children_str = ", ".join(c.name for c in v._children)
            desc = f"Compute **{v.name}** = {v._op}({children_str}) = {v.data:.4f}"

        new_fwd = forward_done | {id(v)}
        steps.append({
            "phase": "Forward Pass",
            "description": desc,
            "latex": None,
            "numeric": None,
            "active_id": id(v),
            "forward_done": set(new_fwd),
            "backward_done": set(),
            "grad_snapshot": dict(empty_grads),
        })
        forward_done = new_fwd

    # -- Backward pass: run step-by-step, capturing grad snapshots --
    backward_order = list(reversed(topo))
    all_forward = set(forward_done)

    # Reset all grads
    for v in topo:
        v.grad = 0.0

    # Seed output grad
    root.grad = 1.0
    backward_done_acc = {id(root)}
    grad_snap = {id(v): v.grad for v in topo}
    steps.append({
        "phase": "Backward Pass",
        "description": f"Seed output grad: **{root.name}**.grad = 1.0",
        "latex": rf"\frac{{\partial L}}{{\partial {root.name}}} = 1",
        "numeric": "1.0",
        "active_id": id(root),
        "forward_done": all_forward,
        "backward_done": set(backward_done_acc),
        "grad_snapshot": dict(grad_snap),
    })

    # Process each node's backward, capturing state after each VJP
    for v in backward_order:
        if not v._children:
            continue

        # Snapshot grads before this node's backward
        children_before = {id(c): c.grad for c in v._children}

        # Run this node's backward
        v._backward()

        # Build VJP descriptions using actual grad deltas
        children = list(v._children)
        op = v._op

        for child in children:
            delta = child.grad - children_before[id(child)]
            if abs(delta) < 1e-15:
                continue  # no contribution to this child from this op

            backward_done_acc = backward_done_acc | {id(child)}
            grad_snap = {id(n): n.grad for n in topo}

            desc = (
                f"Backprop through **{v.name}** ({v._op}) "
                f"to **{child.name}**: grad += {delta:.4f} = {child.grad:.4f}"
            )

            # Build LaTeX and numeric description for this VJP
            latex, numeric = _vjp_latex(v, child, delta)

            steps.append({
                "phase": "Backward Pass",
                "description": desc,
                "latex": latex,
                "numeric": numeric,
                "active_id": id(child),
                "forward_done": all_forward,
                "backward_done": set(backward_done_acc),
                "grad_snapshot": dict(grad_snap),
            })

    return steps


def _vjp_latex(node, child, delta):
    """Return (latex_str, numeric_str) for a single VJP contribution."""
    op = node._op
    children = list(node._children)

    if op == "+":
        latex = (
            rf"\frac{{\partial L}}{{\partial {child.name}}} \mathrel{{+}}= "
            rf"\frac{{\partial L}}{{\partial {node.name}}} \cdot 1"
        )
        numeric = f"{node.grad:.4f} * 1 = {delta:.4f}"

    elif op == "*":
        other = [c for c in children if id(c) != id(child)]
        other = other[0] if other else child  # self*self edge case
        latex = (
            rf"\frac{{\partial L}}{{\partial {child.name}}} \mathrel{{+}}= "
            rf"\frac{{\partial L}}{{\partial {node.name}}} \cdot {other.name}"
        )
        numeric = f"{node.grad:.4f} * {other.data:.4f} = {delta:.4f}"

    elif op.startswith("**"):
        exp_val = float(op[2:])
        latex = (
            rf"\frac{{\partial L}}{{\partial {child.name}}} \mathrel{{+}}= "
            rf"\frac{{\partial L}}{{\partial {node.name}}} \cdot "
            rf"{exp_val:.4g} \cdot {child.name}^{{{exp_val - 1:.4g}}}"
        )
        numeric = (
            f"{node.grad:.4f} * {exp_val:.4g} * {child.data:.4f}^{exp_val - 1:.4g}"
            f" = {delta:.4f}"
        )

    elif op == "sin":
        latex = (
            rf"\frac{{\partial L}}{{\partial {child.name}}} \mathrel{{+}}= "
            rf"\frac{{\partial L}}{{\partial {node.name}}} \cdot \cos({child.name})"
        )
        numeric = f"{node.grad:.4f} * cos({child.data:.4f}) = {delta:.4f}"

    elif op == "cos":
        latex = (
            rf"\frac{{\partial L}}{{\partial {child.name}}} \mathrel{{+}}= "
            rf"\frac{{\partial L}}{{\partial {node.name}}} \cdot (-\\sin({child.name}))"
        )
        numeric = f"{node.grad:.4f} * (-sin({child.data:.4f})) = {delta:.4f}"

    elif op == "exp":
        latex = (
            rf"\frac{{\partial L}}{{\partial {child.name}}} \mathrel{{+}}= "
            rf"\frac{{\partial L}}{{\partial {node.name}}} \cdot {node.name}"
        )
        numeric = f"{node.grad:.4f} * {node.data:.4f} = {delta:.4f}"

    elif op == "log":
        latex = (
            rf"\frac{{\partial L}}{{\partial {child.name}}} \mathrel{{+}}= "
            rf"\frac{{\partial L}}{{\partial {node.name}}} \cdot \\frac{{1}}{{{child.name}}}"
        )
        numeric = f"{node.grad:.4f} * (1/{child.data:.4f}) = {delta:.4f}"

    else:
        latex = None
        numeric = None

    return latex, numeric


# ═══════════════════════════════════════════════════════════════
# 5. Summary table
# ═══════════════════════════════════════════════════════════════

def _summary_table(topo, forward_done, backward_done, grad_snapshot):
    """Return a list of dicts for the summary table."""
    rows = []
    for v in topo:
        vid = id(v)
        rows.append({
            "Node": v.name,
            "Op": v._op or "input",
            "Value": f"{v.data:.4f}" if vid in forward_done else "\u2014",
            "Gradient": f"{grad_snapshot[vid]:.4f}" if vid in backward_done else "\u2014",
        })
    return rows


# ═══════════════════════════════════════════════════════════════
# 6. Streamlit UI
# ═══════════════════════════════════════════════════════════════

st.title("Autograd Visualizer")
st.markdown(
    "Step through the **forward** and **backward** passes of reverse-mode "
    "automatic differentiation on small expression graphs. Watch values propagate "
    "forward and gradients flow backward — one node at a time."
)

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.header("Expression")
preset_name = st.sidebar.selectbox("Preset expression", list(PRESETS.keys()))
preset = PRESETS[preset_name]
st.sidebar.latex(preset["latex"])

st.sidebar.header("Input Values")
x1_val = st.sidebar.slider("x1", -3.0, 3.0, 1.5, 0.1)
x2_val = st.sidebar.slider("x2", -3.0, 3.0, 0.8, 0.1)

st.sidebar.header("Display")
show_internals = st.sidebar.checkbox("Show node internals", False,
    help="Show _backward closure source, captured variables, and graph relationships for each node.")

# ── Build graph ─────────────────────────────────────────────
x1 = Value(x1_val, name="x1")
x2 = Value(x2_val, name="x2")
output = preset["fn"](x1, x2)
topo = _build_topo_order(output)
_auto_name(topo)
output.name = "out"
_refresh_metadata(topo)

# Build step sequence
steps = _build_steps(topo, output)
total_steps = len(steps) - 1

# ── Step controls ───────────────────────────────────────────
st.markdown("---")
col_prev, col_slider, col_next = st.columns([1, 8, 1])
with col_prev:
    if st.button("< Prev", use_container_width=True):
        current = st.session_state.get("ag_step", 0)
        st.session_state["ag_step"] = max(0, current - 1)
with col_next:
    if st.button("Next >", use_container_width=True):
        current = st.session_state.get("ag_step", 0)
        st.session_state["ag_step"] = min(total_steps, current + 1)
with col_slider:
    step_idx = st.slider(
        "Step", 0, total_steps,
        value=st.session_state.get("ag_step", 0),
        key="ag_step",
        label_visibility="collapsed",
    )

step = steps[step_idx]

# ── Phase indicator ─────────────────────────────────────────
phase_color = CLR_ACCENT if step["phase"] == "Forward Pass" else CLR_GREEN
st.markdown(
    f"<div style='text-align:center; padding:4px 0;'>"
    f"<span style='background:{phase_color}; color:#fff; padding:4px 16px; "
    f"border-radius:12px; font-weight:600; font-size:0.9em'>"
    f"{step['phase']}  —  Step {step_idx + 1} / {total_steps + 1}</span></div>",
    unsafe_allow_html=True,
)

# ── Graph visualization ────────────────────────────────────
dot = _build_dot(topo, step["forward_done"], step["backward_done"], step["active_id"], step["grad_snapshot"])
st.graphviz_chart(dot, use_container_width=True)

# ── Detail panel ────────────────────────────────────────────
st.markdown("### Current Step")
st.markdown(step["description"])

if step["latex"]:
    st.latex(step["latex"])
    st.markdown(
        f"<div style='color:{CLR_SECONDARY}; font-family:monospace; "
        f"padding:4px 8px; background:{CLR_BG}; border-radius:6px; "
        f"display:inline-block'>{step['numeric']}</div>",
        unsafe_allow_html=True,
    )

# ── Summary table ───────────────────────────────────────────
st.markdown("### All Nodes")
rows = _summary_table(topo, step["forward_done"], step["backward_done"], step["grad_snapshot"])
st.table(rows)

# ── Node internals panel ──────────────────────────────────
if show_internals:
    parents = _compute_parents(topo)
    node_by_id = {id(v): v for v in topo}

    # Find the active node for this step
    active = node_by_id.get(step["active_id"])

    st.markdown("### Node Internals")
    st.caption("Showing tape metadata stored on each Value object in the computation graph.")

    for v in topo:
        vid = id(v)
        is_active = vid == step["active_id"]
        border = CLR_ACCENT if is_active else CLR_BORDER

        with st.expander(
            f"{'>> ' if is_active else ''}{v.name}"
            f" {'[ACTIVE]' if is_active else ''}"
            f" — {v._op or 'input'}",
            expanded=is_active,
        ):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Identity**")
                st.code(
                    f"name     = {v.name!r}\n"
                    f"_op      = {v._op!r}\n"
                    f"data     = {v.data:.6f}\n"
                    f"grad     = {step['grad_snapshot'].get(vid, 0.0):.6f}",
                    language="python",
                )

                st.markdown("**Graph edges**")
                child_names = [c.name for c in v._children] or ["(none — leaf)"]
                parent_names = [p.name for p in parents.get(vid, [])] or ["(none — root)"]
                st.code(
                    f"_children = {child_names}\n"
                    f" parents  = {parent_names}",
                    language="python",
                )

            with col_b:
                st.markdown("**`_backward` closure**")
                st.code(v._backward_src, language="python")

                if v._captured:
                    st.markdown("**Captured variables**")
                    cap_lines = "\n".join(
                        f"{k:20s} = {val}" for k, val in v._captured.items()
                    )
                    st.code(cap_lines, language="python")
