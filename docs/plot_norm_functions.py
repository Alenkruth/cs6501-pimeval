"""
Visualize RMSNorm and LayerNorm step-by-step transformations.

Uses integer arithmetic matching the PIM kernel implementations:
  - int32 values, integer division (truncation toward zero)
  - Newton-Raphson integer square root
  - epsilon = 1, gamma = 1, beta = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def newton_sqrt(val):
    """Integer square root via Newton-Raphson (matches PIM host code)."""
    if val <= 0:
        return 0
    x = val
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + val // x) // 2
    return x


def rmsnorm_steps(x):
    """Return intermediate arrays for each RMSNorm step (integer arithmetic)."""
    n = len(x)
    sq = x * x                              # Step 1: square
    sum_sq = int(np.sum(sq))                # Step 2: reduce
    mean_sq = sum_sq // n                   # Step 3: mean
    rms = newton_sqrt(mean_sq + 1)          # Step 4: sqrt(mean + epsilon)
    y = x // (rms + 1)                      # Step 5: normalize
    return sq, sum_sq, mean_sq, rms, y


def layernorm_steps(x):
    """Return intermediate arrays for each LayerNorm step (integer arithmetic)."""
    n = len(x)
    s = int(np.sum(x))                      # Step 1: reduce
    mean = s // n                           # Step 2: mean
    c = x - mean                            # Step 3: center
    sq = c * c                              # Step 4: square centered
    sum2 = int(np.sum(sq))                  # Step 5: reduce
    var = sum2 // n                         # Step 6: variance
    sqrt_var = newton_sqrt(var + 1)         # Step 7: sqrt(var + epsilon)
    y = c // sqrt_var                       # Step 8: normalize
    return s, mean, c, sq, sum2, var, sqrt_var, y


# --- Generate sample input ---
np.random.seed(42)
n = 64
x = np.random.randint(-50, 120, size=n).astype(np.int32)
idx = np.arange(n)

# --- Compute steps ---
sq_r, sum_sq, mean_sq, rms, y_rms = rmsnorm_steps(x)
s_l, mean_l, c_l, sq_l, sum2_l, var_l, sqrt_var_l, y_ln = layernorm_steps(x)


# ============================================================
# Figure 1: RMSNorm step-by-step
# ============================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle("RMSNorm  (integer arithmetic, n={})".format(n),
              fontsize=15, fontweight="bold", y=0.98)

# (a) Input vector
ax = axes1[0, 0]
ax.bar(idx, x, width=0.8, color="steelblue", edgecolor="none")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Step 0: Input vector  x")
ax.set_xlabel("index i")
ax.set_ylabel("x[i]")

# (b) Squared values
ax = axes1[0, 1]
ax.bar(idx, sq_r, width=0.8, color="coral", edgecolor="none")
ax.set_title("Step 1: x[i]^2   (pimMul)")
ax.set_xlabel("index i")
ax.set_ylabel("x[i]^2")
ax.annotate(f"sum = {sum_sq}\nmean_sq = {mean_sq}\nrms = {rms}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

# (c) Overlay: input vs output
ax = axes1[1, 0]
w = 0.35
ax.bar(idx - w/2, x, width=w, color="steelblue", label="input x[i]", edgecolor="none")
ax.bar(idx + w/2, y_rms, width=w, color="seagreen", label="output y[i]", edgecolor="none")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Step 5: y[i] = x[i] / (rms + 1)   (pimDivScalar)")
ax.set_xlabel("index i")
ax.set_ylabel("value")
ax.legend(loc="upper left", fontsize=9)

# (d) Scatter: input -> output mapping
ax = axes1[1, 1]
ax.scatter(x, y_rms, s=18, color="seagreen", edgecolors="k", linewidths=0.3, zorder=3)
xs = np.linspace(x.min(), x.max(), 200)
ax.plot(xs, xs / (rms + 1), color="coral", lw=1.5, ls="--", label=f"x / (rms+1) = x / {rms+1}")
ax.axhline(0, color="k", lw=0.4)
ax.axvline(0, color="k", lw=0.4)
ax.set_title("Transfer function: x -> y")
ax.set_xlabel("x[i]")
ax.set_ylabel("y[i]")
ax.legend(fontsize=9)

fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig("/data/akrish/coursework/cs6501-memsytems/assignment_pimeval/docs/rmsnorm_visualization.png",
             dpi=150, bbox_inches="tight")


# ============================================================
# Figure 2: LayerNorm step-by-step
# ============================================================
fig2 = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.38, wspace=0.28)
fig2.suptitle("LayerNorm  (integer arithmetic, n={})".format(n),
              fontsize=15, fontweight="bold", y=0.98)

# (a) Input vector
ax = fig2.add_subplot(gs[0, 0])
ax.bar(idx, x, width=0.8, color="steelblue", edgecolor="none")
ax.axhline(0, color="k", lw=0.5)
ax.axhline(mean_l, color="red", lw=1.2, ls="--", label=f"mean = {mean_l}")
ax.set_title("Step 0-2: Input x  +  mean(x)")
ax.set_xlabel("index i")
ax.set_ylabel("x[i]")
ax.legend(fontsize=9)

# (b) Centered vector
ax = fig2.add_subplot(gs[0, 1])
ax.bar(idx, c_l, width=0.8, color="mediumpurple", edgecolor="none")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Step 3: c[i] = x[i] - mean   (pimSubScalar)")
ax.set_xlabel("index i")
ax.set_ylabel("c[i]")

# (c) Squared centered values
ax = fig2.add_subplot(gs[1, 0])
ax.bar(idx, sq_l, width=0.8, color="coral", edgecolor="none")
ax.set_title("Step 4: c[i]^2   (pimMul)")
ax.set_xlabel("index i")
ax.set_ylabel("c[i]^2")
ax.annotate(f"sum2 = {sum2_l}\nvar = {var_l}\nsqrt_var = {sqrt_var_l}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

# (d) Overlay: input vs output
ax = fig2.add_subplot(gs[1, 1])
w = 0.35
ax.bar(idx - w/2, x, width=w, color="steelblue", label="input x[i]", edgecolor="none")
ax.bar(idx + w/2, y_ln, width=w, color="seagreen", label="output y[i]", edgecolor="none")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Step 8: y[i] = c[i] / sqrt_var   (pimDivScalar)")
ax.set_xlabel("index i")
ax.set_ylabel("value")
ax.legend(loc="upper left", fontsize=9)

# (e) Scatter: input -> output mapping
ax = fig2.add_subplot(gs[2, 0])
ax.scatter(x, y_ln, s=18, color="seagreen", edgecolors="k", linewidths=0.3, zorder=3)
xs = np.linspace(x.min(), x.max(), 200)
ax.plot(xs, (xs - mean_l) / sqrt_var_l, color="coral", lw=1.5, ls="--",
        label=f"(x - {mean_l}) / {sqrt_var_l}")
ax.axhline(0, color="k", lw=0.4)
ax.axvline(mean_l, color="red", lw=0.8, ls=":", label=f"mean = {mean_l}")
ax.set_title("Transfer function: x -> y")
ax.set_xlabel("x[i]")
ax.set_ylabel("y[i]")
ax.legend(fontsize=9)

# (f) Comparison: RMSNorm vs LayerNorm output
ax = fig2.add_subplot(gs[2, 1])
w = 0.35
ax.bar(idx - w/2, y_rms, width=w, color="darkorange", label="RMSNorm y[i]", edgecolor="none")
ax.bar(idx + w/2, y_ln,  width=w, color="seagreen",   label="LayerNorm y[i]", edgecolor="none")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Comparison: RMSNorm vs LayerNorm output")
ax.set_xlabel("index i")
ax.set_ylabel("y[i]")
ax.legend(fontsize=9)

fig2.savefig("/data/akrish/coursework/cs6501-memsytems/assignment_pimeval/docs/layernorm_visualization.png",
             dpi=150, bbox_inches="tight")

print("Saved:")
print("  docs/rmsnorm_visualization.png")
print("  docs/layernorm_visualization.png")
plt.show()
