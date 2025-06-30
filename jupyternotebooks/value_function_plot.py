"""
Interactive plot for

    l(x) = -λ · (0.00167 – x)^(1–γ) / (1–γ)   for x < 0.00167
    g(x) =       (x – 0.00167)^(1–γ) / (1–γ)   for x > 0.00167

Sliders control γ (gamma) and λ (lambda).
Click legend labels to show / hide each curve.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── Constants ────────────────────────────────────────────────────────────────
x_split = 0.00167                         # reference point
x_left   = np.linspace(0.0, x_split - 1e-6, 500)   # domain for l(x)
x_right  = np.linspace(x_split + 1e-6, 0.0030, 500) # domain for g(x)

# Initial parameter values
gamma0  = 0.16
lambda0 = 2.25

# ── Helpers ──────────────────────────────────────────────────────────────────
def l_of_x(x, lam, gam):
    return -lam * (x_split - x) ** (1 - gam) / (1 - gam)

def g_of_x(x, gam):
    return (x - x_split) ** (1 - gam) / (1 - gam)

# ── Figure & first draw ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
plt.subplots_adjust(left=0.10, bottom=0.25)   # leave room for sliders

line_l, = ax.plot(x_left,  l_of_x(x_left,  lambda0, gamma0),
                  lw=2, label=r"$\ell(x)$")
line_g, = ax.plot(x_right, g_of_x(x_right, gamma0),
                  lw=2, label=r"$g(x)$")

ax.set_xlabel("x")
ax.set_ylabel("Value")
ax.set_title(f"γ = {gamma0:.3f},   λ = {lambda0:.3f}")
ax.axvline(x_split, color="k", ls=":", lw=1)
ax.grid(True)

# Interactive legend: clicking toggles visibility
leg = ax.legend(loc="upper right")
legend_lines = leg.get_lines()
plot_lines   = [line_l, line_g]

for leg_line, plot_line in zip(legend_lines, plot_lines):
    leg_line.set_picker(5)  # 5-pt tolerance

def on_pick(event):
    leg_line = event.artist
    try:
        idx = legend_lines.tolist().index(leg_line)
    except ValueError:
        return
    plot_line = plot_lines[idx]
    vis = not plot_line.get_visible()
    plot_line.set_visible(vis)
    # fade legend entry when hidden
    leg_line.set_alpha(1.0 if vis else 0.2)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("pick_event", on_pick)

# ── Sliders ──────────────────────────────────────────────────────────────────
slider_ax_gamma  = plt.axes([0.10, 0.12, 0.80, 0.03])
slider_ax_lambda = plt.axes([0.10, 0.06, 0.80, 0.03])

s_gamma  = Slider(slider_ax_gamma,  r"$\gamma$", 0.12, 0.20, valinit=gamma0)
s_lambda = Slider(slider_ax_lambda, r"$\lambda$", 1.99, 2.50, valinit=lambda0)

def update(val):
    gam = s_gamma.val
    lam = s_lambda.val
    line_l.set_ydata(l_of_x(x_left, lam, gam))
    line_g.set_ydata(g_of_x(x_right, gam))
    ax.set_title(f"γ = {gam:.3f},   λ = {lam:.3f}")
    fig.canvas.draw_idle()

s_gamma.on_changed(update)
s_lambda.on_changed(update)

plt.show()
