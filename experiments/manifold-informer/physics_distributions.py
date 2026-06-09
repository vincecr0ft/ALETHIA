r"""Physics-distributions figure (paper B1): substantiate Table 1.

Panel (a): d sigma/d m_ll for SM, a four-fermion working point, and a vertex
working point, with a lower ratio sub-panel (operator/SM). Shows the four-fermion
tail growth and the flat ~2.5% vertex shift.
Panel (b): d sigma/d cos theta* (SM vs vertex) at a high-mass window, with a
ratio sub-panel, evidencing the vertex's forward-backward-asymmetry structure.

All arrays from modules.analytic_smeft (no invention). Prints the tail-growth
factor, the vertex flat-shift percent, and the A_FB values for cross-check
against Table 1 / Section 5(iii).

python3 experiments/manifold-informer/physics_distributions.py
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modules.analytic_smeft import differential_xs, differential_afb, differential_xs_costheta_bin

LAM = 1000.0
C_4F = {"c_lq^(3)": 0.5}      # four-fermion working point
C_VX = {"c_phi_q^(3)": 0.5}   # vertex working point (matches Table 1's ~2.5%)
OUT = HERE / "output_recovery"; OUT.mkdir(exist_ok=True)

def xs(wc, m): return differential_xs(wc, m, lambda_scale=LAM)["differential_xs"]

# ---- panel (a): mass spectrum ----
m = np.linspace(300.0, 2500.0, 60)
sm, ff, vx = xs({}, m), xs(C_4F, m), xs(C_VX, m)
r_ff, r_vx = ff / sm, vx / sm
print(f"[a] four-fermion tail ratio: {r_ff[0]:.2f} (m={m[0]:.0f}) -> {r_ff[-1]:.1f} "
      f"(m={m[-1]:.0f})  = {r_ff[-1]/r_ff[0]:.0f}x growth")
print(f"[a] vertex ratio: {100*(r_vx.mean()-1):.2f}% mean shift, "
      f"range [{100*(r_vx.min()-1):.2f}%, {100*(r_vx.max()-1):.2f}%] (flat)")

# ---- panel (b): angular spectrum at a high-mass point ----
m_b = 1000.0
edges = np.linspace(-1.0, 1.0, 13)
ctr = 0.5 * (edges[:-1] + edges[1:])
def ang(wc):
    return np.array([differential_xs_costheta_bin(wc, m_b, (edges[k], edges[k+1]),
                     lambda_scale=LAM)["differential_xs"] for k in range(len(ctr))]).ravel()
a_sm, a_vx = ang({}), ang(C_VX)
afb_sm = differential_afb({}, m_b, lambda_scale=LAM)["A_FB"][0]
afb_vx = differential_afb(C_VX, m_b, lambda_scale=LAM)["A_FB"][0]
print(f"[b] A_FB(SM)={afb_sm:.4f}  A_FB(vertex)={afb_vx:.4f}  shift={afb_vx-afb_sm:+.4f}")
print(f"[b] vertex/SM angular ratio range [{(a_vx/a_sm).min():.4f}, {(a_vx/a_sm).max():.4f}] "
      f"(slope in cos-theta* => FB structure, not flat)")

# ---- figure ----
fig = plt.figure(figsize=(10.2, 4.5), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1], hspace=0.05)
# (a)
ax0 = fig.add_subplot(gs[0, 0]); ax0r = fig.add_subplot(gs[1, 0], sharex=ax0)
ax0.semilogy(m/1000, sm, "-", c="k", lw=1.6, label="SM")
ax0.semilogy(m/1000, ff, "--", c="C3", lw=1.6, label=r"$c_{\ell q}^{(3)}=0.5$ (four-fermion)")
ax0.semilogy(m/1000, vx, ":", c="C0", lw=2.0, label=r"$c_{Hq}^{(3)}=0.5$ (vertex)")
ax0.set_ylabel(r"$d\sigma/dm_{\ell\ell}$ [pb/GeV]"); ax0.legend(fontsize=8); ax0.grid(alpha=0.3)
ax0.set_title("(a) mass spectrum"); ax0.tick_params(labelbottom=False)
ax0r.plot(m/1000, r_ff, "--", c="C3"); ax0r.plot(m/1000, r_vx, ":", c="C0", lw=2)
ax0r.axhline(1, c="k", lw=0.7, alpha=0.5); ax0r.set_yscale("log")
ax0r.set_xlabel(r"$m_{\ell\ell}$ [TeV]"); ax0r.set_ylabel("ratio / SM"); ax0r.grid(alpha=0.3)
# (b)
ax1 = fig.add_subplot(gs[0, 1]); ax1r = fig.add_subplot(gs[1, 1], sharex=ax1)
ax1.plot(ctr, a_sm, "-o", c="k", ms=3, label=f"SM ($A_{{FB}}={afb_sm:.3f}$)")
ax1.plot(ctr, a_vx, ":s", c="C0", ms=3, label=f"vertex ($A_{{FB}}={afb_vx:.3f}$)")
ax1.set_ylabel(r"$d\sigma/d\cos\theta^\star$ [pb/GeV]"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax1.set_title(rf"(b) angular spectrum, $m_{{\ell\ell}}={m_b:.0f}$ GeV"); ax1.tick_params(labelbottom=False)
ax1r.plot(ctr, a_vx/a_sm, ":s", c="C0", ms=3)
ax1r.axhline(1, c="k", lw=0.7, alpha=0.5)
ax1r.set_xlabel(r"$\cos\theta^\star_{\rm CS}$"); ax1r.set_ylabel("vertex / SM"); ax1r.grid(alpha=0.3)
fig.savefig(OUT / "physics_distributions.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT}/physics_distributions.png")
