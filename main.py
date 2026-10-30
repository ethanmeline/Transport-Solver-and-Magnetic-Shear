"""
1D radial transport: linear advection, Burgers, and advection–diffusion.

    Linear:     ∂T/∂t + a ∂T/∂r = χ ∂²T/∂r²
    Nonlinear:  ∂T/∂t + T ∂T/∂r = χ ∂²T/∂r²

T: transported scalar (e.g. temperature). r: radial coordinate (dimensionless).
Linear MoC: T(r,t) = φ(r − a t). Burgers breaking time: t_b = −1 / min(∂T₀/∂r).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Nr    = 300
r_min = 0.0
r_max = 10.0
r     = np.linspace(r_min, r_max, Nr)
dr    = r[1] - r[0]

a   = 1.0    # advection speed (linear case only)
chi = float(os.environ.get("CHI", "1.0"))  # diffusivity; 0 for inviscid / breaking-time study

# False: linear (speed a). True: Burgers (speed T); t_b = −1/min(∂T₀/∂r).
nonlinear = os.environ.get("NONLINEAR", "1") == "1"

r0 = 2.0
T0 = np.exp(-((r - r0)**2))
# T0 = np.where((r > 1.5) & (r < 2.5), 1.0, 0.0)
# T0 = np.sin(np.pi * (r - r_min) / (r_max - r_min))


def breaking_time(T_init, dr):
    """Inviscid Burgers: t_b = −1 / min(∂T₀/∂r). Returns (t_b, r_break)."""
    dT0_dr = np.gradient(T_init, dr)
    idx = np.argmin(dT0_dr)
    slope_min = dT0_dr[idx]
    if slope_min >= 0:
        return np.inf, r[idx]
    return -1.0 / slope_min, r[idx]


t_break, r_break = breaking_time(T0, dr)

# CFL: dt ≤ dr/|speed| (advection), dt ≤ dr²/(2χ) (diffusion).
if nonlinear:
    dt_advection = dr / max(np.max(T0), 1e-12)
elif a > 0:
    dt_advection = dr / abs(a)
else:
    dt_advection = np.inf

if chi > 0:
    dt_diffusion = dr**2 / (2.0 * chi)
else:
    dt_diffusion = np.inf

dt = 0.4 * min(dt_advection, dt_diffusion)

if nonlinear and np.isfinite(t_break):
    t_end = 1.5 * t_break
else:
    t_end = 6.0

Nt = int(t_end / dt)

print(f"Grid:       Nr={Nr}, dr={dr:.4f}")
mode = "nonlinear Burgers" if nonlinear else f"linear (a={a})"
print(f"Mode:       {mode}")
print(f"Parameters: χ={chi}")
if nonlinear and np.isfinite(t_break):
    print(f"Breaking time: t_b = {t_break:.4f}  at r ≈ {r_break:.4f}")
    print(f"  min(∂T₀/∂r) = {np.min(np.gradient(T0, dr)):.4f}")
print(f"Time step:  dt={dt:.5f}  (CFL-stable)")
print(f"Steps:      Nt={Nt} to reach t={t_end}")
print()


def analytic_moc(r, t, r0, a):
    """Exact MoC solution for ∂T/∂t + a ∂T/∂r = 0: T(r,t) = exp(−(r−r0−a t)²)."""
    return np.exp(-((r - r0 - a * t)**2))


def solve_transport(T_init, r, dt, Nt, a, chi, snapshot_times, nonlinear=False):
    """
    Explicit FD: forward Euler, upwind advection (a>0), central diffusion.

        Linear:     ∂T/∂t + a ∂T/∂r = χ ∂²T/∂r²
        Nonlinear:  ∂T/∂t + T ∂T/∂r = χ ∂²T/∂r²

    Returns snapshots {t: T} and max_slope_history [(t, max|∂T/∂r|)].
    """
    T        = T_init.copy()
    dr       = r[1] - r[0]
    snapshots = {0.0: T.copy()}
    max_slope_history = [(0.0, np.max(np.abs(np.gradient(T, dr))))]

    for n in range(Nt):
        t_now = (n + 1) * dt

        if nonlinear:
            adv = T[1:-1] * (T[1:-1] - T[:-2]) / dr
        else:
            adv = a * (T[1:-1] - T[:-2]) / dr

        diff = chi * (T[2:] - 2.0 * T[1:-1] + T[:-2]) / dr**2

        T_new        = T.copy()
        T_new[1:-1]  = T[1:-1] - dt * adv + dt * diff
        T_new[0]  = 0.0
        T_new[-1] = 0.0

        T = T_new
        max_slope_history.append(
            (t_now, np.max(np.abs(np.gradient(T, dr))))
        )

        for t_snap in snapshot_times:
            if abs(t_now - t_snap) < dt * 0.55:
                snapshots[round(t_snap, 4)] = T.copy()

    return snapshots, max_slope_history


if nonlinear and np.isfinite(t_break):
    snapshot_times = [
        0.5 * t_break,
        0.95 * t_break,
        t_break,
        1.05 * t_break,
        1.2 * t_break,
    ]
else:
    snapshot_times = [1.0, 2.0, 4.0, 6.0]

snapshots, max_slope_history = solve_transport(
    T0, r, dt, Nt, a, chi, snapshot_times, nonlinear=nonlinear
)

print("Snapshots saved at t =", sorted(snapshots.keys()))


if nonlinear:
    equation_title = r"$\partial T/\partial t + T\,\partial T/\partial r = \chi\,\partial^2 T/\partial r^2$"
else:
    eqiation_title = r"$\partial T/\partial t + a\,\partial T/\partial r = \chi\,\partial^2 T/\partial r^2$"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"1D Transport:  {equation_title}\n  χ = {chi}",
    fontsize=12, fontweight='bold'
)

plot_times  = [0.0] + snapshot_times
n_curves    = len(plot_times)
blue_colors = plt.cm.Blues(np.linspace(0.35, 1.0, n_curves))
orng_colors = plt.cm.Oranges(np.linspace(0.35, 1.0, n_curves))

ax1 = axes[0]

if nonlinear:
    ax1.plot(r, T0, "k--", lw=1.5, alpha=0.5, label=r"$T_0(r)$")
    dT0 = np.gradient(T0, dr)
    ax1b = ax1.twinx()
    ax1b.plot(r, dT0, color="gray", lw=1.2, alpha=0.7, label=r"$\partial T_0/\partial r$")
    ax1b.axhline(0, color="gray", ls=":", lw=0.8)
    ax1b.scatter([r_break], [np.min(dT0)], color="crimson", zorder=5, s=40)
    ax1.set_title(
        f"Initial condition & slope\n"
        rf"$t_b = -1/\min(\partial T_0/\partial r) = {t_break:.3f}$",
        fontsize=10,
    )
    ax1.set_ylabel(r"$T_0(r)$", fontsize=10)
    ax1b.set_ylabel(r"$\partial T_0/\partial r$", fontsize=10, color="gray")
    ax1.legend(loc="upper right", fontsize=9)
    ax1b.legend(loc="upper left", fontsize=9)
else:
    for t_val, col in zip(plot_times, blue_colors):
        T_analytic = analytic_moc(r, t_val, r0, a)
        ax1.plot(r, T_analytic, color=col, lw=2, label=f"t = {t_val:.0f}")
    ax1.set_title(
        "Pure Advection  (χ = 0)\n"
        r"Exact MoC solution: $T(r,t)=\varphi(r - at)$",
        fontsize=10,
    )
    ax1.set_ylabel("T(r, t)  (temperature)", fontsize=10)
    ax1.legend(title="Time", fontsize=9)
    ax1.annotate(
        "Profile shifts rigidly\n— no broadening",
        xy=(r0 + a * 2.5, 0.6), fontsize=8.5, color="steelblue",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )

ax1.set_xlabel("r  (radial coordinate)", fontsize=10)
ax1.set_xlim(r_min, r_max)
ax1.set_ylim(-0.05, 1.15)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]

sorted_times = sorted(snapshots.keys())
for t_val, col in zip(sorted_times, orng_colors):
    lw = 2.5 if nonlinear and abs(t_val - t_break) < 0.02 * t_break else 2
    ax2.plot(r, snapshots[t_val], color=col, lw=lw, label=f"t = {t_val:.3f}")

title_suffix = f"Burgers (χ={chi:g})" if nonlinear else f"Advection + Diffusion (χ={chi:g})"
ax2.set_title(
    f"{title_suffix}\nNumerical solution (finite differences)",
    fontsize=10,
)
ax2.set_xlabel("r  (radial coordinate)", fontsize=10)
ax2.set_ylabel("T(r, t)  (temperature)", fontsize=10)
ax2.legend(title="Time", fontsize=9)
ax2.set_xlim(r_min, r_max)
ax2.set_ylim(-0.05, 1.15)
ax2.grid(True, alpha=0.3)

if nonlinear:
    ax2.axvline(r_break, color="crimson", ls=":", lw=1, alpha=0.6)
    ax2.annotate(
        rf"Vertical slope at $t \approx t_b = {t_break:.3f}$",
        xy=(r_break, 0.85), fontsize=8.5, color="crimson", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )
elif chi > 0:
    ax2.annotate(
        f"Profile shifts AND spreads\n— diffusion (χ={chi}) broadens it",
        xy=(r0 + a * 2.5, 0.55), fontsize=8.5, color="darkorange",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )

plt.tight_layout()
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"transport_1d_plot_chi{chi:g}.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
# also keep a stable default name for the active χ
plt.savefig(output_dir / "transport_1d_plot.png", dpi=150, bbox_inches="tight")

if nonlinear and np.isfinite(t_break):
    times_h, slopes_h = zip(*max_slope_history)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle(
        rf"Breaking-time check: $t_b = {t_break:.4f}$",
        fontsize=12, fontweight="bold",
    )
    ax3.semilogy(times_h, slopes_h, "k-", lw=1.2)
    ax3.axvline(t_break, color="crimson", ls="--", lw=1.5, label=rf"$t_b = {t_break:.3f}$")
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$\max_r |\partial T/\partial r|$")
    ax3.set_title("Slope steepening vs time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    def profile_at(target_t):
        return min(snapshots.items(), key=lambda item: abs(item[0] - target_t))[1]

    t_near = min(times_h, key=lambda t: abs(t - t_break))
    t_before = min(times_h, key=lambda t: abs(t - 0.9 * t_break))
    ax4.plot(r, np.gradient(profile_at(t_near), dr), "C1", lw=2,
             label=rf"$t \approx t_b$ ({t_near:.4f})")
    ax4.plot(r, np.gradient(profile_at(t_before), dr), "C0", lw=1.5, alpha=0.8,
             label=rf"$t < t_b$ ({t_before:.4f})")
    ax4.axvline(r_break, color="crimson", ls=":", lw=1)
    ax4.set_xlabel("r")
    ax4.set_ylabel(r"$\partial T/\partial r$")
    ax4.set_title("Spatial slope near breaking")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    breaking_path = output_dir / "breaking_time_plot.png"
    fig2.savefig(breaking_path, dpi=150, bbox_inches="tight")
    print(f"Breaking-time plot saved to {breaking_path}")

if nonlinear and np.isfinite(t_break):
    past_init = [(t, s) for t, s in max_slope_history if t > 0.05 * t_break]
    t_peak, s_peak = max(past_init, key=lambda x: x[1])
    t_at_tb, s_at_tb = min(past_init, key=lambda x: abs(x[0] - t_break))
    print(f"\nBreaking-time check:")
    print(f"  Theory:  t_b = {t_break:.4f}")
    print(f"  At t ≈ t_b:  max|∂T/∂r| = {s_at_tb:.2f}")
    print(f"  Peak slope:  max|∂T/∂r| = {s_peak:.2f}  at t = {t_peak:.4f}  "
          f"(Δt/t_b = {100*(t_peak-t_break)/t_break:+.1f}%)")
    print("  → Profile steepens sharply around t_b (finite-difference shock lags slightly).")

print(f"\nPlot saved to {output_path}")

# χ ∈ {0, 0.01, 1}: 0.01 interpolates the two limits (no extra qualitative regime).
if os.environ.get("CHI_COMPARE", "0") == "1" and nonlinear and np.isfinite(t_break):
    chi_values = [0.0, 0.01, 1.0]
    compare_times = [0.5 * t_break, t_break, 1.2 * t_break]
    results = {}
    for chi_val in chi_values:
        dt_adv = dr / max(np.max(T0), 1e-12)
        dt_dif = dr**2 / (2.0 * chi_val) if chi_val > 0 else np.inf
        dt_c = 0.4 * min(dt_adv, dt_dif)
        Nt_c = int(t_end / dt_c)
        snaps_c, hist_c = solve_transport(
            T0, r, dt_c, Nt_c, a, chi_val, compare_times, nonlinear=True
        )
        results[chi_val] = (snaps_c, hist_c)

    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle(
        r"Viscous Burgers: $\chi=0$, $0.01$, and $1$"
        "\nintermediate $\\chi$ interpolates the inviscid and strongly diffusive limits",
        fontsize=12, fontweight="bold",
    )
    styles = {0.0: ("C0", "-"), 0.01: ("C1", "--"), 1.0: ("C2", ":")}
    ax_p = axes3[0]
    t_show = t_break
    for chi_val, (snaps_c, _) in results.items():
        t_key = min(snaps_c.keys(), key=lambda t: abs(t - t_show))
        col, ls = styles[chi_val]
        ax_p.plot(r, snaps_c[t_key], color=col, ls=ls, lw=2.2,
                  label=rf"$\chi={chi_val:g}$  ($t\approx t_b$)")
    ax_p.plot(r, T0, "k", lw=1.0, alpha=0.4, label=r"$T_0(r)$")
    ax_p.axvline(r_break, color="crimson", ls=":", lw=1, alpha=0.6)
    ax_p.set_xlabel("r")
    ax_p.set_ylabel("T(r, t)")
    ax_p.set_title(rf"Profiles at $t \approx t_b = {t_break:.3f}$")
    ax_p.set_xlim(r_min, r_max)
    ax_p.set_ylim(-0.05, 1.15)
    ax_p.legend(fontsize=9)
    ax_p.grid(True, alpha=0.3)

    ax_s = axes3[1]
    for chi_val, (_, hist_c) in results.items():
        times_c, slopes_c = zip(*hist_c)
        col, ls = styles[chi_val]
        ax_s.semilogy(times_c, slopes_c, color=col, ls=ls, lw=2.0,
                      label=rf"$\chi={chi_val:g}$")
    ax_s.axvline(t_break, color="crimson", ls="--", lw=1.2, label=rf"$t_b={t_break:.3f}$")
    ax_s.set_xlabel("t")
    ax_s.set_ylabel(r"$\max_r |\partial T/\partial r|$")
    ax_s.set_title("Steepening vs diffusivity")
    ax_s.legend(fontsize=9)
    ax_s.grid(True, alpha=0.3)

    plt.tight_layout()
    compare_path = output_dir / "chi_compare_0_001_1.png"
    fig3.savefig(compare_path, dpi=150, bbox_inches="tight")
    print(f"χ comparison saved to {compare_path}")
    print("\nPeak slope max|∂T/∂r| (after t > 0.05 t_b):")
    for chi_val, (_, hist_c) in results.items():
        past = [(t, s) for t, s in hist_c if t > 0.05 * t_break]
        t_pk, s_pk = max(past, key=lambda x: x[1])
        t_tb, s_tb = min(past, key=lambda x: abs(x[0] - t_break))
        print(f"  χ={chi_val:g}:  peak={s_pk:.3f} at t={t_pk:.4f}  |  at t_b: {s_tb:.3f}")

if __name__ == "__main__":
    plt.show()
