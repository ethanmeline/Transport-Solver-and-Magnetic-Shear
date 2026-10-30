"""
1D radial transport with reversed-shear q(r) and χ(ŝ).

    ∂T/∂t + a ∂T/∂r = ∂/∂r [ χ(r) ∂T/∂r ]

    q(r) = q_min + α (r − r0)²,  α = (q_edge − q_min)/(r_max − r0)²
    ŝ(r) = (r/q) dq/dr
    χ(r) = χ_bg − (χ_bg − χ_min) exp(−ŝ² / 2 σ_s²)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Nr    = 300
r_min = 0.0
r_max = 10.0
r     = np.linspace(r_min, r_max, Nr)
dr    = r[1] - r[0]


def q_reversed_shear(r, r0, q_min, q_edge, r_max):
    """q(r) = q_min + α(r−r0)² with q(r_max)=q_edge. Returns q, α, dq/dr."""
    alpha = (q_edge - q_min) / (r_max - r0)**2
    q     = q_min + alpha * (r - r0)**2
    dq_dr = 2.0 * alpha * (r - r0)
    return q, alpha, dq_dr


def magnetic_shear(r, q, dq_dr):
    """ŝ = (r/q) dq/dr; ŝ=0 at r=0."""
    return np.where(r > 0, (r / q) * dq_dr, 0.0)


def chi_from_shear(s_hat, chi_bg, chi_min, s_w):
    """χ = χ_bg − (χ_bg − χ_min) exp(−ŝ² / 2 σ_s²)."""
    return chi_bg - (chi_bg - chi_min) * np.exp(-s_hat**2 / (2.0 * s_w**2))


r0     = 4.0    # q_min / ITB location (ρ ≈ 0.4)
q_min  = 2.0
q_edge = 6.0

q_prof, alpha, dq_dr = q_reversed_shear(r, r0, q_min, q_edge, r_max)
s_hat  = magnetic_shear(r, q_prof, dq_dr)

chi_bg  = 1.0
chi_min = 0.05
s_w     = 0.5     # suppression width in ŝ

chi_r = chi_from_shear(s_hat, chi_bg, chi_min, s_w)

r0_check = r[np.argmin(q_prof)]
print("=" * 60)
print("Profile sanity checks")
print("=" * 60)
print(f"  q_min = {q_prof.min():.4f}  at r = {r0_check:.3f}  (expected r = {r0})")
print(f"  s_hat at r=r0:   {s_hat[np.argmin(q_prof)]:.6f}  (expected 0)")
print(f"  s_hat range:     [{s_hat.min():.3f},  {s_hat.max():.3f}]")
print(f"  chi range:       [{chi_r.min():.4f},  {chi_r.max():.4f}]")
print(f"  chi at barrier:  {chi_r[np.argmin(q_prof)]:.4f}  (expected {chi_min})")
print(f"  q(0) = {q_prof[0]:.3f},  q(r_max) = {q_prof[-1]:.3f}")
print()

a = 1.0

dt_adv  = dr / abs(a)
dt_diff = dr**2 / (2.0 * chi_r.max())
dt      = 0.4 * min(dt_adv, dt_diff)
t_end   = 6.0
Nt      = int(t_end / dt)

print(f"  dt = {dt:.5f}   (advection CFL: {dt_adv:.5f}, diffusion CFL: {dt_diff:.5f})")
print(f"  Nt = {Nt} steps to t_end = {t_end}")
print()

# Pulse starts inside the barrier (r0_T < r0); arrives at r0 at t ≈ (r0 − r0_T)/a.
r0_T = 2.0
T0   = np.exp(-((r - r0_T)**2))


def solve_transport_variable_chi(T_init, r, dt, Nt, a, chi_arr, snapshot_times):
    """
    Explicit Euler, upwind advection, conservative diffusion:

        ∂T/∂t + a ∂T/∂r = ∂/∂r [ χ(r) ∂T/∂r ]

    Face flux: Γ_{i+1/2} = χ_{i+1/2} (T_{i+1}−T_i)/Δr, χ_{i+1/2}=(χ_i+χ_{i+1})/2.
    Conservative form is required so ∇χ does not act as a spurious source.
    """
    T  = T_init.copy()
    dr = r[1] - r[0]
    snapshots = {0.0: T.copy()}
    chi_face = 0.5 * (chi_arr[:-1] + chi_arr[1:])

    for n in range(Nt):
        t_now = (n + 1) * dt

        adv = a * (T[1:-1] - T[:-2]) / dr
        flux_right = chi_face[1:]  * (T[2:]   - T[1:-1]) / dr
        flux_left  = chi_face[:-1] * (T[1:-1] - T[:-2])  / dr
        diff       = (flux_right - flux_left) / dr

        T_new         = T.copy()
        T_new[1:-1]   = T[1:-1] - dt * adv + dt * diff
        T_new[0]      = 0.0
        T_new[-1]     = 0.0
        T             = T_new

        for t_snap in snapshot_times:
            if abs(t_now - t_snap) < dt * 0.55:
                snapshots[round(t_snap, 4)] = T.copy()

    return snapshots


snapshot_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

print("Running with sheared chi(r)...")
snapshots_shear   = solve_transport_variable_chi(
    T0, r, dt, Nt, a, chi_r, snapshot_times)

print("Running with uniform chi (no barrier) for comparison...")
chi_uniform       = np.full(Nr, chi_bg)
snapshots_uniform = solve_transport_variable_chi(
    T0, r, dt, Nt, a, chi_uniform, snapshot_times)

print("Done.\n")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Phase 1: Magnetic Shear  →  Internal Transport Barrier\n"
    r"$\partial T/\partial t + a\,\partial T/\partial r = "
    r"\partial_r\![\chi(r)\,\partial_r T]$",
    fontsize=12, fontweight="bold",
)

sorted_times = sorted(snapshots_shear.keys())
n_t          = len(sorted_times)
blues        = plt.cm.Blues(np.linspace(0.3, 1.0, n_t))
oranges      = plt.cm.Oranges(np.linspace(0.3, 1.0, n_t))

ax = axes[0, 0]
ax.plot(r, q_prof, "steelblue", lw=2.5)
ax.axvline(r0, color="crimson", ls="--", lw=1.5,
           label=rf"$r_0 = {r0}$  (q_min surface)")
ax.axhline(q_min,  color="gray",  ls=":",  lw=1.0, alpha=0.7,
           label=rf"$q_\min = {q_min}$")
ax.fill_betweenx([q_min - 0.2, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 8.0],
                 r_min, r0, color="coral",   alpha=0.08)
ax.fill_betweenx([q_min - 0.2, 8.0],
                 r0,    r_max, color="skyblue", alpha=0.08)
ax.set_xlabel("r  (radial coordinate)", fontsize=10)
ax.set_ylabel("q(r)", fontsize=10)
ax.set_title("Safety factor — reversed shear", fontsize=10)
ax.set_ylim(q_min - 0.3, q_prof.max() + 0.3)
ax.annotate("reversed\nshear  ŝ < 0", xy=(1.8, q_min + 0.3),
            fontsize=8.5, color="darkred", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="gray", alpha=0.8))
ax.annotate("positive\nshear  ŝ > 0", xy=(7.5, q_min + 1.0),
            fontsize=8.5, color="steelblue", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="aliceblue", ec="gray", alpha=0.8))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(r, s_hat, "darkorange", lw=2.5, label=r"$\hat{s}(r)$")
ax.axhline(0, color="k", lw=0.8, alpha=0.5)
ax.axvline(r0, color="crimson", ls="--", lw=1.5,
           label=rf"$r_0 = {r0}$  (ŝ = 0)")
ax.fill_between(r, s_hat, 0,
                where=(s_hat < 0), color="coral",   alpha=0.25, label="ŝ < 0  (reversed)")
ax.fill_between(r, s_hat, 0,
                where=(s_hat > 0), color="skyblue", alpha=0.25, label="ŝ > 0  (normal)")

s_hat_num = np.where(r > 0, (r / q_prof) * np.gradient(q_prof, dr), 0.0)
ax.plot(r, s_hat_num, "k:", lw=1.0, alpha=0.6, label="numerical gradient (check)")

ax.set_xlabel("r  (radial coordinate)", fontsize=10)
ax.set_ylabel(r"$\hat{s}(r) = (r/q)\,\mathrm{d}q/\mathrm{d}r$", fontsize=10)
ax.set_title("Magnetic shear profile", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(r, chi_r, "forestgreen", lw=2.5, label=r"$\chi(r)$  (shear-suppressed)")
ax.axhline(chi_bg,  color="gray",  ls="--", lw=1.2, alpha=0.7,
           label=rf"$\chi_\mathrm{{bg}} = {chi_bg}$")
ax.axhline(chi_min, color="green", ls=":",  lw=1.0, alpha=0.7,
           label=rf"$\chi_\mathrm{{min}} = {chi_min}$")
ax.axvline(r0, color="crimson", ls="--", lw=1.5, label=rf"barrier at $r_0 = {r0}$")
ax.set_xlabel("r  (radial coordinate)", fontsize=10)
ax.set_ylabel(r"$\chi(r)$", fontsize=10)
ax.set_title(
    r"Diffusivity: $\chi = \chi_\mathrm{bg} - (\chi_\mathrm{bg}-\chi_\mathrm{min})"
    r"\,e^{-\hat{s}^2/2\sigma^2}$",
    fontsize=9.5,
)
ax.set_ylim(-0.05, 1.2)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax.annotate("",
            xy=(r0, chi_min + 0.02), xytext=(r0, chi_bg - 0.02),
            arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
ax.text(r0 + 0.15, 0.5,
        f"suppression\n×{chi_bg/chi_min:.0f}", fontsize=8, color="purple")

ax = axes[1, 1]

barrier_mask = chi_r < 0.5 * chi_bg
r_lo, r_hi   = r[barrier_mask][[0, -1]] if barrier_mask.any() else (r0, r0)
ax.axvspan(r_lo, r_hi, color="lightcoral", alpha=0.18, label="low-χ barrier")
ax.axvline(r0, color="crimson", ls="--", lw=1.2, alpha=0.7)

T_shear_final   = snapshots_shear[sorted_times[-1]]
T_uniform_final = snapshots_uniform[sorted_times[-1]]
ax.plot(r, T_shear_final,   "C1",       lw=2.8, label=f"sheared χ  (t={sorted_times[-1]})")
ax.plot(r, T_uniform_final, "C0",       lw=2.8, label=f"uniform χ  (t={sorted_times[-1]})", ls="--")

for t_val, col in zip(sorted_times[:-1], oranges[:-1]):
    ax.plot(r, snapshots_shear[t_val], color=col, lw=1.2, alpha=0.6)

ax.set_xlabel("r  (radial coordinate)", fontsize=10)
ax.set_ylabel("T(r, t)", fontsize=10)
ax.set_title("Temperature: sheared χ vs uniform χ", fontsize=10)
ax.set_xlim(r_min, r_max)
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax.annotate("pulse slows &\n steepens here",
            xy=(r0, 0.3), fontsize=8.5, color="crimson", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

plt.tight_layout()
output_dir  = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "magnetic_shear_itb.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to  {output_path}")

if __name__ == "__main__":
    plt.show()
