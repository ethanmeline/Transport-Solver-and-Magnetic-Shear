# 1D Radial Transport Equation Solver

## Abstract

This code numerically solves 1D radial transport equations relevant to magnetised plasma physics, with a focus on the steepening of plasma profiles driven by turbulent transport and its interaction with magnetic shear. Two transport regimes are studied: a **linear advection–diffusion** equation and the **nonlinear viscous Burgers' equation**, the latter serving as a minimal model for the self-steepening dynamics observed in plasma temperature and density profiles near transport barriers.

The solver was developed as part of a broader research effort examining how **turbulent transport coupled with magnetic shear** can drive the formation and evolution of steep radial gradients in tokamak plasmas. Specifically, this work investigates how regions of reversed magnetic shear — characterised by a minimum in the safety factor profile q(r) — suppress turbulent diffusivity and create **Internal Transport Barriers (ITBs)**. As a temperature pulse propagates radially outward and encounters this low-diffusivity barrier, the profile steepens sharply, analogous to wave breaking in the Burgers equation. The code captures this phenomenon through both analytical estimates (the Burgers breaking time t_b) and direct numerical simulation, providing a tractable model for studying nonlinear profile evolution in the presence of spatially varying transport coefficients.

## Equations

| Mode | Equation |
|------|----------|
| **Linear** | ∂T/∂t + a ∂T/∂r = χ ∂²T/∂r² |
| **Nonlinear (Burgers)** | ∂T/∂t + T ∂T/∂r = χ ∂²T/∂r² |

where `T` is the transported scalar (e.g. temperature), `r` is the radial coordinate, `a` is the advection speed, and `χ` is the diffusivity.

## Features

- **Explicit finite-difference scheme** (forward Euler, upwind advection, central diffusion)
- **CFL-stable time stepping** — automatically computed from grid spacing and wave speed
- **Method of Characteristics (MoC)** exact solution for pure linear advection (`χ = 0`)
- **Burgers' breaking time** computed analytically: t_b = −1 / min(∂T₀/∂r)
- **Magnetic shear extension** — Internal Transport Barrier (ITB) model in `shear_extension.py`
- Comparison across diffusivity values `χ ∈ {0, 0.01, 1}`

## Project Structure

```
transport_equation/
├── main.py               # Core solver: linear & Burgers modes, plotting
├── shear_extension.py    # ITB magnetic shear extension
└── outputs/              # Generated figures (created on first run)
```

## Installation

Requires Python 3.8+ and the following packages:

```bash
pip install numpy matplotlib
```

## Usage

### Default run (nonlinear Burgers, χ = 1)
```bash
python main.py
```

### Linear advection mode
```bash
NONLINEAR=0 CHI=0 python main.py
```

### Vary diffusivity
```bash
CHI=0.01 python main.py
CHI=1.0  python main.py
```

### Chi comparison plot (χ = 0, 0.01, 1 overlaid)
```bash
CHI_COMPARE=1 python main.py
```

### Magnetic shear / ITB extension
```bash
python shear_extension.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHI` | `1.0` | Diffusivity χ (0 = inviscid) |
| `NONLINEAR` | `1` | `1` for Burgers, `0` for linear advection |
| `CHI_COMPARE` | `0` | Set to `1` to run χ comparison study |

## Output

Figures are saved to the `outputs/` directory:

- `transport_1d_plot.png` — Main transport plot (T vs r at multiple times)
- `transport_1d_plot_chi{χ}.png` — Same, labelled by diffusivity
- `breaking_time_plot.png` — Slope steepening and spatial gradient near t_b
- `chi_compare_0_001_1.png` — Side-by-side comparison across χ values

## Physics Background

For the **inviscid Burgers equation** (χ = 0), a smooth initial profile steepens and eventually forms a shock at the **breaking time**:

```
t_b = -1 / min(∂T₀/∂r)
```

Diffusivity χ > 0 regularises the shock, spreading the front instead of forming a discontinuity. The code demonstrates both regimes and the transition between them.

---

## Research Context

This solver was developed in support of a research paper studying **steepening plasma profiles in turbulent transport coupled with magnetic shear** in tokamak-like geometries. The key physical question motivating the work is:

> *How does spatially structured magnetic shear, through its suppression of turbulent diffusivity, cause radial temperature and density profiles to steepen nonlinearly — and under what conditions does this steepening lead to barrier formation?*

The Burgers equation framework provides a controlled, analytically tractable setting to study the same wave-breaking instability that underlies profile stiffness and avalanche-like transport events observed in gyrokinetic simulations and experimental tokamak data. The `shear_extension.py` module extends this to a physically motivated model with a reversed-shear q(r) profile, a spatially varying χ(r) suppressed at the q_min surface, and a direct comparison against uniform-transport baseline runs.

---

## Future Work

Several extensions are planned or identified as natural next steps for developing this model further:

### Physics Extensions
- **2D radial–poloidal transport** — extend from 1D (r) to a 2D (r, θ) geometry to capture the anisotropy of turbulent transport along and across field lines
- **Gyro-Bohm scaling of χ** — replace the fixed diffusivity with a self-consistent χ ~ ρ\* T^(3/2) / B dependence, coupling the transport coefficient to the evolving profile
- **Zonal flow feedback** — incorporate the back-reaction of sheared E×B flows on χ, which is central to the L–H transition physics that ITBs are related to
- **Multiple species** — solve coupled transport equations for ion temperature T_i, electron temperature T_e, and density n simultaneously, including inter-species energy exchange terms
- **Source terms** — add a radially localised heating source S(r) (e.g. NBI or RF) to study how externally driven profiles interact with the transport barrier
- **Neoclassical corrections** — supplement turbulent χ with a neoclassical diffusivity term that dominates in the barrier region where turbulence is suppressed

### Numerical Extensions
- **Implicit time integration** (Crank–Nicolson or BDF2) to remove the diffusive CFL constraint and allow larger time steps at high χ
- **Adaptive mesh refinement** near the barrier region to better resolve sharp gradients without increasing global computational cost
- **Spectral or finite-element discretisation** for higher-order spatial accuracy at the shock front
- **Sensitivity / uncertainty quantification** — sweep over (q_min, χ_min, σ_s) parameter space to map the conditions under which a barrier forms and persists

### Longer-Term Goals
- **Coupling to a gyrokinetic model** as a reduced-order surrogate, validated against full GENE or GS2 simulations
- **Data-driven χ(ŝ, ∇T)** — train a neural-network closure for the diffusivity using simulation data, replacing the Gaussian suppression model with a learned functional form
- **Time-dependent q(r,t)** — allow the safety factor profile to evolve (e.g. during current ramp-up) and study how the ITB location tracks the q_min surface dynamically
