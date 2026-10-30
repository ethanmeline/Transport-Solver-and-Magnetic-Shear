# 1D Radial Transport Equation Solver

Numerical solver for 1D radial transport equations arising in magnetic shear research. Implements both linear advection–diffusion and nonlinear Burgers' equation using finite differences.

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
