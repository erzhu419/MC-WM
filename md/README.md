# MC-WM: Meta-Cognitive World Model with Residual Dynamics

Sim-to-real transfer via symbolic residual world models with self-hypothesis discovery.

## Core Idea

Train a world model M_sim on simulator data, then learn a small residual δ to adapt it to real dynamics: **M_real = M_sim + δ**. The residual δ uses SINDy for symbolic feature discovery + NAU/NMU for bounded OOD extrapolation.

Key innovation: **self-hypothesis loop** — iteratively discover hidden structure in the residual that polynomial features miss, using orthogonal feature expansion and statistical diagnosis.

## Architecture

```
Phase 1: Pretrain M_sim on sim transitions (frozen after training)
Phase 2: Self-hypothesis loop discovers residual structure
  Round 1: SINDy poly2 → diagnose remainder → find orthogonal features
  Round 2: Expanded SINDy → diagnose → find more features
  Round 3: Final fit → lock sparsity pattern
  NAU/NMU fine-tune with formal OOD bounds
Phase 3: MBPO training
  Policy interacts in real env
  M_real generates model rollouts (data augmentation)
  Online refit: SGD warm-start (smooth coefficient updates, locked sparsity)
```

## Key Results (GravityCheetah: sim=2x gravity, real=1x)

| Method | Real Return | vs Baseline |
|--------|-----------|-------------|
| Raw Sim (model-free) | 875 | baseline |
| MBPO + MLP δ | 4,674 | +434% |
| MBPO + SINDy+NAU (no loop) | 5,147 | +488% |
| **MBPO + SINDy+NAU + hypothesis loop + SGD refit** | **5,179** | **+492%** |
| MBPO + Direct M_real (upper bound) | 6,792 | +676% |

## Self-Hypothesis Loop Discoveries

The loop automatically found features that poly2 (degree-2 polynomials) cannot capture:

**Round 1** (300 → 310 features):
- `sin(3·θ)`: correlation 0.41 with remainder — gravity torque is sinusoidal, not polynomial
- `sin(5·θ)`: correlation 0.29 — higher-frequency component
- `cos(3·θ)`: correlation 0.28
- These are **orthogonal** to poly2 basis (verified by Gram-Schmidt projection)

**Round 2** (310 → 316 features):
- `x₀·x₁·x₅`: 3-way interaction (height × angle × joint) — poly2 only has 2-way
- `x₀·x₁·x₂`, `x₁·x₃·x₅`: more 3-way cross terms

## Why SGD Refit Matters

Previous attempts with hypothesis loop **hurt performance** (5147 → 3999) because:
- SINDy's STLSQ is a batch solver — coefficients jump on each refit
- NAU/NMU chases the jumping coefficients → L_eff (Lipschitz constant) explodes
- OOD bound degrades → model rollouts become unreliable

**Fix**: After the loop discovers sparsity pattern, switch to SGD warm-start for online refit:
- Sparsity pattern (which features are active) is **locked**
- Coefficient values update via gradient descent (smooth, no jumps)
- NAU/NMU sees stable targets → L_eff grows slowly → OOD bound maintained

## Components

| Module | File | Purpose |
|--------|------|---------|
| World Model Ensemble | `mc_wm/residual/world_model.py` | Probabilistic MLP ensemble for M_sim |
| SINDy+NAU Adapter | `mc_wm/residual/sindy_nau_adapter.py` | Symbolic residual δ with OOD bounds |
| NAU/NMU Networks | `mc_wm/networks/nau_nmu.py` | Bounded extrapolation layers |
| Orthogonal Expander | `mc_wm/self_audit/orthogonal_expand.py` | Find features in orthogonal complement |
| Diagnosis Battery | `mc_wm/self_audit/diagnosis.py` | 4 statistical tests on remainder |
| RE-SAC Agent | `mc_wm/policy/resac_agent.py` | Ensemble critic + LCB policy |
| GravityCheetah Env | `mc_wm/envs/hp_mujoco/gravity_cheetah.py` | H2O benchmark (2x gravity) |

## Running

```bash
conda create -n MC-WM python=3.10
conda activate MC-WM
pip install gymnasium mujoco torch pysindy statsmodels scipy scikit-learn

# Step 1: Validate residual model
python experiments/step1_residual_model.py

# Step 2: Full MBPO pipeline (c7 = best config)
python experiments/step2_mbrl_residual.py --mode c7
```

## References

- H2O: When to Trust Your Simulator (Niu et al., NeurIPS 2022)
- ReDRAW: Adapting World Models with Latent-State Dynamics Residuals (Lanier et al., 2025)
- MBPO: When to Trust Your Model (Janner et al., NeurIPS 2019)
- SINDy: Discovering Governing Equations (Brunton et al., PNAS 2016)
- NAU/NMU: Neural Arithmetic Units (Madsen & Johansen, ICML 2020)
