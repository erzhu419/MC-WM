"""
HP-MuJoCo benchmark: 4 environments with known sim-real gaps.

Each env exposes two modes via `mode` argument:
  mode="sim"  → standard MuJoCo (no gap)
  mode="real" → MuJoCo + perturbation (the ground-truth gap)

Gap types (dev manual §9.1):
  AeroCheetah  → state: quadratic drag Δ = -k*v*|v|
  IceWalker    → state: friction drop at x>5; termination: softer angle threshold
  WindHopper   → state: sinusoidal side force; termination: wind-induced falls
  CarpetAnt    → state: damped contacts; reward: motor current penalty; termination: soft falls
"""
