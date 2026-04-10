"""
MC-WM: Meta-Cognitive World Model with Full-Tuple Residual Dynamics

A self-auditing sim-to-real transfer agent that:
  1. Extracts full-tuple residuals (Δs, Δr, Δd)
  2. Discovers their symbolic structure via SINDy
  3. Enforces OOD extrapolation via NAU/NMU (CS-BAPR guarantees)
  4. Gates corrections by uncertainty
  5. Iteratively falsifies its own hypotheses about the sim-real gap
"""
