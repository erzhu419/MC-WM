"""
Standalone smoke test for Claude CLI integration.

Run OUTSIDE an active Claude Code session (e.g., regular terminal).  The
`claude` CLI uses the user's subscription plan — no API key required.
Caches responses to /tmp/mcwm_claude_cache/ so repeated runs are free.

Usage:
    conda run -n MC-WM python3 experiments/llm_smoke_test.py
    conda run -n MC-WM python3 experiments/llm_smoke_test.py --role 2 --env gravity_cheetah
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mc_wm.self_audit.claude_cli_oracle import ClaudeCLIOracle


GRAVITY_CHEETAH_DESC = (
    "HalfCheetah-v4 with gravity doubled in sim (2x g) vs real (1x g). "
    "obs[0]=rootz, obs[1]=torso_angle, obs[2:8]=6 joint angles, "
    "obs[8]=vx, obs[9]=vz, obs[10]=va_torso, obs[11:17]=joint vels. "
    "act in R^6 controls joint torques."
)

FRICTION_WALKER_DESC = (
    "Walker2d-v4 with ground friction 0.3x in sim, 1x in real. "
    "obs[0]=rootz (~1.2 baseline), obs[1]=torso_angle, obs[2:8]=joint angles, "
    "obs[8:17]=velocities. act in R^6 controls joint torques."
)


def role1(oracle, env_desc):
    print(f"\n=== Role #1: Initial Physical Constraints ===")
    constraints = oracle.role1_initial_constraints(env_desc)
    print(f"Got {len(constraints)} constraints:")
    for c in constraints:
        print(f"  - {c.get('name','?')}: {c.get('check','?')}")
        print(f"    why: {c.get('why','')[:100]}")
    return constraints


def role2(oracle, env_desc):
    print(f"\n=== Role #2: Feature Hypotheses ===")
    feats = oracle.role2_feature_hypotheses(
        env_description=env_desc,
        current_basis=["s[0]", "s[1]", "s[0]*s[1]", "s[1]**2", "a[0]*s[1]"],
        diagnosis_summary=(
            "dim 1 (torso angle): heteroscedastic, kurtosis=22. "
            "dim 2 (joint1): non-normal, kurtosis=10. "
            "Residual appears oscillatory with period ~2π/3."
        ),
        obs_dim=17, act_dim=6,
    )
    print(f"Got {len(feats)} feature suggestions:")
    for f in feats:
        print(f"  - {f.get('name','?')}: {f.get('expr','?')}")
        print(f"    why: {f.get('why','')[:100]}")
    return feats


def role3(oracle, env_desc):
    print(f"\n=== Role #3: Runtime Audit ===")
    result = oracle.role3_audit(
        env_description=env_desc,
        suspicious_transitions=[
            {"s": [0.1, 2.5, 0, 0, 0, 0, 0, 0, 3.5, 12, 0, 0, 0, 0, 0, 0, 0],
             "a": [2, 0, 0, 0, 0, 0], "correction_magnitude": 45.2,
             "note": "vz jumped from 0 to 12 in one step"},
        ],
        current_constraints=[
            "abs(s[1]) < 1.5",
            "abs(s[9]) < 10",
        ],
    )
    print(f"Audit result:\n{json.dumps(result, indent=2)}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", type=int, choices=[1, 2, 3, 0], default=0,
                    help="Which role to test. 0=all.")
    ap.add_argument("--env", choices=["gravity_cheetah", "friction_walker"],
                    default="gravity_cheetah")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Default: Haiku 4.5 (cheapest). Use claude-sonnet-4-6 for harder tasks.")
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()

    desc = GRAVITY_CHEETAH_DESC if args.env == "gravity_cheetah" else FRICTION_WALKER_DESC
    oracle = ClaudeCLIOracle(model=args.model, timeout=args.timeout)

    if args.role in (0, 1): role1(oracle, desc)
    if args.role in (0, 2): role2(oracle, desc)
    if args.role in (0, 3): role3(oracle, desc)

    print(f"\n=== Stats: {oracle.stats()} ===")
    print(f"Cache: /tmp/mcwm_claude_cache/  (delete to force fresh calls)")


if __name__ == "__main__":
    main()
