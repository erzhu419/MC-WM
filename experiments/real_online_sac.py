"""
Pure-online SAC (model-free) on real env. Comparison baseline for H2O+.
Usage: python real_online_sac.py --env gravity_soft_ceiling --seed 42
"""
import sys, os, warnings, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import torch
from mc_wm.policy.resac_agent import RESACAgent


def get_env_cls(name):
    if name == "gravity_soft_ceiling":
        from mc_wm.envs.hp_mujoco.gravity_cheetah import GravityCheetahSoftCeilingEnv
        return GravityCheetahSoftCeilingEnv
    if name == "gravity_ceiling":
        from mc_wm.envs.hp_mujoco.gravity_cheetah import GravityCheetahCeilingEnv
        return GravityCheetahCeilingEnv
    if name == "friction_walker_soft_ceiling":
        from mc_wm.envs.hp_mujoco.friction_walker import FrictionWalkerSoftCeilingEnv
        return FrictionWalkerSoftCeilingEnv
    raise ValueError(f"Unknown env: {name}")


class ReplayBuffer:
    def __init__(self, od, ad, cap, device):
        self.max_size = cap; self.ptr = self.size = 0; self.device = device
        self.s = np.zeros((cap, od), np.float32); self.a = np.zeros((cap, ad), np.float32)
        self.r = np.zeros((cap, 1), np.float32); self.s2 = np.zeros((cap, od), np.float32)
        self.d = np.zeros((cap, 1), np.float32)
    def add(self, s, a, r, s2, d):
        i = self.ptr; self.s[i]=s; self.a[i]=a; self.r[i]=r; self.s2[i]=s2; self.d[i]=d
        self.ptr = (i+1) % self.max_size; self.size = min(self.size+1, self.max_size)
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return tuple(torch.FloatTensor(x[idx]).to(self.device) for x in [self.s, self.a, self.r, self.s2, self.d])


def evaluate(agent, env_cls, n_eps=5):
    env = env_cls(mode="real"); rets = []; viols = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep + 200); total = 0.0; v = 0
        for _ in range(1000):
            a = agent.get_action(obs, deterministic=True)
            obs, r, d, tr, info = env.step(a); total += r
            if info.get("constraint_violated", False) or info.get("violation", 0) > 0:
                v += 1
            if d or tr: break
        rets.append(total); viols.append(v)
    env.close()
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(viols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_steps", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=5000)
    args = p.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"[pure-online-sac/{args.env}/s{args.seed}] Device: {DEVICE}", flush=True)

    env_cls = get_env_cls(args.env)
    env = env_cls(mode="real")
    od = env.observation_space.shape[0]; ad = env.action_space.shape[0]
    al = float(env.action_space.high[0])
    agent = RESACAgent(od, ad, al, hidden_dim=256, n_critics=3, beta=-2.0, lr=3e-4, device=DEVICE)
    buf = ReplayBuffer(od, ad, 200_000, DEVICE)

    obs, _ = env.reset(seed=args.seed)
    print(f"\n[Pure Online SAC on {args.env}] {args.total_steps} steps", flush=True)
    history = []
    for step in range(1, args.total_steps + 1):
        a = env.action_space.sample() if step < args.warmup else agent.get_action(obs, deterministic=False)
        obs2, r, d, tr, _ = env.step(a)
        buf.add(obs, a, r, obs2, float(d and not tr))
        obs = obs2
        if d or tr: obs, _ = env.reset()
        if step >= args.warmup and buf.size >= 256:
            agent.update(buf)
        if step % args.eval_every == 0:
            ret, std, v = evaluate(agent, env_cls)
            print(f"  step {step:>6d} | real={ret:7.1f}±{std:4.0f} viol={v:4.1f}", flush=True)
            history.append((ret, v))
    env.close()

    last3 = history[-3:]
    r3 = float(np.mean([x[0] for x in last3]))
    v3 = float(np.mean([x[1] for x in last3]))
    print(f"pure-online-sac last 3 avg: violations={v3:.2f}/ep", flush=True)
    print("\n" + "=" * 50, flush=True)
    print(f"pure-online-sac last 3 avg: real={r3:.1f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
