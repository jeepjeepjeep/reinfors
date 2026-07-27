"""Absolute vs mover-relative chess encoding, decided by measurement.

Two AlphaZero training runs at a fixed wall-clock budget — identical net, knobs, and seed;
only the encoder differs (`minimal` = absolute frame, `relative` = mover's frame with the
matching action view) — then a head-to-head between the trained nets. The relative encoder's
predicted edges: role equivariance (every sample teaches both colors) and infer-cache merging
of color-mirrored positions (higher hit rate at equal budget).

    caffeinate -dis python scripts/compare_chess_encoders.py train --encoder minimal --minutes 45 --out runs/m
    caffeinate -dis python scripts/compare_chess_encoders.py train --encoder relative --minutes 45 --out runs/r
    .venv/bin/python scripts/compare_chess_encoders.py h2h --a runs/chess_min --b runs/chess_rel --games 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import reinfors as rf
import torch
from torch import nn
from torch.nn import functional as F

OBS = (19, 8, 8)
A = 4672


class ChessAzNet(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        c, h, w = OBS
        self.trunk = nn.Sequential(
            nn.Conv2d(c, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width * h * w, 4 * width),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(4 * width, A)
        self.value_head = nn.Linear(4 * width, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        return self.policy_head(z), torch.tanh(self.value_head(z)).squeeze(-1)


def encoder_of(name: str) -> object:
    return {"minimal": rf.encoders.MinimalChess, "relative": rf.encoders.RelativeChess}[name]()


def make_infer(net: ChessAzNet):
    def infer(obs_batch: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_batch)).reshape(-1, *OBS)
            logits, values = net(x)
        return logits.double().numpy(), values.double().numpy()

    return infer


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = rf.Engine(
        rf.games.Chess(max_ticks=args.max_ticks, encoder=encoder_of(args.encoder)),
        rf.Reward(),  # game defaults: win/loss/draw = 1/-1/0
        rf.policies.AlphaZero(num_simulations=args.sims, c_puct=2.0, temperature=1.0, temperature_drop=16),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=args.n_games,
        seed=args.seed,
        infer_cache=args.infer_cache,
    )
    net = ChessAzNet(args.width)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    infer = make_infer(net)
    t0, it = time.perf_counter(), 0
    log = (out / "log.jsonl").open("w")
    while time.perf_counter() - t0 < args.minutes * 60:
        it += 1
        batch = engine.collect(args.collect_size, infer)
        engine.weights_updated()
        obs = torch.from_numpy(batch.obs).reshape(-1, *OBS)
        pi = torch.from_numpy(batch.policy_targets).float()
        z = torch.from_numpy(batch.value_targets).float()
        order = torch.from_numpy(rng.permutation(obs.shape[0]))
        ploss = vloss = nb = 0.0
        for s in range(0, obs.shape[0], args.batch_size):
            idx = order[s : s + args.batch_size]
            logits, values = net(obs[idx])
            pl = -(pi[idx] * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            vl = F.mse_loss(values, z[idx])
            opt.zero_grad()
            (pl + vl).backward()
            opt.step()
            ploss += float(pl.item())
            vloss += float(vl.item())
            nb += 1
        t = batch.telemetry
        eps = t["episodes"]
        row = {
            "it": it,
            "wall": round(time.perf_counter() - t0, 1),
            "records": int(batch.obs.shape[0]),
            "policy_loss": round(ploss / nb, 4),
            "value_loss": round(vloss / nb, 4),
            "ep_len": round(float(np.mean([n for _, n, _ in eps])), 1) if eps else None,
            "fresh_rows": t.get("fresh_rows"),
            "hit_rows": t.get("hit_rows"),
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
    torch.save({"net": net.state_dict(), "encoder": args.encoder, "width": args.width}, out / "final.pt")
    print(f"saved {out}/final.pt after {it} iters", flush=True)


def load(run: str) -> tuple[ChessAzNet, str, object]:
    ck = torch.load(Path(run) / "final.pt", weights_only=True)
    net = ChessAzNet(ck["width"])
    net.load_state_dict(ck["net"])
    net.eval()
    return net, ck["encoder"], encoder_of(ck["encoder"])


def h2h(args: argparse.Namespace) -> None:
    """Lockstep envs (chess is deterministic): each net observes through ITS OWN encoder and
    reads logits through ITS OWN action map; moves are exchanged as game-frame ids."""
    nets = [load(args.a), load(args.b)]
    rng = np.random.default_rng(0)
    score = [0.0, 0.0]
    for g in range(args.games):
        white = g % 2  # alternate colors; index into `nets`
        envs = [
            rf.Env(rf.games.Chess(max_ticks=args.max_ticks, encoder=encoder_of(nets[i][1])), rf.Reward(), seed=g)
            for i in range(2)
        ]
        for e in envs:
            e.reset()
        ply = 0
        while not envs[0].done() and ply < args.max_ticks:
            mover = envs[0].active_agents()[0]
            side = white if mover == 0 else 1 - white
            net, _, enc = nets[side]
            legal = envs[side].legal_actions(mover)
            with torch.no_grad():
                x = torch.from_numpy(envs[side].observe(mover)).reshape(1, *OBS)
                logits, _ = net(x)
            lg = np.array([float(logits[0, enc.head_index(a, mover)]) for a in legal])
            if ply < 10:  # opening diversity
                p = np.exp(lg - lg.max())
                a = legal[int(rng.choice(len(legal), p=p / p.sum()))]
            else:
                a = legal[int(lg.argmax())]
            for e in envs:
                e.step({mover: a})
            ply += 1
        r = envs[0].rewards
        if envs[0].done() and r is not None:
            score[white] += (r[0] + 1) / 2
            score[1 - white] += (r[1] + 1) / 2
        else:
            score[0] += 0.5
            score[1] += 0.5
        print(f"game {g + 1}/{args.games}: score A={score[0]:.1f} B={score[1]:.1f}", flush=True)
    n = args.games
    print(f"FINAL  A({args.a})={score[0]}/{n}  B({args.b})={score[1]}/{n}  B-score={score[1] / n:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--encoder", choices=["minimal", "relative"], required=True)
    tr.add_argument("--minutes", type=float, default=45.0)
    tr.add_argument("--out", required=True)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--sims", type=int, default=48)
    tr.add_argument("--n-games", type=int, default=8)
    tr.add_argument("--collect-size", type=int, default=768)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--width", type=int, default=48)
    tr.add_argument("--max-ticks", type=int, default=160)
    tr.add_argument("--infer-cache", type=int, default=262144)
    hh = sub.add_parser("h2h")
    hh.add_argument("--a", required=True)
    hh.add_argument("--b", required=True)
    hh.add_argument("--games", type=int, default=40)
    hh.add_argument("--max-ticks", type=int, default=160)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else h2h(args)


if __name__ == "__main__":
    main()
