"""Export a Gymnasium CarRacing scene for the Rust renderer's side-by-side eyeball check.

Writes fixture.json (track points, body poses, wheel state, tick) plus gym_frame.png.
Requires `pip install "gymnasium[box2d]"`; dev-only, not wired into CI.

    python scripts/car_racing_fixture.py --seed 5 --ticks 60 --out /tmp/carracing_fixture
"""

import argparse
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    import gymnasium as gym
    import numpy as np
    from PIL import Image

    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    executed = 0
    for _ in range(args.ticks):
        obs, _, terminated, truncated, _ = env.step(3)
        executed += 1
        if terminated or truncated:
            break

    inner = env.unwrapped
    fixture = {
        "tick": executed,
        "tile_visited_count": inner.tile_visited_count,
        "track": [[beta, x, y] for (_alpha, beta, x, y) in inner.track],
        "hull": {
            "pos": list(inner.car.hull.position),
            "angle": inner.car.hull.angle,
            "angvel": inner.car.hull.angularVelocity,
            "linvel": list(inner.car.hull.linearVelocity),
        },
        "wheels": [
            {
                "pos": list(w.position),
                "angle": w.angle,
                "omega": w.omega,
                "phase": w.phase,
                "joint_angle": w.joint.angle,
            }
            for w in inner.car.wheels
        ],
        "reward": inner.reward,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fixture.json").write_text(json.dumps(fixture))
    Image.fromarray(np.asarray(obs)).save(args.out / "gym_frame.png")
    print(f"wrote {args.out}/fixture.json and gym_frame.png")


if __name__ == "__main__":
    main()
