"""Command-line demo for the server-side scripted blue team."""

from __future__ import annotations

import argparse

import gymnasium as gym

from . import TEN_UNIT_ENV_ID
from .env import SnowGymEnv
from .recording import ReplayRecorder, write_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scripted SnowGym blue team")
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--blue-units", type=int, default=3)
    parser.add_argument("--red-units", type=int, default=3)
    parser.add_argument("--arena-width", type=float, default=40.0)
    parser.add_argument("--arena-height", type=float, default=30.0)
    parser.add_argument("--map", help="bundled map id, for example arena6.json")
    parser.add_argument("--max-ticks", type=int, default=60 * 180)
    parser.add_argument("--decision-hz", type=int, default=10)
    parser.add_argument(
        "--red-difficulty", choices=("easy", "normal", "hard"), default="normal"
    )
    parser.add_argument(
        "--red-controller", choices=("scripted", "random"), default="scripted"
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="write detached server observations to a SnowGym replay JSON file",
    )
    args = parser.parse_args()

    environment = gym.make(
        TEN_UNIT_ENV_ID,
        server_url=args.server,
        timeout=30.0,
        blue_units=args.blue_units,
        red_units=args.red_units,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        max_ticks=args.max_ticks,
        decision_hz=args.decision_hz,
        red_difficulty=args.red_difficulty,
        red_controller=args.red_controller,
        map=args.map,
    ).unwrapped
    if not isinstance(environment, SnowGymEnv):
        raise TypeError(f"{TEN_UNIT_ENV_ID} resolved to an unexpected environment")

    _, status = environment.reset(seed=args.seed)
    raw = environment.raw_observation
    if raw is None:
        raise RuntimeError("server reset did not provide a raw observation")
    recorder = ReplayRecorder(raw, status) if args.record else None
    decisions = 0
    terminated = False
    truncated = False
    try:
        while decisions < args.max_decisions and not (terminated or truncated):
            _, _, terminated, truncated, status = environment.step_scripted()
            decisions += 1
            raw = environment.raw_observation
            if recorder is not None and raw is not None:
                recorder.append(raw, status)
    finally:
        environment.close()

    print("SnowGym blue-team demo")
    print(f"  API:       {status['apiVersion']}")
    print(f"  seed:      {status['seed']}")
    configuration = status["configuration"]
    print(
        f"  matchup:   {configuration['blueUnits']} blue vs "
        f"{configuration['redUnits']} red"
    )
    print(f"  map:       {configuration.get('map') or 'open'}")
    print(f"  decisions: {decisions}")
    print(f"  ticks:     {status['tick']}")
    print(f"  survivors: blue={status['blueAlive']} red={status['redAlive']}")
    print(f"  winner:    {status['winner']}")
    if recorder is not None:
        destination = write_replay(args.record, recorder.finish(decisions))
        print(f"  recording: {destination}")
    if not terminated and not truncated:
        raise SystemExit("demo stopped before the episode completed")


if __name__ == "__main__":
    main()
