"""R1m-S12: movement representability under dense on-policy supervision (reviewer E2).

Three arms share one frozen R1f source, one zero-initialized residual decoder
(`q = c + a*tanh(raw_source + delta)`), one controller-aware heading loss, and
one from-reset DAgger loop; they differ only in how the move head's input
features are built. See `reviews/m7b_r1m_s12_declaration.md`.

- A: `GeometryProbe(relative=False)` -- the current production path, reused unmodified.
- A-rel: `GeometryProbe(relative=True)` -- same class, one constructor flag; a free
  control that isolates "egocentric frame" from "attention".
- B: `AttentionGeometryProbe` -- egocentric pair features (relative position/velocity,
  distance, bearing, closing speed, projectile time-to-closest-approach) pooled by a
  fighter-query attention head instead of order-invariant mean/max.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..executor.geometry_probe import GeometryProbe
from ..executor.model import masked_mean_max, select_action_target
from ..ppo import HybridActorCritic, living_unit_mask
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import numpy_actions, tensor_dict
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .control_channels import recommend_movement
from .identity import checkpoint_model
from .interventions import require_capabilities
from .movement_collect import ASSIST_FIELDS, corrected_shots
from .movement_train import REFERENCE, TRAINING, make_wrapper
from .opportunity_audit import plain
from .plans import teacher_option_plan, teacher_option_scenario
from .reservoir import file_digest
from .supervised_probe import write_json

ARMS = ("A", "A-rel", "B")
GAMMA = .9976921765
SLOW_RADIUS = 2.1  # world units; MovementSystem.ts ARRIVAL_SLOW_RADIUS = PLAYER.spacing * 1.5.
ARENA_HALF_EXTENT = (50., 40.)  # frozen 100x80 Engage arena
VELOCITY_SCALE = 20.  # snowgym_client.encoding.VELOCITY_SCALE


def configuration():
    return {"format": "snowgym.geometry-representation-config.v0", "arms": list(ARMS),
        "checkpointDigest": "sha256:10d924ecdfbc554a8e0324387d8d049b9ffe719e8d8f2768123e4886c265a697",
        "trainingSeeds": [100000, 100063], "developmentSeeds": [200000, 200039],
        "replicationDevelopmentSeeds": [210000, 210039], "rounds": 4, "stepsPerRound": 3000,
        "minibatchSize": 256, "learningRate": 3e-4, "maxGradNorm": .5, "trainingSeed": 95201,
        "slowRadius": SLOW_RADIUS, "endpointClip": 3., "simulatorBudget": 250000,
        "selection": "final-round-only", "bootstrapSeed": 952001, "bootstrapSamples": 10000}


# ---------------------------------------------------------------------------
# Arm B: egocentric pair features with fighter-query attention pooling.
# ---------------------------------------------------------------------------

def attention_pool(query: Tensor, keys: Tensor, mask: Tensor) -> Tensor:
    """Single-head scaled dot-product pooling; an all-masked row pools to zero."""
    scale = keys.shape[-1] ** .5
    scores = torch.einsum("nd,ned->ne", query, keys) / scale
    scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(mask.any(-1, keepdim=True), weights, torch.zeros_like(weights))
    return torch.einsum("ne,ned->nd", weights, keys)


class AttentionGeometryProbe(nn.Module):
    """Same forward()/act() contract as GeometryProbe; only features() differs."""

    def __init__(self, source: HybridActorCritic):
        super().__init__()
        self.source = source.requires_grad_(False).eval()
        self.query = nn.Linear(21, 16)
        self.encoders = nn.ModuleDict({
            "allies": nn.Sequential(nn.Linear(25, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()),
            "enemies": nn.Sequential(nn.Linear(25, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()),
            "projectiles": nn.Sequential(nn.Linear(15, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()),
            "obstacles": nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()),
        })
        features = 21 + 4 * 32 + 38 + 40
        self.move = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 2))
        self.shot = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 3))
        for head in (self.move, self.shot):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def pair_features(self, name: str, own: Tensor, raw: Tensor) -> Tensor:
        units = own.shape[1]
        values = raw[:, None].expand(-1, units, -1, -1).clone()
        if name == "obstacles":
            values[..., 1:3] -= own[..., None, 2:4]
            return values
        origin, velocity_origin = own[..., None, 2:4], own[..., None, 4:6]
        values[..., 2:4] -= origin
        values[..., 4:6] -= velocity_origin
        scale = values.new_tensor(ARENA_HALF_EXTENT)
        world_position = values[..., 2:4] * scale
        world_velocity = values[..., 4:6] * VELOCITY_SCALE
        distance = world_position.norm(dim=-1, keepdim=True)
        direction = world_position / distance.clamp_min(1e-6)
        closing_speed = -(world_position * world_velocity).sum(-1, keepdim=True) / distance.clamp_min(1e-6)
        derived = [distance / 50., direction, closing_speed / VELOCITY_SCALE]
        if name in ("allies", "enemies"):
            present = values[..., 10:11]
            values[..., 11:13] = (values[..., 11:13] - origin) * present
            values[..., 13:15] = (values[..., 13:15] - origin) * present
        else:  # projectiles: no controller targets; add time-to-closest-approach instead.
            speed_sq = (world_velocity * world_velocity).sum(-1, keepdim=True)
            raw_ttca = -(world_position * world_velocity).sum(-1, keepdim=True) / speed_sq.clamp_min(1e-6)
            ttca = torch.where(speed_sq > 1e-6, raw_ttca.clamp(0., 3.), torch.zeros_like(raw_ttca))
            closest_gap = (world_position + world_velocity * ttca).norm(dim=-1, keepdim=True)
            derived += [ttca / 3., closest_gap / 50.]
        return torch.cat([values, *derived], -1)

    def pooled(self, name: str, own: Tensor, query: Tensor, observation: dict[str, Tensor]) -> Tensor:
        batch, units = own.shape[:2]
        values = self.pair_features(name, own, observation[name].float())
        mask_name = {"allies": "ally_mask", "enemies": "enemy_mask",
                     "projectiles": "projectile_mask", "obstacles": "obstacle_mask"}[name]
        mask = observation[mask_name].bool()
        if name in ("allies", "enemies"):
            mask = mask & (observation[name][..., 1] > .5)
        mask = mask[:, None].expand(-1, units, -1).reshape(batch * units, -1)
        encoded = self.encoders[name](values).reshape(batch * units, -1, 16)
        mean, maximum = masked_mean_max(encoded, mask)
        if name == "obstacles":
            return torch.cat([mean, maximum], -1).reshape(batch, units, 32)
        attended = attention_pool(query, encoded, mask)
        return torch.cat([attended, maximum], -1).reshape(batch, units, 32)

    def features(self, observation: dict[str, Tensor]) -> Tensor:
        own = observation["allies"].float()
        batch, units = own.shape[:2]
        query = self.query(own.reshape(batch * units, -1))
        pools = [self.pooled(name, own, query, observation) for name in ("allies", "enemies", "projectiles", "obstacles")]
        roles = observation["plan_unit_roles"].float() * observation["plan_group_mask"][:, None].float()
        groups = observation["plan_groups"].float() * observation["plan_group_mask"][..., None].float()
        directives = torch.einsum("bur,brf->buf", roles, groups)
        state = observation["plan_role_state"].float() * observation["plan_group_mask"][..., None].float()
        role = torch.cat([torch.einsum("bur,brf->buf", roles, state),
                          torch.einsum("bur,brf->buf", directives[..., 34:37], state)], -1)
        return torch.cat([own, *pools, directives, role], -1)

    def forward(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        with torch.no_grad():
            inherited = self.source(observation)
        features = self.features(observation)
        live = living_unit_mask(observation)[..., None]
        move, shot = self.move(features) * live, self.shot(features) * live
        base = inherited["target_raw_by_action"]
        raw = torch.stack([base[..., 0, :], base[..., 1, :] + move,
                           base[..., 2, :] + shot[..., :2], base[..., 3, :]], -2)
        power_raw = inherited["power_raw"] + shot[..., 2]
        selected = inherited["action_logits"].argmax(-1)
        return {**inherited, "target_raw_by_action": raw, "target_by_action": torch.tanh(raw),
                "target_raw": select_action_target(raw, selected), "target": torch.tanh(select_action_target(raw, selected)),
                "power_raw": power_raw, "power": torch.sigmoid(power_raw)}

    def act(self, observation: dict[str, Tensor], *, deterministic: bool = False):
        if not deterministic:
            raise ValueError("attention geometry probe is deterministic-only; no sampled/PPO likelihood contract")
        output = self(observation)
        action = {"action_type": output["action_logits"].argmax(-1), "target": output["target"], "power": output["power"]}
        return action, None, output["value"]


def build_arm(source, arm):
    if arm == "A":
        return GeometryProbe(source, relative=False)
    if arm == "A-rel":
        return GeometryProbe(source, relative=True)
    if arm == "B":
        return AttentionGeometryProbe(source)
    raise ValueError("unknown geometry-representation arm")


def move_mask(prediction, observation):
    return living_unit_mask(observation) & (prediction["action_logits"].argmax(-1) == 1)


def heading_loss(prediction, teacher_target, available, observation, *, slow_radius=SLOW_RADIUS, endpoint_clip=3.):
    moves = move_mask(prediction, observation) & available
    scale = prediction["target_by_action"].new_tensor(ARENA_HALF_EXTENT)
    own_world = observation["allies"][..., 2:4].float() * scale
    learned_world = prediction["target_by_action"][..., 1, :] * scale
    teacher_world = teacher_target.float() * scale
    to_teacher, to_learned = teacher_world - own_world, learned_world - own_world
    distance = to_teacher.norm(dim=-1)
    far, near = moves & (distance > slow_radius), moves & (distance <= slow_radius)
    if far.any():
        cosine = F.cosine_similarity(to_learned[far], to_teacher[far], dim=-1, eps=1e-6).clamp(-1., 1.)
        heading = (1 - cosine).mean()
        heading_degrees = float(torch.rad2deg(torch.arccos(cosine)).mean().detach())
    else:
        heading, heading_degrees = learned_world.sum() * 0, None
    gap = (learned_world - teacher_world).norm(dim=-1).clamp(max=endpoint_clip)
    endpoint = gap[near].square().mean() if near.any() else learned_world.sum() * 0
    return {"total": heading + endpoint, "heading": heading, "endpoint": endpoint,
            "movingUnits": int(moves.sum()), "farUnits": int(far.sum()), "nearUnits": int(near.sum()),
            "headingErrorDegrees": heading_degrees}


# ---------------------------------------------------------------------------
# From-reset DAgger collection and evaluation, shared by round 0's teacher
# trajectory, rounds 1-3's per-arm on-policy trajectories, and evaluation.
# ---------------------------------------------------------------------------

def teacher_forced_action(source, observation, raw):
    with torch.no_grad():
        action, _, _ = source.act(tensor_dict(observation), deterministic=True)
    executed = numpy_actions(action)
    rec = recommend_movement(raw, executed["action_type"].shape[1])
    move = (executed["action_type"][0] == 1) & rec["valid"][0]
    executed["target"][0, move] = rec["target"][0, move]
    return corrected_shots(executed, [raw])


def episode(model, source, wrapper, seed, *, teacher_move, collect):
    plan, spec = teacher_option_plan("engage")
    observation, _ = wrapper.reset([seed], [teacher_option_scenario("engage")],
        [f"geometry-representation-{seed}"], [plan], [spec])
    rows, rejected, total = [], 0, 0
    for _ in range(spec.horizon):
        raw = wrapper.environment.raw_observations[0]
        if collect:
            # The classifier is frozen and shared by every arm, so labeling never needs the
            # (possibly untrained, possibly None during round 0) arm-specific model.
            rec = recommend_movement(raw, wrapper.environment.max_team_units)
            available = torch.as_tensor(rec["valid"][0]) & torch.as_tensor(
                observation["unit_action_mask"][0, :, 1].astype(bool))
            rows.append({"observation": {k: v.clone() for k, v in tensor_dict(observation).items()},
                "moveTarget": torch.as_tensor(rec["target"][0], dtype=torch.float32), "moveAvailable": available})
        if teacher_move:
            executed = teacher_forced_action(source, observation, raw)
        else:
            with torch.no_grad():
                action, _, _ = model.act(tensor_dict(observation), deterministic=True)
            executed = corrected_shots(numpy_actions(action), [raw])
        observation, _, terminated, truncated, infos = wrapper.step(executed)
        results = infos[0].get("actionResults", [])
        rejected += sum(r.get("accepted") is False for r in results); total += len(results)
        if terminated[0] or truncated[0]:
            break
    final = infos[0]["option"]
    summary = {"seed": seed, "success": bool(final["success"]), "progress": float(final["progress"]),
        "decisions": final["decision"], "rejectedActions": rejected, "totalActions": total,
        "simulatorDecisions": final["decision"]}
    return rows, summary


def collect_round(model, source, wrapper, seeds, *, teacher_move):
    rows, summaries = [], []
    for seed in seeds:
        episode_rows, summary = episode(model, source, wrapper, seed, teacher_move=teacher_move, collect=True)
        rows.extend(episode_rows); summaries.append(summary)
    return rows, summaries


class Dataset:
    def __init__(self, rows):
        if not rows:
            raise ValueError("empty DAgger dataset")
        self.observation = {k: torch.stack([r["observation"][k][0] for r in rows]) for k in rows[0]["observation"]}
        self.target = torch.stack([r["moveTarget"] for r in rows])
        self.available = torch.stack([r["moveAvailable"] for r in rows])

    def __len__(self):
        return len(self.target)

    def batch(self, indices):
        return ({k: v[indices] for k, v in self.observation.items()}, self.target[indices], self.available[indices])


def fit_round(model, optimizer, dataset, cfg, *, seed):
    generator = torch.Generator().manual_seed(seed)
    traces = []
    size = len(dataset)
    step = 0
    while step < cfg["stepsPerRound"]:
        order = torch.randperm(size, generator=generator)
        for start in range(0, size, cfg["minibatchSize"]):
            if step >= cfg["stepsPerRound"]:
                break
            indices = order[start:start + cfg["minibatchSize"]]
            observation, target, available = dataset.batch(indices)
            prediction = model(observation)
            losses = heading_loss(prediction, target, available, observation,
                slow_radius=cfg["slowRadius"], endpoint_clip=cfg["endpointClip"])
            if not torch.isfinite(losses["total"]):
                raise ValueError("non-finite geometry-representation loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["maxGradNorm"], error_if_nonfinite=True)
            optimizer.step()
            traces.append({"step": step, "total": float(losses["total"].detach()), "heading": float(losses["heading"].detach()),
                "endpoint": float(losses["endpoint"].detach()), "headingErrorDegrees": losses["headingErrorDegrees"],
                "gradientNorm": float(norm)})
            step += 1
    return traces


def evaluate_arm(model, source, wrapper, seeds):
    return [episode(model, source, wrapper, seed, teacher_move=False, collect=False)[1] for seed in seeds]


def teacher_ceiling(source, wrapper, seeds):
    return [episode(None, source, wrapper, seed, teacher_move=True, collect=False)[1] for seed in seeds]


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    cfg = configuration()
    metadata, state = load_ppo_checkpoint(REFERENCE)
    if metadata["checkpointDigest"] != cfg["checkpointDigest"]:
        raise ValueError("R1m must use the historical R1h source")
    source = checkpoint_model(metadata)
    source.load_state_dict(state["model"])
    source.eval().requires_grad_(False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    before = semantic_state_digest(source.state_dict())
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "source": metadata,
        "implementationDigest": file_digest(Path(__file__)),
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1m_s12_declaration.md"),
        "gitCommit": resolve_git_commit(), **ASSIST_FIELDS})
    training_seeds = list(range(cfg["trainingSeeds"][0], cfg["trainingSeeds"][1] + 1))
    development_seeds = list(range(cfg["developmentSeeds"][0], cfg["developmentSeeds"][1] + 1))
    replication_seeds = list(range(cfg["replicationDevelopmentSeeds"][0], cfg["replicationDevelopmentSeeds"][1] + 1))
    steps = 0
    def account(rows):
        nonlocal steps
        steps += sum(x["simulatorDecisions"] for x in rows)
        if steps > cfg["simulatorBudget"]:
            raise ValueError("geometry-representation simulator budget exceeded")
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        wrapper = make_wrapper(client, 1, GAMMA)
        ceiling = teacher_ceiling(source, wrapper, development_seeds)
        floor_models = {arm: build_arm(copy.deepcopy(source), arm) for arm in ARMS}
        floor = {arm: evaluate_arm(floor_models[arm], source, wrapper, development_seeds) for arm in ARMS}
        account(ceiling); [account(rows) for rows in floor.values()]
        write_json(root / "ceiling.json", ceiling)
        write_json(root / "floor.json", {arm: rows for arm, rows in floor.items()})
        shared_rows, shared_summaries = collect_round(None, source, wrapper, training_seeds, teacher_move=True)
        account(shared_summaries)
        # Only the trajectory summary is archived; collected observation tensors are large,
        # intermediate, and exactly reproducible from this same deterministic teacher rollout.
        write_json(root / "round-000-shared-summary.json", plain(shared_summaries))
        models = {arm: build_arm(copy.deepcopy(source), arm) for arm in ARMS}
        optimizers = {arm: torch.optim.Adam([p for p in models[arm].parameters() if p.requires_grad],
            lr=cfg["learningRate"]) for arm in ARMS}
        history, evaluations = {arm: [] for arm in ARMS}, {}
        for arm in ARMS:
            rows = list(shared_rows)
            for round_index in range(1, cfg["rounds"]):
                traces = fit_round(models[arm], optimizers[arm], Dataset(rows), cfg,
                    seed=cfg["trainingSeed"] + round_index)
                history[arm].append({"round": round_index, "datasetSize": len(rows), "steps": traces})
                new_rows, new_summaries = collect_round(models[arm], source, wrapper, training_seeds, teacher_move=False)
                account(new_summaries)
                rows = rows + new_rows
                print(json.dumps({"arm": arm, "round": round_index, "datasetSize": len(rows),
                    "stochasticSuccesses": sum(s["success"] for s in new_summaries)}), flush=True)
            final_traces = fit_round(models[arm], optimizers[arm], Dataset(rows), cfg,
                seed=cfg["trainingSeed"] + cfg["rounds"])
            history[arm].append({"round": cfg["rounds"], "datasetSize": len(rows), "steps": final_traces})
            evaluations[arm] = {"training": evaluate_arm(models[arm], source, wrapper, training_seeds),
                "development": evaluate_arm(models[arm], source, wrapper, development_seeds),
                "replication": evaluate_arm(models[arm], source, wrapper, replication_seeds)}
            for split, rows_eval in evaluations[arm].items():
                account(rows_eval)
            write_json(root / f"evaluation-{arm}.json", evaluations[arm])
            print(json.dumps({"arm": arm, "final": True,
                "developmentSuccesses": sum(x["success"] for x in evaluations[arm]["development"])}), flush=True)
    if semantic_state_digest(source.state_dict()) != before:
        raise ValueError("geometry-representation source changed")
    report = {"format": "snowgym.geometry-representation-report.v0", **ASSIST_FIELDS,
        "ceilingSuccesses": sum(x["success"] for x in ceiling),
        "floorSuccesses": {arm: sum(x["success"] for x in rows) for arm, rows in floor.items()},
        "developmentSuccesses": {arm: sum(x["success"] for x in evaluations[arm]["development"]) for arm in ARMS},
        "replicationSuccesses": {arm: sum(x["success"] for x in evaluations[arm]["replication"]) for arm in ARMS},
        "trainingSuccesses": {arm: sum(x["success"] for x in evaluations[arm]["training"]) for arm in ARMS},
        "simulatorDecisions": steps}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.geometry-representation-manifest.v0", **ASSIST_FIELDS,
        "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
