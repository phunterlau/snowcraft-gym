"""R1n-i: frozen-checkpoint failure diagnosis for R1n-h's mixture condition.

No policy is trained, fine-tuned, or newly evaluated for success/death here. This
reproduces R1n-h's own archived `eval-normal` episodes, on R1n-h's own seeds, for all
six of its checkpoints, then recovers the per-decision observation/label data a
`record=True` collection call already computes but `mixture_imitation.train_condition`
discards after computing `label_error`'s aggregate. From that data it derives a
model-conditioned, ground-truth-relative view of the model's own executed actions,
targeting the two distinct failure modes an erratum to `m7b_r1n_h_results.md`
identified: 97101 rarely makes contact; 97103 makes contact but dies finishing.
See `reviews/m7b_r1n_i_declaration.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW
from ..executor.full_authority_ppo import ARENA_HALF_EXTENT
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import full_authority_imitation as fi
from . import full_authority_train_v1 as v1
from . import mixture_imitation as mi
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged, write_episodes
from .opponent_transfer import scenario_override
from .reservoir import file_digest
from .supervised_probe import write_json

CHECKPOINTS = (("M", 97101), ("M", 97102), ("M", 97103), ("C", 97101), ("C", 97102), ("C", 97103))
ENGAGE_RANGE = 9.0  # SimpleBlueAgent.ts's own effective throw range; the same reference R1n-f used.


def configuration():
    return {**mi.configuration(), "format": "snowgym.r1n-i-checkpoint-failure-config.v0",
        "sourceRun": "m7b_engage_r1n_h_v0", "checkpoints": CHECKPOINTS, "engageRange": ENGAGE_RANGE,
        "collectionBlockWorlds": 50, "budgetCap": 200000,
        "assistType": "diagnostic read of frozen checkpoints; none at runtime",
        "assistVersion": "snowgym.checkpoint-failure-diagnostic.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    per_checkpoint = cfg["evaluationEpisodes"] * cfg["optionHorizon"]
    total = len(cfg["checkpoints"]) * per_checkpoint
    return {"perCheckpoint": per_checkpoint, "checkpoints": len(cfg["checkpoints"]), "total": total}


# -- Loading and archive reference (declaration §1) -----------------------------------


def load_checkpoint(cfg, condition, seed):
    model = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
        target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])
    path = TRAINING / "runs" / cfg["sourceRun"] / f"condition-{condition}" / f"seed-{seed}" / "fit-4.pt"
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model.eval().requires_grad_(False)


def archived_episodes_path(cfg, condition, seed):
    return (TRAINING / "runs" / cfg["sourceRun"] / f"condition-{condition}" / f"seed-{seed}"
            / "eval-normal-deterministic" / "episodes.jsonl")


def archived_label_error(cfg, condition, seed):
    path = TRAINING / "runs" / cfg["sourceRun"] / f"condition-{condition}" / f"seed-{seed}" / "label-error.json"
    return json.loads(path.read_text(encoding="utf-8"))


# -- Collection with per-decision attribution (declaration §2, amendment A1) -----------


def collect_with_attribution(client, seeds, cfg, *, model, source, block_worlds, account):
    """Like `fi.collect(..., record=True)` — reusing the same `Labeler`/`run_block`
    primitives, unchanged — but additionally retains each stored row's world seed and
    within-episode decision index via `v1.flatten_block`/`v1.concatenate` (imported
    unchanged; the same pair `mixture_imitation.collect_fold_mixture` already reuses for
    an analogous purpose). `fi.collect` computes this attribution internally (`run_block`
    already returns it in `stored`) but discards it when building `part`. Verified
    equivalent to `fi.collect`'s own `part["observation"]`/`part["labels"]` for the same
    seeds and model (test_checkpoint_failure_diagnostic.py) before being trusted for
    anything new — see declaration amendment A1."""
    episodes, parts, label_parts = [], [], []
    for start in range(0, len(seeds), block_worlds):
        block = seeds[start:start + block_worlds]
        labeler = fi.Labeler(model, deterministic=True, record=True)
        wrapper = v1.make_wrapper(client, len(block), cfg["gamma"])
        found, stored, used = v1.run_block(wrapper, block, cfg, choose=labeler, source=source, keep_observations=True)
        account(used)
        if len(stored) != len(labeler.labels):
            raise RuntimeError("teacher labels are misaligned with stored rows")
        parts.append(v1.flatten_block(found, stored, cfg["gamma"], episode_offset=len(episodes)))
        label_parts.append({key: torch.cat([entry[key] for entry in labeler.labels]) for key in fi.LABEL_KEYS})
        episodes.extend(found)
    merged = v1.concatenate(parts)
    labels = {key: torch.cat([p[key] for p in label_parts]) for key in fi.LABEL_KEYS}
    return episodes, {"observation": merged["observation"], "labels": labels,
                       "seed": merged["seed"], "decision": merged["decision"]}


def reproduction_gate(regenerated_episodes, archived_path):
    """Mandatory before any Step B number is interpreted (declaration §2): every regenerated
    world's success/death/timeout must match the archive exactly, in `dr.outcomes`'s own
    three-field shape. Checks only the worlds actually regenerated (a test can run on a
    strict subset of the archived 422000-422099 range without the missing worlds reading as
    mismatches); a real collection regenerates all 100 and this reduces to a full check."""
    rows = {int(v1.episode_row(e)["seed"]): v1.episode_row(e) for e in regenerated_episodes}
    regenerated = {seed: {"success": float(row["success"]), "death": float(not row["blueAliveAtEnd"]),
                          "timeout": float(row["timedOut"])} for seed, row in rows.items()}
    archived = dr.outcomes(archived_path)
    tested = sorted(regenerated)
    missing = [seed for seed in tested if seed not in archived]
    mismatches = [seed for seed in tested if seed in archived and regenerated[seed] != archived[seed]]
    return {"testedWorlds": len(tested), "matchedWorlds": len(tested) - len(mismatches) - len(missing),
            "mismatchedSeeds": mismatches, "missingFromArchive": missing,
            "passed": not mismatches and not missing}


# -- Deployed-action view: model-conditioned, ground-truth-relative (declaration §2 Step B) --


def deployed_view(model, part, cfg, chunk=2048):
    """Recomputes the model's own action and enemy-position-relative geometry from the
    exact tensors the live policy acted on. `run_block`'s per-step `rows` dict is passed
    unmodified to both `choose(...)` (the live action) and `stored.append(...)`
    (`full_authority_train_v1.py:176/183/186`), and `part["observation"]` is built from
    `stored` — so this is a recomputation on the identical input, not an approximation
    (confirmed by the reproduction gate on the episode outcomes it produces). Chunked over
    decisions the same way `label_error` chunks (`chunk=2048`) — R1n-h's notes recorded
    ~3.6 GB peaks on full-rollout forward passes over comparably-sized row counts, and the
    host kills low-memory background tasks."""
    obs, labels = part["observation"], part["labels"]
    scale = obs["allies"].new_tensor(ARENA_HALF_EXTENT)
    size = len(labels["action_type"])
    fields = ("live", "model_type", "teacher_type", "distance_to_enemy", "enemy_seen", "in_range",
              "model_throws", "close_range_throw", "deployed_aim_error", "red_present", "red_nearby")
    chunks = {field: [] for field in fields}
    with torch.no_grad():
        for start in range(0, size, chunk):
            o = {key: value[start:start + chunk] for key, value in obs.items()}
            lab_type = labels["action_type"][start:start + chunk].long()
            prediction = model(o, with_value=False)
            live = prediction["living"]
            model_type = prediction["action_logits"].argmax(-1)

            own = o["allies"][..., 2:4].float() * scale
            # `enemy_mask` (padding) AND the alive flag (feature index 1) — `features()`
            # composes both (masked by `mask & (observation[name][...,1] > .5)` where
            # `mask = observation[mask_name]`); omitting `enemy_mask` would let a nonzero
            # zero-initialized padding row through undetected.
            enemy_present = (o["enemy_mask"].bool() & (o["enemies"][..., 1] > .5)).float()
            enemy_seen = enemy_present.sum(dim=-1) > .5
            enemy_count = enemy_present.sum(dim=-1)
            if bool((enemy_count[enemy_seen] > 1.5).any()):
                raise RuntimeError("more than one live enemy in a 1v1 decision — "
                                    "enemy_pos would silently sum positions")
            enemy_pos = (o["enemies"][..., 2:4].float() * enemy_present[..., None]).sum(dim=-2) * scale
            distance_to_enemy = (enemy_pos[:, None, :] - own).norm(dim=-1)

            model_throws = live & (model_type == ACTION_THROW) & enemy_seen[:, None]
            aim_point = torch.tanh(prediction["throw_raw"]) * scale
            aim_vector, target_vector = aim_point - own, enemy_pos[:, None, :] - own
            aim_cos = F.cosine_similarity(aim_vector, target_vector, dim=-1, eps=1e-6).clamp(-1, 1)
            deployed_aim_error = torch.rad2deg(torch.acos(aim_cos))
            deployed_aim_error = deployed_aim_error.where(enemy_seen[:, None].expand_as(deployed_aim_error),
                                                           torch.full_like(deployed_aim_error, float("nan")))

            in_range = distance_to_enemy <= cfg["engageRange"]
            close_range_throw = model_throws & in_range

            red_projectile = o["projectile_mask"].bool() & (o["projectiles"][..., 1] > 0)
            red_present = red_projectile.any(dim=-1)
            # A weaker, honestly-labelled proxy for "somewhere in this world" than for "a
            # direct threat to this unit": no projectile trajectory/impact check, just
            # live-in-world.
            projectile_pos = o["projectiles"][..., 2:4].float() * scale
            projectile_distance = (projectile_pos[:, :, None, :] - own[:, None, :, :]).norm(dim=-1)
            red_nearby = (red_projectile[:, :, None] & (projectile_distance <= cfg["engageRange"])).any(dim=1)

            for field, value in zip(fields, (live, model_type, lab_type, distance_to_enemy, enemy_seen, in_range,
                                              model_throws, close_range_throw, deployed_aim_error, red_present,
                                              red_nearby)):
                chunks[field].append(value)
    result = {field: torch.cat(values) for field, values in chunks.items()}
    result["seed"], result["decision"] = part["seed"], part["decision"]
    return result


def label_error_recheck(model, part, cfg, archived):
    """Confirms the retained `part`'s pipeline reproduces the already-archived aggregate
    (declaration §10) before any new, ground-truth-conditioned number is trusted."""
    fresh = fi.label_error(model, part, cfg)

    def delta(key):
        a, f = archived.get(key), fresh.get(key)
        return None if a is None or f is None else abs(a - f)

    scalar_keys = ("typeAccuracy", "moveHeadingErrorDegrees", "moveEndpointErrorWorld",
                   "throwAimHeadingErrorDegrees", "powerMeanAbsoluteError")
    return {"fresh": fresh, "deltas": {key: delta(key) for key in scalar_keys},
            "maxDelta": max((d for d in (delta(key) for key in scalar_keys) if d is not None), default=None)}


# -- Contact-failure diagnostic (declaration §3, targets 97101-style outcomes) ---------


def contact_failure_summary(view):
    """Pooled rates alone cannot separate "never reaches range" from "reaches range but
    doesn't throw" (the two halves of declaration §4's first two predictions) if the pooled
    denominator is tiny for one checkpoint and large for another — the per-episode counts
    below make a small denominator visible instead of silently averaging over it."""
    in_range_live = view["in_range"] & view["live"]
    close_throw = view["close_range_throw"] & view["live"]
    thrown = view["model_throws"] & view["live"]
    n_in_range, n_thrown = int(in_range_live.sum()), int(thrown.sum())
    aim = view["deployed_aim_error"][thrown]

    seed = view["seed"]
    decision_in_range = in_range_live.any(dim=-1)
    decision_close_throw = close_throw.any(dim=-1)
    unique_seeds = [int(s) for s in seed.unique()]
    per_episode_in_range = {s: int((decision_in_range & (seed == s)).sum()) for s in unique_seeds}
    episodes_ever_in_range = sum(1 for c in per_episode_in_range.values() if c > 0)
    episodes_with_close_throw = sum(1 for s in unique_seeds if int((decision_close_throw & (seed == s)).sum()) > 0)
    counts = sorted(per_episode_in_range.values())
    return {"decisionsInRange": n_in_range,
            "closeRangeThrowRate": (int(close_throw.sum()) / n_in_range) if n_in_range else None,
            "deployedThrowCount": n_thrown,
            "deployedAimErrorDegrees": float(aim.mean()) if n_thrown else None,
            "episodes": len(unique_seeds), "episodesEverInRange": episodes_ever_in_range,
            "episodesWithCloseRangeThrow": episodes_with_close_throw,
            "perEpisodeInRangeDecisionCounts": {"min": counts[0] if counts else None,
                "median": counts[len(counts) // 2] if counts else None,
                "max": counts[-1] if counts else None}}


# -- Finishing-failure diagnostic (declaration §3, targets 97103-style outcomes) -------


def episode_windows(episodes):
    """seed -> (firstHitDecision, finalDecision), in the same 0-indexed decision numbering
    `part["decision"]` uses. `v1.episode_row`'s `firstHitDecision` is 1-indexed (`first + 1`
    over the 0-indexed `targetDamage` list), hence the `- 1` here."""
    windows = {}
    for episode in episodes:
        row = v1.episode_row(episode)
        if row["firstHitDecision"] is not None:
            windows[int(row["seed"])] = (row["firstHitDecision"] - 1, row["finalDecision"])
    return windows


def finishing_failure_summary(view, episodes):
    """Two honestly-distinguished threat proxies, neither a trajectory/impact check:
    "anywhere" is any live red projectile in the world; "nearby" additionally requires one
    within `engageRange` of the acting unit (`deployed_view`'s `red_nearby`). Reporting only
    the world-level version under an unqualified "under threat" label would overstate what
    was measured."""
    windows = episode_windows(episodes)
    seed, decision = view["seed"], view["decision"]
    in_window = torch.zeros_like(seed, dtype=torch.bool)
    for world_seed, (start, end) in windows.items():
        in_window |= (seed == world_seed) & (decision >= start) & (decision <= end)
    live_window = view["live"] & in_window[:, None]
    post_contact_live = live_window.any(dim=-1)
    moving = ((view["model_type"] == ACTION_MOVE) & live_window).any(dim=-1)

    threatened_anywhere = view["red_present"] & post_contact_live
    threatened_nearby = (view["red_nearby"] & live_window).any(dim=-1) & post_contact_live
    n_anywhere, n_nearby = int(threatened_anywhere.sum()), int(threatened_nearby.sum())
    return {"episodesWithContact": len(windows), "postContactDecisions": int(post_contact_live.sum()),
            "underThreatAnywhereDecisions": n_anywhere, "underThreatNearbyDecisions": n_nearby,
            "keepsMovingUnderThreatAnywhereRate": (int((moving & threatened_anywhere).sum()) / n_anywhere)
                if n_anywhere else None,
            "keepsMovingUnderThreatNearbyRate": (int((moving & threatened_nearby).sum()) / n_nearby)
                if n_nearby else None}


# -- Per-checkpoint orchestration (declaration §1, §7) ---------------------------------


def run_checkpoint(root, cfg, client, condition, seed, account):
    directory = Path(root) / f"condition-{condition}-seed-{seed}"
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    model = load_checkpoint(cfg, condition, seed)
    with scenario_override(mi.EVAL_ARMS["normal"]):
        episodes, part = collect_with_attribution(client, mi.eval_seeds(cfg, "normal"), cfg, model=model,
            source=f"r1n-i-{condition}-{seed}-normal", block_worlds=cfg["collectionBlockWorlds"], account=account)
    write_episodes(directory / "eval-normal-deterministic", episodes)
    gate = reproduction_gate(episodes, archived_episodes_path(cfg, condition, seed))
    view = deployed_view(model, part, cfg)
    archived_errors = archived_label_error(cfg, condition, seed)
    recheck = label_error_recheck(model, part, cfg, archived_errors)
    report = {"condition": condition, "seed": seed, "reproductionGate": gate,
        "labelErrorRecheck": recheck, "contactFailure": contact_failure_summary(view),
        "finishingFailure": finishing_failure_summary(view, episodes)}
    write_json(directory / "checkpoint-report.json", report)
    return report


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    here = Path(__file__).resolve()
    imitation_path = here.parent / "full_authority_imitation.py"
    train_path = here.parent / "full_authority_train_v1.py"
    mixture_path = here.parent / "mixture_imitation.py"
    imitation_digest, train_digest, mixture_digest = (file_digest(imitation_path), file_digest(train_path),
                                                        file_digest(mixture_path))
    h_declaration = json.loads((TRAINING / "runs" / cfg["sourceRun"] / "declaration.json")
                                .read_text(encoding="utf-8"))
    if imitation_digest != h_declaration["imitationImplementationDigest"]:
        raise RuntimeError("full_authority_imitation.py digest no longer matches R1n-h's declaration.json")
    if train_digest != h_declaration["trainImplementationDigest"]:
        raise RuntimeError("full_authority_train_v1.py digest no longer matches R1n-h's declaration.json")
    if mixture_digest != h_declaration["implementationDigest"]:
        raise RuntimeError("mixture_imitation.py digest no longer matches R1n-h's declaration.json")
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg),
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_i_declaration.md"),
        "implementationDigest": file_digest(here),
        "imitationImplementationDigest": imitation_digest, "trainImplementationDigest": train_digest,
        "mixtureImplementationDigest": mixture_digest, "pinnedE3Digests": pinned,
        "sourceRunManifest": dr.verify_sealed(TRAINING / "runs" / cfg["sourceRun"]),
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    reports = {f"{condition}-{seed}": json.loads((root / f"condition-{condition}-seed-{seed}"
               / "checkpoint-report.json").read_text(encoding="utf-8")) for condition, seed in cfg["checkpoints"]}
    gates_passed = all(r["reproductionGate"]["passed"] for r in reports.values())
    report = {"format": "snowgym.r1n-i-checkpoint-failure-report.v0",
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "checkpoints": reports, "allReproductionGatesPassed": gates_passed}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.r1n-i-manifest.v0")
    return report


def run(output):
    root = Path(output)
    cfg = configuration()
    declare(root, cfg)
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["budgetCap"]:
            raise ValueError("R1n-i budget exceeded")

    with SnowGymBatchClient() as client:
        for condition, seed in cfg["checkpoints"]:
            run_checkpoint(root, cfg, client, condition, seed, account)
    return aggregate(root, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
