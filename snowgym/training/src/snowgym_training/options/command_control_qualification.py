"""M8-S17: qualification replication of S16's left/right command control (`reviews/m8_s17_declaration.md`).

Reuses `command_control_audit` (S16) unchanged; `declare()` refuses to run unless its digest equals the
implementation digest sealed in S16's declaration. Adds the qualification panel (S16's reserved 2730000-2730399), a
held-out-geometry secondary panel (d = 15, 25), the conditions `correct`/`other`, and the replication verdict over
the three S14 finals. No training; no existing module is edited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..trainer import resolve_git_commit
from . import command_control_audit as s16
from . import death_rate_ppo as dr
from . import enemy_relative_throw_ppo_retention as s15
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .interventions import require_capabilities
from .reservoir import file_digest
from .supervised_probe import write_json

CONDITIONS = ("correct", "other")
PANELS = ("qualification", "heldout")
VERDICTS = ("replicated", "partially-replicated", "not-replicated")
S16_DEVELOPMENT_FINAL_DELTA = .951  # m8_s16_results.md, cohort-averaged final deterministic Δo


def configuration():
    return {**s16.configuration(), "format": "snowgym.m8-s17-command-control-qualification-config.v0",
        "s16Run": "m8_s16_command_control_audit_v0", "conditions": CONDITIONS,
        # top-level panel keys mirror the qualification panel so no S16 helper can fall back to S16's dev panel
        "worldSeedBase": 2730000, "worlds": 400, "spreads": (10, 20, 30),
        "panels": {"qualification": {"worldSeedBase": 2730000, "worlds": 400, "spreads": (10, 20, 30)},
                   "heldout": {"worldSeedBase": 2740000, "worlds": 200, "spreads": (15, 25)}},
        "torchSeedBase": 987000, "bootstrapSeed": 987001, "predictionTolerance": .10,
        "budgetCap": 6_400_000,
        "assistType": "none at learned-policy runtime; plan-aware teacher only as a separate positive control",
        "assistVersion": "snowgym.m8-s17.v0", "autonomousQualificationEligible": False}


def panel_cfg(cfg, panel):
    """S16's functions read the panel from `worldSeedBase`/`worlds`/`spreads`; give them this panel's values."""
    return {**cfg, **cfg["panels"][panel]}


def panel_worlds(cfg, panel):
    return s16.panel(panel_cfg(cfg, panel))


def budget_bound(cfg):
    horizon = cfg["commandHorizon"]
    worlds = sum(p["worlds"] for p in cfg["panels"].values())
    gate = s16.budget_bound(cfg)["gate"]
    teacher = worlds * len(s16.TASKS) * len(CONDITIONS) * horizon
    learned = (worlds * len(s16.TASKS) * len(CONDITIONS) * len(cfg["cohorts"]) * len(s16.POLICIES) * len(s16.MODES)
               * horizon)
    return {"gate": gate, "teacher": teacher, "learned": learned, "total": gate + teacher + learned}


def stochastic_torch_seed(cfg, cohort, panel, policy, condition, task):
    return (cfg["torchSeedBase"] + 100 * cohort + 40 * PANELS.index(panel) + 20 * CONDITIONS.index(condition)
            + 2 * s16.TASKS.index(task) + s16.POLICIES.index(policy))


def reserved_seeds(cfg):
    return {panel: set(range(p["worldSeedBase"], p["worldSeedBase"] + p["worlds"]))
            for panel, p in cfg["panels"].items()}


# -- Collection -------------------------------------------------------------------------------------------------


def cell_selects(condition, task, worlds):
    return [s16.execution_select(condition, task, index, None) for index, _, _ in worlds]


def teacher_cells(client, cfg, root, account):
    for panel in PANELS:
        worlds = panel_worlds(cfg, panel)
        for condition in CONDITIONS:
            for task in s16.TASKS:
                rows = s16.run_cell(client, cfg, worlds, requested_select=task, mirror_select=s16.MIRROR[task],
                    execution_selects=cell_selects(condition, task, worlds), choose=s16.teacher_chooser,
                    source=f"s17-teacher-{panel}-{condition}-{task}", account=account)
                s16.write_rows(Path(root) / "teacher" / panel / condition / task, rows)


def learned_cells(client, cfg, policies, root, account):
    for panel in PANELS:
        worlds = panel_worlds(cfg, panel)
        for cohort in cfg["cohorts"]:
            for policy in s16.POLICIES:
                for mode in s16.MODES:
                    for condition in CONDITIONS:
                        for task in s16.TASKS:
                            if mode == "stochastic":
                                torch.manual_seed(stochastic_torch_seed(cfg, cohort, panel, policy, condition, task))
                            rows = s16.run_cell(client, cfg, worlds, requested_select=task,
                                mirror_select=s16.MIRROR[task], execution_selects=cell_selects(condition, task, worlds),
                                choose=s16.model_chooser(policies[cohort][policy], mode == "deterministic"),
                                source=f"s17-{panel}-c{cohort}-{policy}-{mode}-{condition}-{task}", account=account)
                            s16.write_rows(Path(root) / "learned" / panel / f"cohort-{cohort}" / policy / mode
                                           / condition / task, rows)
                print(json.dumps({"stage": "learned", "panel": panel, "cohort": cohort, "policy": policy}), flush=True)


# -- Analysis (S16 §6 rules; declaration §4, §5) ------------------------------------------------------------------


def teacher_check(root, cfg, panel):
    pcfg, worlds = panel_cfg(cfg, panel), panel_worlds(cfg, panel)
    cells = s16.load_cells(Path(root) / "teacher" / panel, CONDITIONS, worlds)
    summaries = {c: s16.condition_summary(cells[c], worlds) for c in CONDITIONS}
    delta = s16.contrasts(cells, worlds, pcfg, CONDITIONS)
    by_spread = {f"d{d}": float(np.mean(s16.world_values(cells["correct"], s16.order_metric,
                 [w for w in worlds if w[2] == d]))) for d in pcfg["spreads"]}
    passed = (summaries["correct"]["order"] >= cfg["teacherOrderMin"]
              and delta["other"]["all"]["order"]["interval95"][0] >= cfg["teacherDeltaLowerMin"]
              and all(value >= cfg["teacherSpreadOrderMin"] for value in by_spread.values()))
    identity = s16.mirror_identity(cells)
    return {"passed": passed, "summaries": summaries, "contrasts": delta, "correctOrderBySpread": by_spread,
            "mirrorIdentityViolation": identity,
            "mirrorIdentityWithinTolerance": all(v <= cfg["teacherIdentityTolerance"] for v in identity.values())}


def learned_panel(root, cfg, panel):
    pcfg, worlds = panel_cfg(cfg, panel), panel_worlds(cfg, panel)
    policies, cell_sets = {}, {}
    for cohort in cfg["cohorts"]:
        for policy in s16.POLICIES:
            for mode in s16.MODES:
                cells = s16.load_cells(Path(root) / "learned" / panel / f"cohort-{cohort}" / policy / mode,
                                       CONDITIONS, worlds)
                cell_sets.setdefault(f"{policy}-{mode}", []).append(cells)
                summaries = {c: s16.condition_summary(cells[c], worlds) for c in CONDITIONS}
                delta = s16.contrasts(cells, worlds, pcfg, CONDITIONS)
                identity = s16.mirror_identity(cells)
                policies[f"c{cohort}-{policy}-{mode}"] = {"summaries": summaries, "contrasts": delta,
                    "class": s16.classify(delta["other"]["all"]["order"], summaries["correct"]["informative"], cfg),
                    "mirrorIdentityViolation": identity,
                    "mirrorIdentityWithinTolerance": None if mode == "stochastic" else all(
                        v <= cfg["learnedIdentityTolerance"] for v in identity.values())}
    averaged = {key: s16.contrasts(s16.averaged_cells(sets), worlds, pcfg, CONDITIONS)
                for key, sets in cell_sets.items()}
    return {"policies": policies, "cohortAveraged": averaged}


def verdict(qualification, cfg):
    classes = {c: qualification["policies"][f"c{c}-final-deterministic"]["class"] for c in cfg["cohorts"]}
    replicated = sum(value == "controllable" for value in classes.values())
    name = ("replicated" if replicated == len(classes) else "partially-replicated" if replicated else
            "not-replicated")
    identity_ok = all(entry["mirrorIdentityWithinTolerance"] for key, entry in qualification["policies"].items()
                      if key.endswith("-deterministic"))
    return {"verdict": name, "finalClasses": classes, "controllableFinals": replicated,
            "identityWithinTolerance": identity_ok,
            "note": None if identity_ok else "identity check exceeded tolerance; investigate before interpreting",
            "authorizes": "nothing; a follow-up needs its own declaration"}


def prediction_scores(teacher, qualification, heldout, rules, cfg):
    averaged = qualification["cohortAveraged"]["final-deterministic"]["other"]["all"]["order"]["mean"]
    d10 = {c: qualification["policies"][f"c{c}-initializer-deterministic"]["contrasts"]["other"]["d10"]["order"]
           for c in (1, 3) if c in cfg["cohorts"]}
    return {"1-teacherGatePasses": teacher["qualification"]["passed"],
        "2-replicated": rules["verdict"] == "replicated",
        "3-averagedFinalDeltaNearDevelopment":
            abs(averaged - S16_DEVELOPMENT_FINAL_DELTA) <= cfg["predictionTolerance"],
        "4-initializersInsensitiveAtD10": all(d["interval95"][0] <= 0 <= d["interval95"][1] for d in d10.values()),
        "5-heldoutFinalsControllable": all(heldout["policies"][f"c{c}-final-deterministic"]["class"] == "controllable"
                                            for c in cfg["cohorts"]),
        "averagedFinalDelta": averaged, "initializerD10Delta": d10}


# -- Declaration, run, archive --------------------------------------------------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    runs = TRAINING / "runs"
    s16_declaration = json.loads((runs / cfg["s16Run"] / "declaration.json").read_text(encoding="utf-8"))
    reused = file_digest(Path(s16.__file__).resolve())
    if reused != s16_declaration["implementationDigest"]:
        raise RuntimeError("command_control_audit.py differs from the implementation S16 sealed")
    manifests = {"s16": dr.verify_sealed(runs / cfg["s16Run"]), "s15": dr.verify_sealed(runs / cfg["s15Run"]),
                 "s14Training": dr.verify_sealed(runs / cfg["finalRun"]),
                 "s14Probe": dr.verify_sealed(runs / cfg["probeRun"]), "s12": dr.verify_sealed(runs / cfg["sourceRun"])}
    checkpoints = {str(c): {"final": file_digest(s15.final_checkpoint_path(cfg, c)),
                            "initializer": file_digest(s15.initializer_checkpoint_path(cfg, c))}
                   for c in cfg["cohorts"]}
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "sourceManifests": manifests, "checkpointDigests": checkpoints,
        "reusedImplementationDigest": reused,
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s17_declaration.md"),
        "implementationDigest": file_digest(Path(__file__).resolve()), "pinnedE3Digests": pinned,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, gate, teacher, learned, spent):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    gate_passed = bool(gate) and all(entry["exact"] for entry in gate.values())
    outcome = ("gate-failed" if not gate_passed else "teacher-gate-failed"
               if not teacher.get("qualification", {}).get("passed") else "complete")
    rules = verdict(learned["qualification"], cfg) if learned else None
    report = {"format": "snowgym.m8-s17-command-control-qualification-report.v0", "outcome": outcome,
        "regressionGate": {"passed": gate_passed, "cells": gate}, "teacher": teacher, "learned": learned,
        "verdict": rules, "heldoutTeacherValid": teacher.get("heldout", {}).get("passed"),
        "predictions": prediction_scores(teacher, learned["qualification"], learned["heldout"], rules, cfg)
            if learned else None,
        "torchThreads": torch.get_num_threads(), "blockWorlds": cfg["blockWorlds"],
        "simulatorDecisions": spent, "totalSimulatorDecisions": sum(spent.values()),
        "budgetBound": budget_bound(cfg), "withinBudgetCap": sum(spent.values()) <= cfg["budgetCap"],
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s17-manifest.v0")
    return report


def run(output, cfg=None):
    root = Path(output)
    cfg = cfg or configuration()
    declare(root, cfg)
    spent = {"gate": 0, "teacher": 0, "learned": 0}

    def accountant(stage):
        def account(count):
            spent[stage] += count
            if sum(spent.values()) > cfg["budgetCap"]:
                raise ValueError("M8-S17 budget exceeded")
        return account

    teacher, learned = {}, None
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        policies = {cohort: s15.load_policies(cfg, cohort) for cohort in cfg["cohorts"]}
        gate = s16.regression_gate(client, cfg, policies, root, accountant("gate"))
        print(json.dumps({"stage": "gate", "passed": all(e["exact"] for e in gate.values()), "decisions": spent}),
              flush=True)
        if all(entry["exact"] for entry in gate.values()):
            teacher_cells(client, cfg, root, accountant("teacher"))
            teacher = {panel: teacher_check(root, cfg, panel) for panel in PANELS}
            print(json.dumps({"stage": "teacher", "passed": {p: t["passed"] for p, t in teacher.items()},
                              "decisions": spent}), flush=True)
            if teacher["qualification"]["passed"]:
                learned_cells(client, cfg, policies, root, accountant("learned"))
                learned = {panel: learned_panel(root, cfg, panel) for panel in PANELS}
    return aggregate(root, cfg, gate, teacher, learned, spent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args().output)
    print(json.dumps({"outcome": report["outcome"], "verdict": (report["verdict"] or {}).get("verdict"),
                      "decisions": report["totalSimulatorDecisions"]}))
