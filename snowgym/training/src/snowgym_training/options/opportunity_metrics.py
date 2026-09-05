"""Conditional-head labels and read-only geometry diagnostics for R1k/R1l."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch

from ..executor.geometry_probe import geometry_loss


def conditional_loss(prediction, observation, labels):
    """R1i coefficients, with independent masks/targets for each geometry head.

    Recommendations never change the categorical teacher label. Empty heads
    contribute differentiable zero and each nonempty head normalizes itself.
    """
    zeros = torch.zeros_like(labels["move_mask"], dtype=torch.long)
    move = geometry_loss(prediction, {
        "action_type": torch.where(labels["move_mask"], 1, zeros),
        "target": labels["move_target"], "power": labels["power"],
    }, observation)["move"]
    shot = geometry_loss(prediction, {
        "action_type": torch.where(labels["shot_mask"], 2, zeros),
        "target": labels["shot_target"], "power": labels["power"],
    }, observation)
    return {"move": move, "direction": shot["direction"],
            "endpoint": shot["endpoint"], "power": shot["power"],
            "total": move + shot["direction"] + .1 * shot["endpoint"] + .5 * shot["power"]}


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return None if not len(values) else {
        "count": len(values), "mean": float(values.mean()),
        "p50": float(np.quantile(values, .5)), "p90": float(np.quantile(values, .9)),
        "p99": float(np.quantile(values, .99)), "max": float(values.max()),
    }


def physical_errors(position, move, shot, power, move_reference, shot_reference, power_reference):
    scale = np.asarray([50., 40.])
    ray, reference = (np.asarray(shot)-position)*scale, (np.asarray(shot_reference)-position)*scale
    length, reference_length = np.linalg.norm(ray), np.linalg.norm(reference)
    valid = length > 1e-6 and reference_length > 1e-6
    angle = math.atan2(abs(ray[0]*reference[1]-ray[1]*reference[0]), float(ray @ reference)) if valid else None
    return {
        "moveErrorWorld": float(np.linalg.norm((np.asarray(move)-move_reference)*scale)),
        "angleDegrees": None if angle is None else math.degrees(angle),
        "chordMissWorld": None if angle is None else float(2*reference_length*math.sin(angle/2)),
        "powerError": abs(float(power-power_reference)), "rayLength": float(length),
        "degenerateRay": not valid,
        "moveSaturation": bool(np.any(np.abs(move) >= .99)),
        "shotSaturation": bool(np.any(np.abs(shot) >= .99)),
    }


def cross_tabs(rows):
    cells = {}
    for teacher in range(4):
        for learner in range(4):
            cell = [r for r in rows if r["teacherType"] == teacher and r["learnerType"] == learner]
            cells[f"{teacher}:{learner}"] = {
                "count": len(cell), "deploymentFraction": len(cell)/max(len(rows), 1),
                **{key: describe([r[key] for r in cell if r[key] is not None])
                   for key in ("moveErrorWorld", "angleDegrees", "powerError", "chordMissWorld")},
                "moveAvailable": sum(r["moveAvailable"] for r in cell),
                "shotAvailable": sum(r["shotAvailable"] for r in cell),
                "moveLegal": sum(r["legal"][1] for r in cell),
                "shotLegal": sum(r["legal"][2] for r in cell),
            }
    exclusion = {}
    for name, kind in (("move", 1), ("shot", 2)):
        used = [r for r in rows if r["learnerType"] == kind]
        excluded = sum(r["teacherType"] != kind for r in used)
        exclusion[name] = {"used": len(used), "oldMaskExcluded": excluded,
                           "fraction": excluded/len(used) if used else None}
    return {"orientation": "teacher:learner", "cells": cells, "oldMaskExclusion": exclusion}


def select_opportunities(rows, channel, *, limit=64, per_episode=4, seed_range=None):
    """Hardest error first within agreement strata; deterministic capped balance."""
    kind = 1 if channel == "move" else 2
    key = {"move": "moveErrorWorld", "aim": "angleDegrees", "power": "powerError"}[channel]
    candidates = [r for r in rows if r["learnerType"] == kind and r["legal"][kind]
                  and r["moveAvailable" if kind == 1 else "shotAvailable"] and r[key] is not None
                  and (seed_range is None or seed_range[0] <= r["seed"] <= seed_range[1])]
    bins = [[r for r in candidates if (r["teacherType"] == kind) == agree] for agree in (True, False)]
    for group in bins:
        group.sort(key=lambda r: (-r[key], r["seed"], r["decision"], r["unitId"]))
    result, counts = [], {}
    while len(result) < limit and any(bins):
        for group in bins:
            while group and counts.get(group[0]["seed"], 0) >= per_episode:
                group.pop(0)
            if group and len(result) < limit:
                row = group.pop(0)
                result.append(row)
                counts[row["seed"]] = counts.get(row["seed"], 0)+1
    return result


def fit_batch(states, rows):
    if not rows:
        raise ValueError("hard opportunity set is empty")
    selected = [states[r["stateIndex"]] for r in rows]
    observation = {k: torch.as_tensor(np.stack([s["observation"][k][0] for s in selected]))
                   for k in selected[0]["observation"]}
    labels = {k: torch.as_tensor(np.stack([s["labels"][k][0] for s in selected]))
              for k in selected[0]["labels"]}
    # Supervise only this opportunity's invoked conditional head, never all
    # other units merely because they shared a selected world state.
    labels["move_mask"] = torch.zeros_like(labels["move_mask"], dtype=torch.bool)
    labels["shot_mask"] = torch.zeros_like(labels["shot_mask"], dtype=torch.bool)
    for index, row in enumerate(rows):
        labels["move_mask" if row["learnerType"] == 1 else "shot_mask"][index, row["slot"]] = True
    return observation, labels


def gradient_audit(model, observation, labels):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    components, vectors = {}, {}
    weights = {"move": 1., "direction": 1., "endpoint": .1, "power": .5}
    for name, weight in weights.items():
        model.zero_grad(set_to_none=True)
        loss = conditional_loss(model(observation), observation, labels)[name]
        loss.backward()
        vector = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                            for _, p in parameters]).detach()
        if not torch.isfinite(vector).all():
            raise ValueError("non-finite component gradient")
        vectors[name] = vector
        components[name] = {"loss": float(loss.detach()), "weightedLoss": weight*float(loss.detach()),
            "validLabels": int(labels["move_mask" if name == "move" else "shot_mask"].sum()),
            "gradientNorm": float(vector.norm()), "weightedGradientNorm": weight*float(vector.norm()),
            "modules": {group: sum(float(p.grad.square().sum()) for n, p in parameters
                        if n.startswith(group+".") and p.grad is not None)**.5
                        for group in ("encoders", "move", "shot")}}
    cosine = {f"{a}:{b}": float((va @ vb)/(va.norm()*vb.norm()))
              if va.norm() > 0 and vb.norm() > 0 else None
              for a, va in vectors.items() for b, vb in vectors.items()}
    model.zero_grad(set_to_none=True)
    return {"components": components, "cosines": cosine,
            "targetEncoderReachable": all(components[k]["modules"]["encoders"] > 0
                                          for k in ("move", "direction", "power"))}


def physical_jacobian_audit(model, observation, labels):
    """Central differences of physical outputs against residual output biases."""
    probe = copy.deepcopy(model)
    results = {}
    for channel in ("move", "shot"):
        mask = labels["move_mask" if channel == "move" else "shot_mask"]
        indices = mask.nonzero()
        if not len(indices):
            results[channel] = {"available": False}
            continue
        batch, unit = indices[0].tolist()
        obs = {k: v[batch:batch+1] for k, v in observation.items()}
        parameter = getattr(probe, channel)[-1].bias
        original = parameter.detach().clone()

        def outputs():
            prediction = probe(obs)
            point = prediction["target_by_action"][0, unit, 1 if channel == "move" else 2]
            if channel == "move":
                return point*point.new_tensor([50., 40.])
            ray = (point-obs["allies"][0, unit, 2:4])*point.new_tensor([50., 40.])
            direction = torch.nn.functional.normalize(ray, dim=0, eps=1e-6)
            return torch.cat([direction, prediction["power"][0, unit].reshape(1)])

        values = outputs()
        analytic = torch.stack([torch.autograd.grad(value, parameter, retain_graph=True)[0]
                                for value in values]).detach()
        numerical = torch.zeros_like(analytic)
        for index in range(parameter.numel()):
            with torch.no_grad():
                parameter.copy_(original)
                parameter[index] += .001
                plus = outputs().clone()
                parameter[index] -= .002
                minus = outputs().clone()
                numerical[:, index] = (plus-minus)/.002
        with torch.no_grad():
            parameter.copy_(original)
        results[channel] = {"available": True, "analytic": analytic.tolist(),
                            "finiteDifference": numerical.tolist(), "norm": float(analytic.norm()),
                            "passed": bool(torch.allclose(analytic, numerical, atol=.01, rtol=.05))}
    return results


def hard_fit(model, states, rows, *, steps=200, learning_rate=.001):
    from ..checkpoint import semantic_state_digest
    training, validation = [], []
    for channel in ("move", "aim"):
        training.extend(select_opportunities(rows, channel, limit=64, seed_range=(100000, 100031)))
        validation.extend(select_opportunities(rows, channel, limit=32, seed_range=(100032, 100039)))
    if not training or not validation:
        return {"passed": False, "reason": "missing hard training or validation opportunities"}
    train_obs, train_labels = fit_batch(states, training)
    val_obs, val_labels = fit_batch(states, validation)
    source_digest = semantic_state_digest(model.state_dict())
    probe = copy.deepcopy(model)
    optimizer = torch.optim.Adam([p for p in probe.parameters() if p.requires_grad], lr=learning_rate)

    def losses(obs, labels):
        with torch.no_grad():
            return {k: float(v) for k, v in conditional_loss(probe(obs), obs, labels).items()}

    before, val_before = losses(train_obs, train_labels), losses(val_obs, val_labels)
    gradients = gradient_audit(probe, train_obs, train_labels)
    jacobians = physical_jacobian_audit(probe, train_obs, train_labels)
    trace = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = conditional_loss(probe(train_obs), train_obs, train_labels)["total"]
        if not torch.isfinite(loss):
            raise ValueError("non-finite hard-fit loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in probe.parameters() if p.requires_grad], .5,
                                      error_if_nonfinite=True)
        optimizer.step()
        trace.append(float(loss.detach()))
    after, val_after = losses(train_obs, train_labels), losses(val_obs, val_labels)
    unchanged = semantic_state_digest(model.state_dict()) == source_digest
    if not unchanged or semantic_state_digest(probe.source.state_dict()) != semantic_state_digest(model.source.state_dict()):
        raise RuntimeError("hard fit changed frozen source")
    reduction = 1-after["total"]/max(before["total"], 1e-12)
    passed = (reduction >= .5 and val_after["total"] < val_before["total"]
              and gradients["targetEncoderReachable"]
              and all(v.get("passed", False) for v in jacobians.values()))
    return {"passed": passed, "reduction": reduction, "before": before, "after": after,
            "validationBefore": val_before, "validationAfter": val_after,
            "gradients": gradients, "physicalJacobians": jacobians, "lossTrace": trace,
            "trainingOpportunities": [r["opportunityId"] for r in training],
            "validationOpportunities": [r["opportunityId"] for r in validation],
            "sourceUnchanged": unchanged, "disposable": True, "steps": steps}
