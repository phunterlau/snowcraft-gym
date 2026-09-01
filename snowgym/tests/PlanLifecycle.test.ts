import { describe, expect, it } from 'vitest';
import type { Observation, UnitObservation } from '../observations/Observation';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
  type EngagementPolicy,
  type GroupCommand,
} from '../orchestration/command/CommandPlan';
import { commandedTenVsTenPlan } from '../orchestration/examples/CommandedReplayExample';
import { PlanGrounder } from '../orchestration/grounding/PlanGrounder';
import { PlanLifecycle } from '../orchestration/lifecycle/PlanLifecycle';
import type { CandidatePlanEnvelope } from '../orchestration/lifecycle/PlanReconciler';
import { PlanStore } from '../orchestration/runtime/PlanStore';
import { observationWith } from './orchestrationTestHelpers';

describe('PlanLifecycle', () => {
  it('atomically activates a reconciled stale candidate and records its repairs', () => {
    const initial = observationWith({ allies: [unit(1, 'blue', -8, 0)] });
    const store = storeWith(oneGroupEngagePlan(), initial);
    const lifecycle = new PlanLifecycle(store);
    const current = atTick(initial, 20);

    const outcome = lifecycle.activateCandidate(candidate(commandedTenVsTenPlan(), 2), current);

    expect(outcome.status).toBe('repaired');
    expect(outcome).toMatchObject({ sourceAgeTicks: 18, snapshot: { version: 2 } });
    expect(store.current().plan.groups.map(({ role }) => role)).toEqual(['main']);
    expect(lifecycle.events()).toMatchObject([
      {
        type: 'candidate_activated',
        tick: 20,
        previousPlanId: 'initial-plan',
        activePlanId: 'candidate-plan',
        version: 2,
      },
    ]);
  });

  it('rejects invalid candidates without disturbing the active snapshot', () => {
    const observation = observationWith();
    const store = storeWith(oneGroupEngagePlan(), observation);
    const lifecycle = new PlanLifecycle(store);
    const before = store.current();
    const invalid = { ...candidate(oneGroupEngagePlan()), decision: { schemaVersion: 'wrong' } };

    const outcome = lifecycle.activateCandidate(invalid, observation);

    expect(outcome.status).toBe('rejected');
    expect(outcome.snapshot).toBe(before);
    expect(store.current()).toBe(before);
    expect(lifecycle.events()[0]).toMatchObject({
      type: 'candidate_rejected',
      activePlanId: 'initial-plan',
      version: 1,
    });
  });

  it('expires a plan into a deterministic fallback and emits an activation trace', () => {
    const initial = observationWith();
    const store = storeWith(oneGroupEngagePlan(), initial);
    const lifecycle = new PlanLifecycle(store, undefined, undefined, undefined, {
      maxPlanAgeTicks: 10,
    });
    const current = atTick(initial, 10);

    expect(lifecycle.evaluate(current)).toEqual(['plan_expired']);
    const fallback = lifecycle.maintain(current);

    expect(fallback).toMatchObject({
      activatedAtTick: 10,
      version: 2,
      plan: {
        envelope: {
          planId: 'fallback-10-1',
          source: { requestId: 'fallback-plan-expired-10-1', sourceTick: 10 },
        },
      },
    });
    expect(fallback?.plan.groups).toHaveLength(1);
    expect(lifecycle.events()[0]).toMatchObject({
      type: 'fallback_activated',
      trigger: 'plan_expired',
      previousPlanId: 'initial-plan',
      version: 2,
    });
  });

  it('detects major loss and eliminated assigned groups before switching to fallback', () => {
    const initial = observationWith({
      allies: [unit(1, 'blue', -8, -1), unit(2, 'blue', -8, 1)],
    });
    const store = storeWith(twoGroupPlan(), initial);
    const lifecycle = new PlanLifecycle(store);
    const current = atTick(
      observationWith({
        allies: [unit(1, 'blue', -8, -1), unit(2, 'blue', -8, 1, false)],
      }),
      12,
    );

    expect(lifecycle.evaluate(current)).toEqual(['own_force_loss_major', 'group_eliminated']);
    expect(lifecycle.maintain(current)?.plan.envelope.planId).toBe('fallback-12-1');
    expect(lifecycle.events()[0]).toMatchObject({
      type: 'fallback_activated',
      trigger: 'own_force_loss_major',
    });
  });

  it('recognizes completion against a stable map-region anchor', () => {
    const initial = observationWith({ allies: [unit(1, 'blue', -10, 0)] });
    const store = storeWith(oneGroupAdvancePlan(), initial);
    const lifecycle = new PlanLifecycle(store, undefined, undefined, undefined, {
      completionRadius: 0.1,
    });
    const objective = store.current().plan.groups[0].objective.anchor;
    const arrived = atTick(
      observationWith({ allies: [unit(1, 'blue', objective.x, objective.y)] }),
      5,
    );

    expect(lifecycle.evaluate(arrived)).toEqual(['objective_completed']);
  });

  it('returns detached trace records', () => {
    const observation = observationWith();
    const lifecycle = new PlanLifecycle(storeWith(oneGroupEngagePlan(), observation));
    lifecycle.activateCandidate(candidate(oneGroupEngagePlan()), observation);
    const events = lifecycle.events() as unknown as Array<{ type: string }>;
    events[0].type = 'changed';
    events.push({ type: 'added' });

    expect(lifecycle.events()).toHaveLength(1);
    expect(lifecycle.events()[0].type).toBe('candidate_activated');
  });
});

function storeWith(decision: CommandPlan, observation: Observation): PlanStore {
  const envelope: CommandPlanEnvelope = {
    planId: 'initial-plan',
    source: { requestId: 'initial-request', sourceTick: observation.tick },
    decision,
  };
  return new PlanStore(new PlanGrounder().ground(envelope, observation), observation.tick);
}

function candidate(decision: unknown, sourceTick = 0): CandidatePlanEnvelope {
  return {
    planId: 'candidate-plan',
    source: { requestId: 'candidate-request', sourceTick },
    decision,
  };
}

function oneGroupEngagePlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [engageGroup('main')],
  };
}

function twoGroupPlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [engageGroup('main'), engageGroup('reserve')],
  };
}

function oneGroupAdvancePlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [
      {
        role: 'main',
        allocationWeight: 1,
        selection: 'balanced',
        order: {
          mission: 'advance',
          objective: { kind: 'region', region: 'center_lane' },
          approach: 'direct',
          engagement: engagement(),
        },
      },
    ],
  };
}

function engageGroup(role: 'main' | 'reserve'): GroupCommand {
  return {
    role,
    allocationWeight: 1,
    selection: 'balanced' as const,
    order: {
      mission: 'engage' as const,
      objective: { kind: 'enemy_cluster' as const, select: 'nearest' as const },
      approach: 'direct' as const,
      engagement: engagement(),
    },
  };
}

function engagement(): EngagementPolicy {
  return {
    posture: 'balanced' as const,
    fire: 'focus' as const,
    preferredRange: 'medium' as const,
    cohesion: 'normal' as const,
  };
}

function atTick(observation: Observation, tick: number): Observation {
  return { ...observation, tick };
}

function unit(
  id: number,
  team: UnitObservation['team'],
  x: number,
  y: number,
  alive = true,
): UnitObservation {
  return {
    id,
    team,
    x,
    y,
    vx: 0,
    vy: 0,
    health: alive ? 100 : 0,
    maxHealth: 100,
    alive,
    state: alive ? 'idle' : 'defeated',
    throwCooldown: 0,
    charge: 0,
  };
}
