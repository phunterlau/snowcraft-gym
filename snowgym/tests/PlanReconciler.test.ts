import { describe, expect, it } from 'vitest';
import type { UnitObservation } from '../observations/Observation';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
  type EngagementPolicy,
} from '../orchestration/command/CommandPlan';
import { commandedTenVsTenPlan } from '../orchestration/examples/CommandedReplayExample';
import {
  PlanReconciler,
  PlanReconciliationError,
  type CandidatePlanEnvelope,
} from '../orchestration/lifecycle/PlanReconciler';
import { observationWith } from './orchestrationTestHelpers';

describe('PlanReconciler', () => {
  it('accepts stale provenance while shrinking a three-group plan to the living roster', () => {
    const observation = atTick(observationWith({ allies: [unit(1, 'blue', -8, 0)] }), 120);
    const result = new PlanReconciler().reconcile(
      candidate(commandedTenVsTenPlan(), 0, 'stale-state-hash'),
      observation,
    );

    expect(result.sourceAgeTicks).toBe(120);
    expect(result.plan.envelope.source.sourceStateHash).toBe('stale-state-hash');
    expect(result.plan.groups.map(({ role }) => role)).toEqual(['main']);
    expect(result.plan.groups[0].assignment.unitIds).toEqual([1]);
    expect(result.repairs.map(({ reason }) => reason)).toEqual([
      'roster_too_small',
      'roster_too_small',
    ]);
  });

  it('repairs support orders whose target group was removed', () => {
    const result = new PlanReconciler().reconcile(
      candidate(supportFirstPlan()),
      observationWith({ allies: [unit(1, 'blue', -8, 0)] }),
    );

    expect(result.plan.groups).toHaveLength(1);
    expect(result.plan.groups[0].command).toMatchObject({
      role: 'main',
      selection: 'balanced',
      order: { mission: 'engage', objective: { kind: 'enemy_cluster', select: 'nearest' } },
    });
    expect(result.repairs.map(({ reason }) => reason)).toEqual([
      'roster_too_small',
      'missing_support_target',
    ]);
  });

  it('turns enemy objectives into a safe hold after enemy elimination', () => {
    const result = new PlanReconciler().reconcile(
      candidate(oneGroupEngagePlan()),
      observationWith({ enemies: [] }),
    );

    expect(result.plan.groups[0].command.order).toMatchObject({
      mission: 'hold',
      objective: { kind: 'current_position' },
    });
    expect(result.repairs.map(({ reason }) => reason)).toEqual(['enemy_force_eliminated']);
  });

  it('rejects future provenance and invalid model output before grounding', () => {
    const reconciler = new PlanReconciler();
    expect(() =>
      reconciler.reconcile(candidate(oneGroupEngagePlan(), 1), observationWith()),
    ).toThrow(PlanReconciliationError);
    expect(() =>
      reconciler.reconcile(
        { ...candidate(oneGroupEngagePlan()), decision: { schemaVersion: 'wrong' } },
        observationWith(),
      ),
    ).toThrow('schemaVersion');
  });
});

function candidate(
  decision: CommandPlan,
  sourceTick = 0,
  sourceStateHash?: string,
): CandidatePlanEnvelope {
  return {
    planId: 'candidate-plan',
    source: { requestId: 'candidate-request', sourceTick, sourceStateHash },
    decision,
  };
}

function oneGroupEngagePlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [
      {
        role: 'main',
        allocationWeight: 1,
        selection: 'balanced',
        order: {
          mission: 'engage',
          objective: { kind: 'enemy_cluster', select: 'nearest' },
          approach: 'direct',
          engagement: engagement(),
        },
      },
    ],
  };
}

function supportFirstPlan(): CommandPlanEnvelope['decision'] {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [
      {
        role: 'main',
        allocationWeight: 1,
        selection: 'balanced',
        order: {
          mission: 'support',
          objective: { kind: 'ally_group', role: 'reserve' },
          approach: 'direct',
          engagement: engagement(),
        },
      },
      {
        ...oneGroupEngagePlan().groups[0],
        role: 'reserve',
      },
    ],
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

function atTick(
  observation: ReturnType<typeof observationWith>,
  tick: number,
): ReturnType<typeof observationWith> {
  return { ...observation, tick };
}

function unit(id: number, team: UnitObservation['team'], x: number, y: number): UnitObservation {
  return {
    id,
    team,
    x,
    y,
    vx: 0,
    vy: 0,
    health: 100,
    maxHealth: 100,
    alive: true,
    state: 'idle',
    throwCooldown: 0,
    charge: 0,
  };
}
