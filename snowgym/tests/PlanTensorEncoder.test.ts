import { describe, expect, it } from 'vitest';
import type { UnitObservation } from '../observations/Observation';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
  type EngagementPolicy,
} from '../orchestration/command/CommandPlan';
import { PlanGrounder } from '../orchestration/grounding/PlanGrounder';
import {
  PLAN_FEATURE_LAYOUT,
  PLAN_FEATURE_VECTOR_SIZE,
  PLAN_GROUP_SLOTS,
  encodePlanTensor,
  planRoleSlot,
} from '../training/plan/PlanTensorEncoder';
import { observationWith } from './orchestrationTestHelpers';

describe('PlanTensorEncoder', () => {
  it('encodes stable role slots, directives, geometry, fractions, support, and age', () => {
    const observation = observationWith({
      allies: units(10, 'blue', -10),
      enemies: units(10, 'red', 10, 100),
    });
    const grounder = new PlanGrounder();
    const grounded = grounder.ground(envelope(splitPlan()), observation);
    const encoded = encodePlanTensor(
      { plan: grounded, activatedAtTick: 60, version: 1 },
      observation,
      960,
    );

    expect(encoded.groups).toHaveLength(PLAN_GROUP_SLOTS * PLAN_FEATURE_VECTOR_SIZE);
    expect([...encoded.groupMask]).toEqual([1, 1, 1]);
    const main = row(encoded.groups, planRoleSlot('main'));
    const reserve = row(encoded.groups, planRoleSlot('reserve'));
    expect(main[PLAN_FEATURE_LAYOUT.mission.offset]).toBe(1);
    expect(main[PLAN_FEATURE_LAYOUT.allocationFraction.offset]).toBeCloseTo(0.6);
    expect(main[PLAN_FEATURE_LAYOUT.assignedFraction.offset]).toBeCloseTo(0.6);
    expect(reserve[PLAN_FEATURE_LAYOUT.mission.offset + 4]).toBe(1);
    expect(reserve[PLAN_FEATURE_LAYOUT.supportRole.offset]).toBe(1);
    expect(reserve[PLAN_FEATURE_LAYOUT.planAge.offset]).toBeCloseTo(0.5);
    expect([...encoded.groups].every(Number.isFinite)).toBe(true);
    expect([...encoded.groups].every((value) => value >= -1 && value <= 1)).toBe(true);
  });

  it('changes directives under a counterfactual plan while keeping the same state', () => {
    const observation = observationWith({
      allies: units(3, 'blue', -10),
      enemies: units(3, 'red', 10, 100),
    });
    const grounder = new PlanGrounder();
    const engage = grounder.ground(envelope(singlePlan('engage')), observation);
    const hold = grounder.ground(envelope(singlePlan('hold')), observation);
    const engageTensor = encodePlanTensor(
      { plan: engage, activatedAtTick: 0, version: 1 },
      observation,
      0,
    );
    const holdTensor = encodePlanTensor(
      { plan: hold, activatedAtTick: 0, version: 1 },
      observation,
      0,
    );

    expect(engageTensor.groupMask).toEqual(holdTensor.groupMask);
    expect(engageTensor.groups).not.toEqual(holdTensor.groups);
    expect(() =>
      encodePlanTensor({ plan: engage, activatedAtTick: 10, version: 1 }, observation, 9),
    ).toThrow('at or after plan activation');
  });
});

function row(values: Float32Array, slot: number): Float32Array {
  return values.subarray(slot * PLAN_FEATURE_VECTOR_SIZE, (slot + 1) * PLAN_FEATURE_VECTOR_SIZE);
}

function envelope(decision: CommandPlan): CommandPlanEnvelope {
  return {
    planId: 'tensor-plan',
    source: { requestId: 'tensor-request', sourceTick: 0 },
    decision,
  };
}

function singlePlan(mission: 'engage' | 'hold'): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [
      {
        role: 'main',
        allocationWeight: 1,
        selection: 'balanced',
        order:
          mission === 'engage'
            ? {
                mission,
                objective: { kind: 'enemy_cluster', select: 'nearest' },
                approach: 'direct',
                engagement: engagement(),
              }
            : {
                mission,
                objective: { kind: 'current_position' },
                approach: 'direct',
                engagement: engagement(),
              },
      },
    ],
  };
}

function splitPlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [
      {
        role: 'main',
        allocationWeight: 6,
        selection: 'frontline',
        order: {
          mission: 'engage',
          objective: { kind: 'enemy_cluster', select: 'nearest' },
          approach: 'direct',
          engagement: engagement(),
        },
      },
      {
        role: 'maneuver',
        allocationWeight: 3,
        selection: 'nearest_right_lane',
        order: {
          mission: 'advance',
          objective: { kind: 'region', region: 'right_lane' },
          approach: 'right_flank',
          engagement: engagement(),
        },
      },
      {
        role: 'reserve',
        allocationWeight: 1,
        selection: 'healthiest',
        order: {
          mission: 'support',
          objective: { kind: 'ally_group', role: 'main' },
          approach: 'avoid_center',
          engagement: engagement(),
        },
      },
    ],
  };
}

function engagement(): EngagementPolicy {
  return {
    posture: 'balanced',
    fire: 'focus',
    preferredRange: 'medium',
    cohesion: 'normal',
  };
}

function units(
  count: number,
  team: UnitObservation['team'],
  x: number,
  idOffset = 0,
): UnitObservation[] {
  return Array.from({ length: count }, (_, index) => ({
    id: idOffset + index + 1,
    team,
    x,
    y: index * 1.2 - ((count - 1) * 1.2) / 2,
    vx: 0,
    vy: 0,
    health: 100,
    maxHealth: 100,
    alive: true,
    state: 'idle',
    throwCooldown: 0,
    charge: 0,
  }));
}
