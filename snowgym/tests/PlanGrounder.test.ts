import { describe, expect, it } from 'vitest';
import type { UnitObservation } from '../observations/Observation';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlanEnvelope,
} from '../orchestration/command/CommandPlan';
import { commandedTenVsTenPlan } from '../orchestration/examples/CommandedReplayExample';
import { PlanGrounder, PlanGroundingError } from '../orchestration/grounding/PlanGrounder';
import { observationWith } from './orchestrationTestHelpers';

describe('PlanGrounder', () => {
  it('assembles stable 6:3:1 assignments and late-bound support objectives', () => {
    const observation = observationWith({
      allies: units(10, 'blue', -10),
      enemies: units(10, 'red', 10, 100),
    });
    const plan = new PlanGrounder().ground(envelope(commandedTenVsTenPlan()), observation);

    expect(plan.groups.map(({ role, assignment }) => [role, assignment.unitIds.length])).toEqual([
      ['main', 6],
      ['maneuver', 3],
      ['reserve', 1],
    ]);
    expect(
      plan.groups.flatMap(({ assignment }) => assignment.unitIds).sort((a, b) => a - b),
    ).toEqual(Array.from({ length: 10 }, (_, index) => index + 1));
    expect(plan.groups.find(({ role }) => role === 'reserve')?.objective).toMatchObject({
      kind: 'ally_group',
      role: 'main',
      unitIds: plan.groups.find(({ role }) => role === 'main')?.assignment.unitIds,
    });
  });

  it('defensively rejects assignment-dependent nearest-objective plans', () => {
    const decision = commandedTenVsTenPlan();
    const invalid = {
      ...decision,
      groups: decision.groups.map((group) =>
        group.role === 'reserve' ? { ...group, selection: 'nearest_objective' as const } : group,
      ),
    };
    expect(() =>
      new PlanGrounder().ground(
        envelope(invalid),
        observationWith({ allies: units(10, 'blue', -10) }),
      ),
    ).toThrow(PlanGroundingError);
  });
});

function envelope(decision: CommandPlanEnvelope['decision']): CommandPlanEnvelope {
  return {
    planId: 'test-plan',
    source: { requestId: 'test-request', sourceTick: 0 },
    decision: { ...decision, schemaVersion: COMMAND_PLAN_VERSION },
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
