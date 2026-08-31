import { describe, expect, it } from 'vitest';
import type { UnitObservation } from '../observations/Observation';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type GroupCommand,
} from '../orchestration/command/CommandPlan';
import { GroupAllocationError, GroupAllocator } from '../orchestration/grounding/GroupAllocator';
import { observationWith } from './orchestrationTestHelpers';

describe('GroupAllocator', () => {
  it('normalizes 6:3:1 weights exactly across a ten-unit roster', () => {
    const allocator = new GroupAllocator();
    const assignments = allocator.allocate(
      plan([group('main', 6), group('maneuver', 3), group('reserve', 1)]),
      observationWith({ allies: units(10, -10) }),
    );

    expect(assignments).toEqual([
      { role: 'main', unitIds: [1, 2, 3, 4, 5, 6] },
      { role: 'maneuver', unitIds: [7, 8, 9] },
      { role: 'reserve', unitIds: [10] },
    ]);
  });

  it('gives every declared group one unit when a skewed plan is used in 3v3', () => {
    const assignments = new GroupAllocator().allocate(
      plan([group('main', 10), group('maneuver', 1), group('reserve', 1)]),
      observationWith({ allies: units(3, -10) }),
    );
    expect(assignments.map(({ unitIds }) => unitIds.length)).toEqual([1, 1, 1]);
    expect(assignments.flatMap(({ unitIds }) => unitIds).sort((a, b) => a - b)).toEqual([1, 2, 3]);
  });

  it('is stable under input order and uses team-relative selection strategies', () => {
    const roster = [unit(1, -12, 4), unit(2, -8, -4), unit(3, -5, 3), unit(4, -15, -3)];
    const command = plan([
      { ...group('main', 1), selection: 'frontline' },
      { ...group('reserve', 1), selection: 'rearline' },
    ]);
    const allocator = new GroupAllocator();

    const first = allocator.allocate(command, observationWith({ allies: roster }));
    const replay = allocator.allocate(command, observationWith({ allies: [...roster].reverse() }));

    expect(replay).toEqual(first);
    expect(first).toEqual([
      { role: 'main', unitIds: [2, 3] },
      { role: 'reserve', unitIds: [1, 4] },
    ]);
  });

  it('requires an anchor for nearest-objective selection and enough living units', () => {
    const nearest = plan([{ ...group('main', 1), selection: 'nearest_objective' }]);
    const observation = observationWith({ allies: units(2, -10) });
    expect(() => new GroupAllocator().allocate(nearest, observation)).toThrow(
      'group main requires an objective anchor',
    );

    const assigned = new GroupAllocator().allocate(nearest, observation, {
      objectiveAnchors: { main: { x: -9, y: 0 } },
    });
    expect(assigned[0].unitIds).toEqual([1, 2]);

    expect(() =>
      new GroupAllocator().allocate(
        plan([group('main', 1), group('reserve', 1)]),
        observationWith({ allies: units(1, -10) }),
      ),
    ).toThrow(GroupAllocationError);
  });
});

function plan(groups: readonly GroupCommand[]): CommandPlan {
  return { schemaVersion: COMMAND_PLAN_VERSION, intentSummary: null, groups };
}

function group(role: GroupCommand['role'], allocationWeight: number): GroupCommand {
  return {
    role,
    allocationWeight,
    selection: 'balanced',
    order: {
      mission: 'engage',
      objective: { kind: 'enemy_cluster', select: 'largest' },
      approach: 'direct',
      engagement: {
        posture: 'balanced',
        fire: 'focus',
        preferredRange: 'medium',
        cohesion: 'normal',
      },
    },
  };
}

function units(count: number, x: number): UnitObservation[] {
  return Array.from({ length: count }, (_, index) => unit(index + 1, x + index * 0.1, index));
}

function unit(id: number, x: number, y: number): UnitObservation {
  return {
    id,
    team: 'blue',
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
