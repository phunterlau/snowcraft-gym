import { describe, expect, it } from 'vitest';
import type { UnitObservation } from '../observations/Observation';
import type { GroupCommand } from '../orchestration/command/CommandPlan';
import type { GroupAssignment } from '../orchestration/grounding/GroupAllocator';
import { TargetResolutionError, TargetResolver } from '../orchestration/grounding/TargetResolver';
import { observationWith } from './orchestrationTestHelpers';

describe('TargetResolver', () => {
  it('resolves dynamic enemy-cluster selectors with deterministic ties', () => {
    const observation = observationWith({
      allies: [unit(1, 'blue', -10, 0)],
      enemies: [
        unit(10, 'red', 4, 5, 100),
        unit(11, 'red', 5, 5, 100),
        unit(12, 'red', 7, -5, 30),
        unit(13, 'red', 8, -5, 30),
        unit(14, 'red', 9, -5, 30),
      ],
    });
    const resolver = new TargetResolver(2);

    expect(resolveEnemyIds(resolver, observation, 'nearest')).toEqual([10, 11]);
    expect(resolveEnemyIds(resolver, observation, 'largest')).toEqual([12, 13, 14]);
    expect(resolveEnemyIds(resolver, observation, 'weakest')).toEqual([12, 13, 14]);
    expect(resolveEnemyIds(resolver, observation, 'leftmost')).toEqual([10, 11]);
    expect(resolveEnemyIds(resolver, observation, 'rightmost')).toEqual([12, 13, 14]);
  });

  it('maps lanes relative to team facing rather than world axes', () => {
    const observation = observationWith({
      allies: [unit(1, 'blue', 0, -10)],
      enemies: [unit(10, 'red', 0, 10)],
    });
    const resolver = new TargetResolver();
    const left = resolver.resolve(regionGroup('left_lane'), observation);
    const right = resolver.resolve(regionGroup('right_lane'), observation);

    expect(left.anchor.x).toBeLessThan(0);
    expect(right.anchor.x).toBeGreaterThan(0);
    expect(Math.abs(left.anchor.y - right.anchor.y)).toBeLessThan(1e-9);
  });

  it('late-binds current-position and ally-group objectives from assignments', () => {
    const observation = observationWith({
      allies: [unit(1, 'blue', -9, 1), unit(2, 'blue', -7, 3)],
    });
    const assignments: GroupAssignment[] = [
      { role: 'main', unitIds: [1, 2] },
      { role: 'reserve', unitIds: [] },
    ];
    const resolver = new TargetResolver();

    expect(resolver.resolve(holdCurrentGroup(), observation, assignments)).toMatchObject({
      kind: 'current_position',
      anchor: { x: -8, y: 2 },
    });
    expect(resolver.resolve(supportGroup(), observation, assignments)).toMatchObject({
      kind: 'ally_group',
      role: 'main',
      anchor: { x: -8, y: 2 },
      unitIds: [1, 2],
    });
    expect(() => resolver.resolve(holdCurrentGroup(), observation)).toThrow(TargetResolutionError);
  });

  it('rejects enemy objectives after elimination', () => {
    const observation = observationWith({ enemies: [] });
    expect(() => new TargetResolver().resolve(engageGroup('largest'), observation)).toThrow(
      'no living enemy cluster exists',
    );
  });
});

function resolveEnemyIds(
  resolver: TargetResolver,
  observation: ReturnType<typeof observationWith>,
  selector: 'nearest' | 'largest' | 'weakest' | 'leftmost' | 'rightmost',
): readonly number[] {
  const objective = resolver.resolve(engageGroup(selector), observation);
  if (objective.kind !== 'enemy_cluster') throw new Error('expected enemy cluster');
  return objective.enemyIds;
}

function engageGroup(
  selector: 'nearest' | 'largest' | 'weakest' | 'leftmost' | 'rightmost',
): GroupCommand {
  return {
    ...baseGroup('main'),
    order: {
      mission: 'engage',
      objective: { kind: 'enemy_cluster', select: selector },
      approach: 'direct',
      engagement: engagement(),
    },
  };
}

function regionGroup(region: 'left_lane' | 'right_lane'): GroupCommand {
  return {
    ...baseGroup('main'),
    order: {
      mission: 'advance',
      objective: { kind: 'region', region },
      approach: 'direct',
      engagement: engagement(),
    },
  };
}

function holdCurrentGroup(): GroupCommand {
  return {
    ...baseGroup('main'),
    order: {
      mission: 'hold',
      objective: { kind: 'current_position' },
      approach: 'direct',
      engagement: engagement(),
    },
  };
}

function supportGroup(): GroupCommand {
  return {
    ...baseGroup('reserve'),
    order: {
      mission: 'support',
      objective: { kind: 'ally_group', role: 'main' },
      approach: 'direct',
      engagement: engagement(),
    },
  };
}

function baseGroup(role: GroupCommand['role']): Omit<GroupCommand, 'order'> {
  return { role, allocationWeight: 1, selection: 'balanced' };
}

function engagement(): GroupCommand['order']['engagement'] {
  return {
    posture: 'balanced',
    fire: 'focus',
    preferredRange: 'medium',
    cohesion: 'normal',
  };
}

function unit(
  id: number,
  team: UnitObservation['team'],
  x: number,
  y: number,
  health = 100,
): UnitObservation {
  return {
    id,
    team,
    x,
    y,
    vx: 0,
    vy: 0,
    health,
    maxHealth: 100,
    alive: true,
    state: 'idle',
    throwCooldown: 0,
    charge: 0,
  };
}
