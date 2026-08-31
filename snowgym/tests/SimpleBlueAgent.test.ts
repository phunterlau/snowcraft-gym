import { describe, expect, it } from 'vitest';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import type { Observation, SnowTeam, UnitObservation } from '../observations/Observation';

describe('SimpleBlueAgent', () => {
  it('throws at the nearest living opponent without referring to engine entities', () => {
    const policy = new SimpleBlueAgent();
    const observation = makeObservation();
    observation.enemies.push(unit(20, 'red', 7, 0), unit(21, 'red', 5, 0));

    expect(policy.act(observation).actions).toEqual([
      { type: 'throw', unitId: 10, x: 5, y: 0, power: 0.52 },
    ]);
  });

  it('moves perpendicular to an approaching hostile projectile before attacking', () => {
    const policy = new SimpleBlueAgent();
    const observation = makeObservation();
    observation.enemies.push(unit(20, 'red', 5, 0));
    observation.projectiles.push({
      id: 30,
      ownerId: 20,
      team: 'red',
      x: -2,
      y: 0,
      vx: 10,
      vy: 0,
      height: 1,
      heightVelocity: 0,
    });

    expect(policy.act(observation).actions[0]).toEqual({ type: 'move', unitId: 10, x: 0, y: 2.4 });
  });
});

function makeObservation(): Observation {
  return {
    tick: 1,
    selfTeam: 'blue',
    simulationHz: 60,
    arena: { width: 40, height: 30 },
    allies: [unit(10, 'blue', 0, 0)],
    enemies: [],
    projectiles: [],
    match: { blueAlive: 1, redAlive: 0 },
  };
}

function unit(id: number, team: SnowTeam, x: number, y: number): UnitObservation {
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
