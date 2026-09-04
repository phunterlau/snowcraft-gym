import { describe, expect, it } from 'vitest';
import { createEmptyArena } from '../../src/game/Arena';
import { Team } from '../../src/game/types';
import { World } from '../../src/game/World';
import { Vector2 } from '../../src/utils/Vector2';
import { observeWorld } from '../observations/Observation';

describe('observeWorld', () => {
  it('sorts entities and returns values detached from the mutable world', () => {
    const world = new World(createEmptyArena(), 7);
    const blue = world.addPlayer(Team.Player, -4, 1);
    const red = world.addPlayer(Team.Enemy, 5, 2);
    blue.moveTarget = new Vector2(3, -2);
    blue.aimDirection.set(0.6, 0.8);
    blue.stunTimer = 0.25;
    blue.immunityTimer = 1.5;
    blue.speedTimer = 2.5;

    const observation = observeWorld(world, Team.Player, 12, {
      decisionHz: 10,
      maxTicks: 120,
      steeringTarget: (player) => (player.id === blue.id ? { x: 1, y: -1 } : null),
    });
    blue.position.set(99, 99);
    red.health = 1;

    expect(observation.tick).toBe(12);
    expect(observation.allies.map((unit) => unit.id)).toEqual([blue.id]);
    expect(observation.enemies.map((unit) => unit.id)).toEqual([red.id]);
    expect(observation.allies[0]).toMatchObject({ x: -4, y: 1, health: 100 });
    expect(observation.allies[0]).toMatchObject({
      moveTarget: { x: 3, y: -2 },
      steeringTarget: { x: 1, y: -1 },
      aimDirection: { x: 0.6, y: 0.8 },
      stunRemaining: 0.25,
      immunityRemaining: 1.5,
      speedRemaining: 2.5,
    });
    expect(observation.decision).toEqual({
      hz: 10,
      dt: 0.1,
      maxTicks: 120,
      remainingFraction: 0.9,
    });
    expect(observation.enemies[0].health).toBe(100);
    expect(observation.match).toEqual({ blueAlive: 1, redAlive: 1 });
  });
});
