import { describe, expect, it } from 'vitest';
import { createEmptyArena } from '../../src/game/Arena';
import { Team } from '../../src/game/types';
import { World } from '../../src/game/World';
import { observeWorld } from '../observations/Observation';

describe('observeWorld', () => {
  it('sorts entities and returns values detached from the mutable world', () => {
    const world = new World(createEmptyArena(), 7);
    const blue = world.addPlayer(Team.Player, -4, 1);
    const red = world.addPlayer(Team.Enemy, 5, 2);

    const observation = observeWorld(world, Team.Player, 12);
    blue.position.set(99, 99);
    red.health = 1;

    expect(observation.tick).toBe(12);
    expect(observation.allies.map((unit) => unit.id)).toEqual([blue.id]);
    expect(observation.enemies.map((unit) => unit.id)).toEqual([red.id]);
    expect(observation.allies[0]).toMatchObject({ x: -4, y: 1, health: 100 });
    expect(observation.enemies[0].health).toBe(100);
    expect(observation.match).toEqual({ blueAlive: 1, redAlive: 1 });
  });
});
