import { describe, expect, it } from 'vitest';
import { hasLineOfSight } from '../../src/physics/LineOfSight';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import { SnowEnvironment, type StepResult } from '../core/SnowEnvironment';
import { buildArena, createMapScenario, createOpenScenario } from '../scenarios/Scenario';
import { MAP_IDS, MAX_MAP_OBSTACLES } from '../scenarios/maps';
import { IdAllocator } from '../../src/ecs/Entity';

describe('map scenarios', () => {
  it('loads every bundled map with obstacles and both team spawns', () => {
    for (const mapId of MAP_IDS) {
      const scenario = createMapScenario(mapId, { seed: 1 });
      expect(scenario.map).toBe(mapId);
      expect(scenario.blueSpawns.length).toBeGreaterThan(0);
      expect(scenario.redSpawns.length).toBeGreaterThan(0);
      expect(scenario.mapData?.objects.length).toBeGreaterThan(0);
    }
    expect(MAX_MAP_OBSTACLES).toBeGreaterThan(0);
  });

  it('builds an obstacle-bearing arena that blocks line of sight', () => {
    const open = createOpenScenario({ seed: 1 });
    const openArena = buildArena(open, new IdAllocator());
    expect(openArena.obstacles).toHaveLength(0);

    const scenario = createMapScenario('arena1.json', { seed: 1 });
    const arena = buildArena(scenario, new IdAllocator());
    expect(arena.obstacles.length).toBeGreaterThan(0);
    expect(arena.width).toBe(40);
    expect(arena.height).toBe(30);

    // The central fort (0,0,5x1.2) blocks the straight east-west line through it.
    expect(hasLineOfSight(arena, -4, 0, 4, 0)).toBe(false);
    expect(hasLineOfSight(openArena, -4, 0, 4, 0)).toBe(true);
  });

  it('exposes obstacles in the observation in deterministic id order', () => {
    const environment = new SnowEnvironment({
      scenario: createMapScenario('arena4.json', { seed: 2 }),
    });
    const observation = environment.reset(2);
    expect(observation.obstacles.length).toBe(41);
    const ids = observation.obstacles.map((o) => o.id);
    expect([...ids].sort((a, b) => a - b)).toEqual(ids);
    expect(observation.obstacles[0]).toMatchObject({ type: expect.any(String) });
  });

  it('rejects an unknown map id', () => {
    expect(() => createMapScenario('arena99.json')).toThrow(RangeError);
  });

  it('runs a deterministic episode on a map', () => {
    const first = runOnMap('arena1.json', 42);
    const second = runOnMap('arena1.json', 42);
    expect(second).toEqual(first);
    expect(first.winner).not.toBeNull();
  });

  it('produces a different episode on terrain versus the open arena at the same seed', () => {
    const onMap = runOnMap('arena2.json', 11);
    const open = runOpen(11);
    expect(onMap).not.toEqual(open);
  });
});

interface MapSummary {
  tick: number;
  winner: 'blue' | 'red' | null;
  health: number[];
  obstacles: number;
}

function runOnMap(mapId: string, seed: number): MapSummary {
  const environment = new SnowEnvironment({ scenario: createMapScenario(mapId, { seed }) });
  return run(environment, seed);
}

function runOpen(seed: number): MapSummary {
  const scenario = createOpenScenario({
    seed,
    blueUnits: 3,
    redUnits: 3,
    arenaWidth: 44,
    arenaHeight: 30,
  });
  const environment = new SnowEnvironment({ scenario });
  return run(environment, seed);
}

function run(environment: SnowEnvironment, seed: number): MapSummary {
  const policy = new SimpleBlueAgent();
  let observation = environment.reset(seed);
  let result: StepResult | null = null;
  while (!environment.status().terminated && !environment.status().truncated) {
    result = environment.step(policy.act(observation));
    observation = result.observation;
  }
  const status = environment.status();
  expect(result).not.toBeNull();
  return {
    tick: status.tick,
    winner: status.winner,
    health: [...observation.allies, ...observation.enemies].map((u) => u.health),
    obstacles: observation.obstacles.length,
  };
}
