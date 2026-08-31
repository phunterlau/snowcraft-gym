import { describe, expect, it } from 'vitest';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import { SnowEnvironment } from '../core/SnowEnvironment';
import { createMapScenario, createOpenScenario, MAX_TEAM_SIZE } from '../scenarios/Scenario';

describe('configurable SnowGym scenarios', () => {
  it.each([
    [1, 1],
    [1, 3],
    [3, 1],
    [3, 3],
    [MAX_TEAM_SIZE, MAX_TEAM_SIZE],
  ])('creates a deterministic non-overlapping %iv%i roster', (blueUnits, redUnits) => {
    const first = createOpenScenario({ blueUnits, redUnits });
    const second = createOpenScenario({ blueUnits, redUnits });

    expect(first).toEqual(second);
    expect(first.blueSpawns).toHaveLength(blueUnits);
    expect(first.redSpawns).toHaveLength(redUnits);
    expect(allPairDistances([...first.blueSpawns, ...first.redSpawns])).toBeGreaterThanOrEqual(1);
    const environment = new SnowEnvironment({ scenario: first });
    const observation = environment.reset(17);
    expect(observation.allies).toHaveLength(blueUnits);
    expect(observation.enemies).toHaveLength(redUnits);
  });

  it('supports custom arena, cadence, timeout, difficulty, and seeded autonomous completion', () => {
    const scenario = createOpenScenario({
      blueUnits: 3,
      redUnits: 1,
      arenaWidth: 50,
      arenaHeight: 20,
      maxTicks: 1_200,
    });
    const summary = runBattle(scenario, 20, 'hard', 7);
    const replay = runBattle(scenario, 20, 'hard', 7);

    expect(summary).toEqual(replay);
    expect(summary.tick).toBeLessThanOrEqual(1_200);
    expect(summary.terminated || summary.truncated).toBe(true);
  });

  it('selects smaller map rosters evenly from native spawn points', () => {
    const scenario = createMapScenario('arena6.json', { blueUnits: 3, redUnits: 2 });

    expect(scenario.blueSpawns).toEqual([
      { x: -29, y: -18 },
      { x: -28, y: 2 },
      { x: -28, y: 18 },
    ]);
    expect(scenario.redSpawns).toEqual([
      { x: 29, y: -18 },
      { x: 28, y: 18 },
    ]);
  });

  it('rejects a map roster above its native spawn capacity', () => {
    expect(() => createMapScenario('arena1.json', { blueUnits: 4 })).toThrow(
      'blueUnits must be at most 3 on map "arena1.json"',
    );
  });

  it.each([
    [{ blueUnits: 0 }, 'blueUnits must be positive'],
    [{ redUnits: MAX_TEAM_SIZE + 1 }, `redUnits must be at most ${MAX_TEAM_SIZE}`],
    [{ arenaWidth: 10 }, 'arenaWidth must be in [12, 120]'],
    [{ blueUnits: 2, blueSpawns: [{ x: -1, y: 0 }] }, 'blueSpawns must contain 2 positions'],
    [
      {
        blueUnits: 2,
        blueSpawns: [
          { x: -1, y: 0 },
          { x: -1, y: 0 },
        ],
      },
      'blueSpawns positions must not overlap',
    ],
  ])('rejects invalid configuration %#', (options, message) => {
    expect(() => createOpenScenario(options)).toThrow(message);
  });
});

function allPairDistances(points: ReadonlyArray<{ x: number; y: number }>): number {
  let minimum = Number.POSITIVE_INFINITY;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      minimum = Math.min(minimum, Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y));
    }
  }
  return minimum;
}

function runBattle(
  scenario: ReturnType<typeof createOpenScenario>,
  decisionHz: number,
  redDifficulty: 'easy' | 'normal' | 'hard',
  seed: number,
): { tick: number; winner: string | null; terminated: boolean; truncated: boolean } {
  const environment = new SnowEnvironment({ scenario, decisionHz, redDifficulty });
  const policy = new SimpleBlueAgent();
  let observation = environment.reset(seed);
  while (!environment.status().terminated && !environment.status().truncated) {
    observation = environment.step(policy.act(observation)).observation;
  }
  const status = environment.status();
  return {
    tick: status.tick,
    winner: status.winner,
    terminated: status.terminated,
    truncated: status.truncated,
  };
}
