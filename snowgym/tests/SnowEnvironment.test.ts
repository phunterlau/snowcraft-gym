import { describe, expect, it } from 'vitest';
import { Team } from '../../src/game/types';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import { SnowEnvironment, type StepResult } from '../core/SnowEnvironment';
import { THREE_VS_THREE_OPEN } from '../scenarios/Scenario';

describe('SnowEnvironment', () => {
  it('advances one 10 Hz decision as six 60 Hz physics ticks', () => {
    const environment = new SnowEnvironment({ decisionHz: 10 });
    const observation = environment.reset(42);
    const action = new SimpleBlueAgent().act(observation);

    const result = environment.step(action);

    expect(result.observation.tick).toBe(6);
    expect(result.info).toMatchObject({
      seed: 42,
      simulationHz: 60,
      decisionHz: 10,
      ticksPerDecision: 6,
      terminated: false,
      truncated: false,
    });
    expect(result.reward).toBe(0);
  });

  it('completes and exactly replays autonomous 3v3 without DOM or input', () => {
    const first = runBattle();
    const replay = runBattle();

    expect(first).toEqual(replay);
    expect(first.winner).not.toBeNull();
    expect(first.tick).toBeLessThanOrEqual(THREE_VS_THREE_OPEN.maxTicks);
    expect(first.blueUnits).toBe(3);
    expect(first.redUnits).toBe(3);
    expect(first.finalReward).toBe(first.winner === 'blue' ? 1 : -1);
  });

  it('returns observations for either team without changing the world', () => {
    const environment = new SnowEnvironment();
    const blue = environment.observe(Team.Player);
    const red = environment.observe(Team.Enemy);

    expect(blue.allies.map((unit) => unit.id)).toEqual(red.enemies.map((unit) => unit.id));
    expect(blue.enemies.map((unit) => unit.id)).toEqual(red.allies.map((unit) => unit.id));
    expect(environment.status().tick).toBe(0);
  });
});

function runBattle(): BattleSummary {
  const environment = new SnowEnvironment();
  const policy = new SimpleBlueAgent();
  let observation = environment.reset(THREE_VS_THREE_OPEN.seed);
  let finalResult: StepResult | null = null;

  while (!environment.status().terminated && !environment.status().truncated) {
    finalResult = environment.step(policy.act(observation));
    observation = finalResult.observation;
  }

  const status = environment.status();
  return {
    tick: status.tick,
    winner: status.winner,
    blueUnits: observation.allies.length,
    redUnits: observation.enemies.length,
    finalReward: finalResult?.reward ?? 0,
    health: [...observation.allies, ...observation.enemies].map((unit) => unit.health),
  };
}

interface BattleSummary {
  tick: number;
  winner: 'blue' | 'red' | null;
  blueUnits: number;
  redUnits: number;
  finalReward: number;
  health: number[];
}
