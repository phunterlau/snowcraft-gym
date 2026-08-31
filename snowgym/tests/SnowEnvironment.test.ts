import { describe, expect, it } from 'vitest';
import { Team } from '../../src/game/types';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import type { TeamAction } from '../actions/UnitAction';
import { EpisodeCompleteError, SnowEnvironment, type StepResult } from '../core/SnowEnvironment';
import { createOpenScenario, THREE_VS_THREE_OPEN } from '../scenarios/Scenario';

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

  it('replays an exact blue action trace to the same public state hashes', () => {
    const first = recordActionTrace();
    const replay = replayActionTrace(first.actions);

    expect(replay).toEqual(first.hashes);
    expect(new Set(first.hashes).size).toBeGreaterThan(1);
  });

  it('truncates at the scenario tick limit and requires reset', () => {
    const scenario = createOpenScenario({ blueUnits: 1, redUnits: 1, maxTicks: 1 });
    const environment = new SnowEnvironment({ scenario });
    const observation = environment.reset(42);

    const result = environment.step({
      actions: observation.allies.map((unit) => ({ type: 'noop', unitId: unit.id })),
    });

    expect(result).toMatchObject({
      reward: 0,
      terminated: false,
      truncated: true,
      observation: { tick: 1 },
      info: { tick: 1, winner: null, terminated: false, truncated: true },
    });
    expect(() => environment.step({ actions: [] })).toThrow(EpisodeCompleteError);
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

function recordActionTrace(): { actions: TeamAction[]; hashes: string[] } {
  const environment = new SnowEnvironment();
  const policy = new SimpleBlueAgent();
  let observation = environment.reset(42);
  const actions: TeamAction[] = [];
  const hashes = [environment.status().stateHash];

  for (let decision = 0; decision < 20; decision++) {
    const action = policy.act(observation);
    actions.push(structuredClone(action));
    observation = environment.step(action).observation;
    hashes.push(environment.status().stateHash);
  }

  return { actions, hashes };
}

function replayActionTrace(actions: readonly TeamAction[]): string[] {
  const environment = new SnowEnvironment();
  environment.reset(42);
  const hashes = [environment.status().stateHash];

  for (const action of actions) {
    environment.step(structuredClone(action));
    hashes.push(environment.status().stateHash);
  }

  return hashes;
}
