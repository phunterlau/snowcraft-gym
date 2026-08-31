import { describe, expect, it } from 'vitest';
import type { AiDifficulty } from '../../src/systems/AISystem';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import { SnowEnvironment, type StepResult } from '../core/SnowEnvironment';
import { createOpenScenario, type Scenario } from '../scenarios/Scenario';

interface EpisodeSummary {
  tick: number;
  winner: 'blue' | 'red' | null;
  blueAlive: number;
  redAlive: number;
  health: number[];
  positions: number[];
}

describe('SnowEnvironment red controllers', () => {
  it('reproduces an episode exactly from the same seed and action policy', () => {
    const first = runEpisode({ seed: 42 });
    const second = runEpisode({ seed: 42 });
    expect(second).toEqual(first);
    expect(first.winner).not.toBeNull();
  });

  it('runs the random red controller deterministically from the scenario seed', () => {
    const first = runEpisode({ seed: 7, redController: 'random' });
    const second = runEpisode({ seed: 7, redController: 'random' });
    expect(second).toEqual(first);
    expect(first.winner).not.toBeNull();
  });

  it('produces a different episode for random versus scripted red at the same seed', () => {
    const scripted = runEpisode({ seed: 7, redController: 'scripted' });
    const random = runEpisode({ seed: 7, redController: 'random' });
    expect(random).not.toEqual(scripted);
  });

  it('exposes the selected red controller in the status configuration', () => {
    const scripted = new SnowEnvironment({ scenario: createOpenScenario({ seed: 3 }) });
    expect(scripted.status().configuration.redController).toBe('scripted');

    const random = new SnowEnvironment({
      scenario: createOpenScenario({ seed: 3 }),
      redController: 'random',
    });
    expect(random.status().configuration.redController).toBe('random');
  });

  it.each<[string, Scenario, AiDifficulty]>([
    ['3v3 normal', createOpenScenario({ seed: 42 }), 'normal'],
    ['3v3 hard', createOpenScenario({ seed: 42 }), 'hard'],
    [
      '5v2 hard wide arena',
      createOpenScenario({ seed: 7, blueUnits: 5, redUnits: 2, arenaWidth: 50, arenaHeight: 24 }),
      'hard',
    ],
    ['1v3 easy', createOpenScenario({ seed: 99, blueUnits: 1, redUnits: 3 }), 'easy'],
    ['1v1 normal', createOpenScenario({ seed: 1234, blueUnits: 1, redUnits: 1 }), 'normal'],
  ])('completes a deterministic scripted-red episode: %s', (_name, scenario, redDifficulty) => {
    const first = runEpisode({ scenario, redDifficulty });
    const second = runEpisode({ scenario, redDifficulty });
    expect(second).toEqual(first);
    expect(first.winner).not.toBeNull();
    expect(first.tick).toBeLessThanOrEqual(scenario.maxTicks);
  });
});

interface EpisodeOptions {
  seed?: number;
  scenario?: Scenario;
  redDifficulty?: AiDifficulty;
  redController?: 'scripted' | 'random';
}

function runEpisode(options: EpisodeOptions): EpisodeSummary {
  const scenario = options.scenario ?? createOpenScenario({ seed: options.seed ?? 0 });
  const environment = new SnowEnvironment({
    scenario,
    decisionHz: 10,
    redDifficulty: options.redDifficulty ?? 'normal',
    redController: options.redController ?? 'scripted',
  });
  const policy = new SimpleBlueAgent();
  let observation = environment.reset(scenario.seed);
  let result: StepResult | null = null;

  while (!environment.status().terminated && !environment.status().truncated) {
    result = environment.step(policy.act(observation));
    observation = result.observation;
  }

  const status = environment.status();
  const units = [...observation.allies, ...observation.enemies];
  expect(result).not.toBeNull();
  return {
    tick: status.tick,
    winner: status.winner,
    blueAlive: status.blueAlive,
    redAlive: status.redAlive,
    health: units.map((unit) => unit.health),
    positions: units.flatMap((unit) => [unit.x, unit.y]),
  };
}
