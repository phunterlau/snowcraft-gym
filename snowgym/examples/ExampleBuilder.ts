import type { AiDifficulty } from '../../src/systems/AISystem';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import type { RedControllerType } from '../agents/opponents';
import { SnowEnvironment } from '../core/SnowEnvironment';
import {
  parseReplayRecording,
  REPLAY_FORMAT,
  type ReplayRecording,
} from '../replay/ReplayRecording';
import {
  createMapScenario,
  createOpenScenario,
  DEFAULT_MAX_TICKS,
  type Scenario,
} from '../scenarios/Scenario';

export interface ReplayExampleOptions {
  blueUnits: number;
  redUnits: number;
  map?: string;
  seed?: number;
  arenaWidth?: number;
  arenaHeight?: number;
  maxTicks?: number;
  maxDecisions?: number;
  decisionHz?: number;
  redDifficulty?: AiDifficulty;
  redController?: RedControllerType;
}

/** Builds a complete, deterministic, renderer-free scripted-blue replay. */
export function buildReplayExample(options: ReplayExampleOptions): ReplayRecording {
  const seed = options.seed ?? 42;
  const maxTicks = options.maxTicks ?? DEFAULT_MAX_TICKS;
  const scenario = buildScenario(options, seed, maxTicks);
  const environment = new SnowEnvironment({
    scenario,
    decisionHz: options.decisionHz,
    redDifficulty: options.redDifficulty,
    redController: options.redController,
  });
  const policy = new SimpleBlueAgent();
  let observation = environment.reset(seed);
  let status = environment.status();
  const frames = [observation];
  const actions = [];
  const stateHashes = [status.stateHash];
  const maxDecisions = positiveInteger(options.maxDecisions ?? 10_000, 'maxDecisions');

  while (!status.terminated && !status.truncated && actions.length < maxDecisions) {
    const action = policy.act(observation);
    const result = environment.step(action);
    actions.push(action);
    observation = result.observation;
    frames.push(observation);
    status = environment.status();
    stateHashes.push(status.stateHash);
  }
  if (!status.terminated && !status.truncated) {
    throw new RangeError(`episode did not complete within ${maxDecisions} decisions`);
  }

  return parseReplayRecording({
    format: REPLAY_FORMAT,
    apiVersion: status.apiVersion,
    simulationVersion: status.simulationVersion,
    stateHashVersion: status.stateHashVersion,
    upstreamBaseCommit: status.upstreamBaseCommit,
    scenario: status.scenario,
    seed: status.seed,
    simulationHz: status.simulationHz,
    decisionHz: status.decisionHz,
    ticksPerDecision: status.ticksPerDecision,
    configuration: status.configuration,
    frames,
    actions,
    stateHashes,
    outcome: {
      decisions: actions.length,
      terminated: status.terminated,
      truncated: status.truncated,
      winner: status.winner,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      finalTick: status.tick,
    },
  });
}

function buildScenario(options: ReplayExampleOptions, seed: number, maxTicks: number): Scenario {
  if (options.map && options.map !== 'open') {
    if (options.arenaWidth !== undefined || options.arenaHeight !== undefined) {
      throw new RangeError('arenaWidth and arenaHeight apply only to map "open"');
    }
    return createMapScenario(options.map, {
      seed,
      maxTicks,
      blueUnits: options.blueUnits,
      redUnits: options.redUnits,
    });
  }
  return createOpenScenario({
    seed,
    maxTicks,
    blueUnits: options.blueUnits,
    redUnits: options.redUnits,
    arenaWidth: options.arenaWidth,
    arenaHeight: options.arenaHeight,
  });
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
}
