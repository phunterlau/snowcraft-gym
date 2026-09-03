import type { MapData } from '../../src/game/types';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import { SnowEnvironment } from '../core/SnowEnvironment';
import {
  parseReplayRecording,
  REPLAY_FORMAT,
  type ReplayRecording,
} from '../replay/ReplayRecording';
import { createGeneratedMapScenario } from './GeneratedScenario';
import { canonicalizeMap, digestJson } from './MapValidator';
import {
  MAP_EVALUATION_VERSION,
  type MapEvaluationEpisode,
  type MapEvaluationReport,
} from './types';

export interface MapEvaluationOptions {
  seeds?: number[];
  maxTicks?: number;
  decisionHz?: number;
  includeSwappedSpawns?: boolean;
  replaySeed?: number;
}

export interface EvaluatedMap {
  report: MapEvaluationReport;
  replay?: ReplayRecording;
}

/** Runs deterministic renderer-free policy probes; balance metrics are descriptive, not an acceptance gate. */
export function evaluateGeneratedMap(
  map: MapData,
  options: MapEvaluationOptions = {},
): EvaluatedMap {
  const seeds = options.seeds ?? [41, 42, 43];
  if (seeds.length === 0 || seeds.some((seed) => !Number.isSafeInteger(seed))) {
    throw new RangeError('seeds must be a non-empty array of safe integers');
  }
  const swaps = options.includeSwappedSpawns === false ? [false] : [false, true];
  const episodes: MapEvaluationEpisode[] = [];
  let replay: ReplayRecording | undefined;
  for (const seed of seeds) {
    for (const swappedSpawns of swaps) {
      const result = runEpisode(map, seed, swappedSpawns, options);
      episodes.push(result.episode);
      if (seed === (options.replaySeed ?? seeds[0]) && !swappedSpawns) replay = result.replay;
    }
  }
  const terminated = episodes.filter((episode) => episode.terminated).length;
  const blueWins = episodes.filter((episode) => episode.winner === 'blue').length;
  const redWins = episodes.filter((episode) => episode.winner === 'red').length;
  const normalBlueWins = episodes.filter(
    (episode) => !episode.swappedSpawns && episode.winner === 'blue',
  ).length;
  const swappedBlueWins = episodes.filter(
    (episode) => episode.swappedSpawns && episode.winner === 'blue',
  ).length;
  const sideDenominator = Math.max(1, episodes.filter((episode) => !episode.swappedSpawns).length);
  return {
    report: {
      schemaVersion: MAP_EVALUATION_VERSION,
      mapDigest: digestJson(canonicalizeMap(map)),
      episodes,
      summary: {
        episodeCount: episodes.length,
        terminationRate: terminated / episodes.length,
        blueWinRate: blueWins / episodes.length,
        redWinRate: redWins / episodes.length,
        meanTicks: episodes.reduce((sum, episode) => sum + episode.ticks, 0) / episodes.length,
        sideAdvantage: (normalBlueWins - swappedBlueWins) / sideDenominator,
      },
    },
    replay,
  };
}

function runEpisode(
  map: MapData,
  seed: number,
  swappedSpawns: boolean,
  options: MapEvaluationOptions,
): { episode: MapEvaluationEpisode; replay: ReplayRecording } {
  const scenario = createGeneratedMapScenario(map, {
    seed,
    swappedSpawns,
    maxTicks: options.maxTicks,
  });
  const environment = new SnowEnvironment({ scenario, decisionHz: options.decisionHz });
  const policy = new SimpleBlueAgent();
  let observation = environment.reset(seed);
  let status = environment.status();
  const frames = [observation];
  const actions = [];
  const stateHashes = [status.stateHash];
  let rejectedActions = 0;
  while (!status.terminated && !status.truncated) {
    const action = policy.act(observation);
    const result = environment.step(action);
    rejectedActions += result.info.actionResults.filter((item) => !item.accepted).length;
    actions.push(action);
    observation = result.observation;
    frames.push(observation);
    status = environment.status();
    stateHashes.push(status.stateHash);
  }
  const replay = parseReplayRecording({
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
  return {
    episode: {
      seed,
      swappedSpawns,
      decisions: actions.length,
      ticks: status.tick,
      terminated: status.terminated,
      truncated: status.truncated,
      winner: status.winner,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      rejectedActions,
      finalStateHash: status.stateHash,
    },
    replay,
  };
}
