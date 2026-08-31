import type { TeamAction } from '../actions/UnitAction';
import type { Observation } from '../observations/Observation';
import { STATE_HASH_VERSION } from '../protocol/Version';
import { hashObservation } from '../reproducibility/StateHash';

export const REPLAY_FORMAT = 'snowgym.replay.v0' as const;

export interface ReplayOutcome {
  decisions: number;
  terminated: boolean;
  truncated: boolean;
  winner: 'blue' | 'red' | null;
  blueAlive: number;
  redAlive: number;
  finalTick: number;
}

export interface ReplayRecording {
  format: typeof REPLAY_FORMAT;
  apiVersion: 'snowgym.v0';
  simulationVersion?: string;
  stateHashVersion?: typeof STATE_HASH_VERSION;
  upstreamBaseCommit?: string;
  scenario: string;
  seed: number;
  simulationHz: number;
  decisionHz: number;
  ticksPerDecision: number;
  configuration?: {
    blueUnits: number;
    redUnits: number;
    arenaWidth: number;
    arenaHeight: number;
    maxTicks: number;
    redDifficulty: 'easy' | 'normal' | 'hard';
    redController?: 'scripted' | 'random';
    map?: string | null;
  };
  frames: Observation[];
  actions: TeamAction[];
  stateHashes?: string[];
  outcome: ReplayOutcome;
}

/** Validates untrusted replay JSON before it reaches the render-only world. */
export function parseReplayRecording(value: unknown): ReplayRecording {
  const replay = record(value, 'replay');
  if (replay.format !== REPLAY_FORMAT) {
    throw new ReplayFormatError(`expected format ${REPLAY_FORMAT}`);
  }
  if (replay.apiVersion !== 'snowgym.v0') {
    throw new ReplayFormatError('expected API snowgym.v0');
  }

  const frames = array(replay.frames, 'frames');
  if (frames.length === 0) throw new ReplayFormatError('frames must not be empty');
  let previousTick = -1;
  for (const [index, value] of frames.entries()) {
    const frame = record(value, `frames[${index}]`);
    const tick = integer(frame.tick, `frames[${index}].tick`);
    if (tick <= previousTick) throw new ReplayFormatError('frame ticks must increase');
    previousTick = tick;
    const arena = record(frame.arena, `frames[${index}].arena`);
    positive(arena.width, `frames[${index}].arena.width`);
    positive(arena.height, `frames[${index}].arena.height`);
    array(frame.allies, `frames[${index}].allies`);
    array(frame.enemies, `frames[${index}].enemies`);
    array(frame.projectiles, `frames[${index}].projectiles`);
    array(frame.obstacles ?? [], `frames[${index}].obstacles`);
  }

  const actions = array(replay.actions, 'actions');
  if (actions.length !== frames.length - 1) {
    throw new ReplayFormatError('actions must contain one entry per frame transition');
  }
  if (replay.stateHashes !== undefined) {
    const stateHashes = array(replay.stateHashes, 'stateHashes');
    if (stateHashes.length !== frames.length) {
      throw new ReplayFormatError('stateHashes must contain one entry per frame');
    }
    for (const [index, hash] of stateHashes.entries()) {
      if (typeof hash !== 'string' || !/^fnv1a64:[0-9a-f]{16}$/.test(hash)) {
        throw new ReplayFormatError(`stateHashes[${index}] is invalid`);
      }
      if (hash !== hashObservation(frames[index] as Observation)) {
        throw new ReplayFormatError(`stateHashes[${index}] does not match its frame`);
      }
    }
  }
  const outcome = record(replay.outcome, 'outcome');
  if (integer(outcome.finalTick, 'outcome.finalTick') !== previousTick) {
    throw new ReplayFormatError('outcome.finalTick must match the final frame');
  }
  positive(replay.simulationHz, 'simulationHz');
  positive(replay.decisionHz, 'decisionHz');
  positive(replay.ticksPerDecision, 'ticksPerDecision');
  optionalNonEmptyString(replay.simulationVersion, 'simulationVersion');
  optionalNonEmptyString(replay.upstreamBaseCommit, 'upstreamBaseCommit');
  if (replay.stateHashVersion !== undefined && replay.stateHashVersion !== STATE_HASH_VERSION) {
    throw new ReplayFormatError(`expected state hash version ${STATE_HASH_VERSION}`);
  }
  if (replay.configuration !== undefined) {
    const configuration = record(replay.configuration, 'configuration');
    positive(configuration.blueUnits, 'configuration.blueUnits');
    positive(configuration.redUnits, 'configuration.redUnits');
    positive(configuration.arenaWidth, 'configuration.arenaWidth');
    positive(configuration.arenaHeight, 'configuration.arenaHeight');
    positive(configuration.maxTicks, 'configuration.maxTicks');
    if (!['easy', 'normal', 'hard'].includes(String(configuration.redDifficulty))) {
      throw new ReplayFormatError('configuration.redDifficulty is invalid');
    }
    if (
      configuration.map !== undefined &&
      configuration.map !== null &&
      (typeof configuration.map !== 'string' || configuration.map.length === 0)
    ) {
      throw new ReplayFormatError('configuration.map is invalid');
    }
  }
  integer(replay.seed, 'seed');
  if (typeof replay.scenario !== 'string' || replay.scenario.length === 0) {
    throw new ReplayFormatError('scenario must be a non-empty string');
  }

  return replay as unknown as ReplayRecording;
}

export class ReplayFormatError extends Error {}

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ReplayFormatError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new ReplayFormatError(`${name} must be an array`);
  return value;
}

function integer(value: unknown, name: string): number {
  if (!Number.isSafeInteger(value)) throw new ReplayFormatError(`${name} must be an integer`);
  return value as number;
}

function positive(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new ReplayFormatError(`${name} must be positive`);
  }
  return value;
}

function optionalNonEmptyString(value: unknown, name: string): void {
  if (value !== undefined && (typeof value !== 'string' || value.length === 0)) {
    throw new ReplayFormatError(`${name} must be a non-empty string`);
  }
}
