import type { TeamAction } from '../actions/UnitAction';
import type { Observation } from '../observations/Observation';

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
  };
  frames: Observation[];
  actions: TeamAction[];
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
  }

  const actions = array(replay.actions, 'actions');
  if (actions.length !== frames.length - 1) {
    throw new ReplayFormatError('actions must contain one entry per frame transition');
  }
  const outcome = record(replay.outcome, 'outcome');
  if (integer(outcome.finalTick, 'outcome.finalTick') !== previousTick) {
    throw new ReplayFormatError('outcome.finalTick must match the final frame');
  }
  positive(replay.simulationHz, 'simulationHz');
  positive(replay.decisionHz, 'decisionHz');
  positive(replay.ticksPerDecision, 'ticksPerDecision');
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
