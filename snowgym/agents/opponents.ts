import { type Team } from '../../src/game/types';
import type { World } from '../../src/game/World';
import type { AiDifficulty } from '../../src/systems/AISystem';
import type { ThrowSystem } from '../../src/systems/ThrowSystem';
import { RandomAgent } from './RandomAgent';
import { ScriptedAiAgent } from './ScriptedAiAgent';
import type { TeamController } from './TeamController';

/** Opponent behaviors selectable for the red team. */
export type RedControllerType = 'scripted' | 'random';

export interface RedControllerOptions {
  type?: RedControllerType;
  difficulty?: AiDifficulty;
}

export const DEFAULT_RED_CONTROLLER: RedControllerType = 'scripted';
export const RED_CONTROLLER_TYPES: readonly RedControllerType[] = ['scripted', 'random'];

export function parseRedControllerType(value: unknown): RedControllerType {
  if (value === undefined) return DEFAULT_RED_CONTROLLER;
  if (value === 'scripted' || value === 'random') return value;
  throw new RangeError(`redController must be one of: ${RED_CONTROLLER_TYPES.join(', ')}`);
}

/**
 * Builds the red-team policy. Every option runs behind the same
 * TeamController boundary: the scripted opponent is the classic squad AI with
 * per-tick reactive behavior; random is a seeded baseline.
 */
export function createRedController(
  options: RedControllerOptions,
  world: World,
  throwSystem: ThrowSystem,
  redTeam: Team,
  blueTeam: Team,
  difficulty: AiDifficulty,
): TeamController {
  const type = options.type ?? DEFAULT_RED_CONTROLLER;
  if (type === 'random') return new RandomAgent(world);
  return new ScriptedAiAgent(
    world,
    throwSystem,
    redTeam,
    blueTeam,
    options.difficulty ?? difficulty,
  );
}
