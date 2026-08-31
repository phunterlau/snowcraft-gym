import { Team } from '../../src/game/types';
import type { AiDifficulty } from '../../src/systems/AISystem';
import type { TeamAction, UnitAction } from '../actions/UnitAction';
import { parseRedControllerType, type RedControllerType } from '../agents/opponents';
import { SimpleBlueAgent } from '../agents/SimpleBlueAgent';
import { EpisodeCompleteError, SnowEnvironment, type StepResult } from '../core/SnowEnvironment';
import {
  createMapScenario,
  createOpenScenario,
  type OpenScenarioOptions,
  type SpawnPosition,
} from '../scenarios/Scenario';
import { isMapId, MAP_IDS } from '../scenarios/maps';

export interface ServiceResponse {
  status: number;
  body: unknown;
}

/** Transport-independent request handler used by the HTTP server and tests. */
export class SnowGymService {
  private readonly bluePolicy = new SimpleBlueAgent();
  private environment: SnowEnvironment;

  constructor(environment = new SnowEnvironment()) {
    this.environment = environment;
  }

  handle(method: string, path: string, body: unknown = undefined): ServiceResponse {
    try {
      if (method === 'GET' && path === '/health') {
        return { status: 200, body: { ok: true } };
      }
      if (method === 'GET' && path === '/status') {
        return { status: 200, body: this.snapshot() };
      }
      if (method === 'POST' && path === '/reset') {
        const request = parseReset(body);
        if (request.map !== undefined) {
          this.environment = new SnowEnvironment({
            scenario: createMapScenario(request.map, { seed: request.seed }),
            decisionHz: request.decisionHz,
            redDifficulty: request.redDifficulty,
            redController: request.redController,
          });
        } else if (request.scenario) {
          const scenario = createOpenScenario({
            ...request.scenario,
            seed: request.seed,
          });
          this.environment = new SnowEnvironment({
            scenario,
            decisionHz: request.decisionHz,
            redDifficulty: request.redDifficulty,
            redController: request.redController,
          });
        } else {
          this.environment.reset(request.seed);
        }
        return { status: 200, body: this.snapshot() };
      }
      if (method === 'POST' && path === '/step') {
        const action = parseOptionalAction(body) ?? this.defaultBlueAction();
        const result = this.environment.step(action);
        return { status: 200, body: { ...result, info: { ...result.info, action } } };
      }
      if (method === 'POST' && path === '/autoplay') {
        return { status: 200, body: this.autoplay(parseMaxDecisions(body)) };
      }
      return { status: 404, body: { error: 'not_found' } };
    } catch (error) {
      if (error instanceof EpisodeCompleteError) {
        return { status: 409, body: { error: 'episode_complete', message: error.message } };
      }
      if (error instanceof RequestValidationError || error instanceof RangeError) {
        return { status: 400, body: { error: 'invalid_request', message: error.message } };
      }
      throw error;
    }
  }

  private snapshot(): object {
    return {
      status: this.environment.status(),
      observation: this.environment.observe(Team.Player),
    };
  }

  private defaultBlueAction(): TeamAction {
    return this.bluePolicy.act(this.environment.observe(Team.Player));
  }

  private autoplay(maxDecisions: number): object {
    let decisions = 0;
    let result: StepResult | null = null;
    while (decisions < maxDecisions) {
      const status = this.environment.status();
      if (status.terminated || status.truncated) break;
      result = this.environment.step(this.defaultBlueAction());
      decisions++;
    }
    return { decisions, result, ...this.snapshot() };
  }
}

class RequestValidationError extends Error {}

interface ResetRequest {
  seed?: number;
  scenario?: OpenScenarioOptions;
  map?: string;
  decisionHz?: number;
  redDifficulty?: AiDifficulty;
  redController?: RedControllerType;
}

function parseReset(body: unknown): ResetRequest {
  if (body === undefined || body === null) return {};
  const record = asRecord(body);
  const seed = optionalSafeInteger(record.seed, 'seed');
  if (record.scenario === undefined) return { seed };

  const scenario = asRecord(record.scenario);
  const allowed = new Set([
    'blueUnits',
    'redUnits',
    'arenaWidth',
    'arenaHeight',
    'blueSpawns',
    'redSpawns',
    'maxTicks',
    'decisionHz',
    'redDifficulty',
    'redController',
    'map',
  ]);
  const unknown = Object.keys(scenario).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw new RequestValidationError(`unknown scenario fields: ${unknown.sort().join(', ')}`);
  }

  const redDifficulty = optionalDifficulty(scenario.redDifficulty);
  let redController: RedControllerType | undefined;
  try {
    redController = parseRedControllerType(scenario.redController);
  } catch (error) {
    throw new RequestValidationError((error as RangeError).message);
  }

  if (scenario.map !== undefined) {
    if (!isMapId(scenario.map)) {
      throw new RequestValidationError(`scenario.map must be one of: ${MAP_IDS.join(', ')}`);
    }
    const conflicting = [
      'blueUnits',
      'redUnits',
      'arenaWidth',
      'arenaHeight',
      'blueSpawns',
      'redSpawns',
    ].filter((key) => scenario[key] !== undefined);
    if (conflicting.length > 0) {
      throw new RequestValidationError(
        `scenario.map fixes the terrain and rosters; remove: ${conflicting.sort().join(', ')}`,
      );
    }
    return {
      seed,
      map: scenario.map,
      decisionHz: optionalSafeInteger(scenario.decisionHz, 'scenario.decisionHz'),
      redDifficulty,
      redController,
    };
  }

  return {
    seed,
    decisionHz: optionalSafeInteger(scenario.decisionHz, 'scenario.decisionHz'),
    redDifficulty,
    redController: scenario.redController === undefined ? undefined : redController,
    scenario: {
      blueUnits: optionalSafeInteger(scenario.blueUnits, 'scenario.blueUnits'),
      redUnits: optionalSafeInteger(scenario.redUnits, 'scenario.redUnits'),
      arenaWidth: optionalFiniteNumber(scenario.arenaWidth, 'scenario.arenaWidth'),
      arenaHeight: optionalFiniteNumber(scenario.arenaHeight, 'scenario.arenaHeight'),
      blueSpawns: parseOptionalSpawns(scenario.blueSpawns, 'scenario.blueSpawns'),
      redSpawns: parseOptionalSpawns(scenario.redSpawns, 'scenario.redSpawns'),
      maxTicks: optionalSafeInteger(scenario.maxTicks, 'scenario.maxTicks'),
    },
  };
}

function parseOptionalAction(body: unknown): TeamAction | null {
  if (body === undefined || body === null) return null;
  const record = asRecord(body);
  if (record.action === undefined) return null;
  const actionRecord = asRecord(record.action);
  if (!Array.isArray(actionRecord.actions)) {
    throw new RequestValidationError('action.actions must be an array');
  }
  return { actions: actionRecord.actions.map(parseUnitAction) };
}

function parseUnitAction(value: unknown): UnitAction {
  const action = asRecord(value);
  if (!Number.isSafeInteger(action.unitId)) {
    throw new RequestValidationError('unitId must be a safe integer');
  }
  const unitId = action.unitId as number;
  if (action.type === 'noop') return { type: 'noop', unitId };
  if (action.type !== 'move' && action.type !== 'throw') {
    throw new RequestValidationError('action type must be noop, move, or throw');
  }
  const x = finiteNumber(action.x, 'x');
  const y = finiteNumber(action.y, 'y');
  if (action.type === 'move') return { type: 'move', unitId, x, y };
  return { type: 'throw', unitId, x, y, power: finiteNumber(action.power, 'power') };
}

function parseMaxDecisions(body: unknown): number {
  if (body === undefined || body === null) return 10_000;
  const record = asRecord(body);
  if (record.maxDecisions === undefined) return 10_000;
  if (!Number.isSafeInteger(record.maxDecisions) || (record.maxDecisions as number) <= 0) {
    throw new RequestValidationError('maxDecisions must be a positive safe integer');
  }
  return Math.min(record.maxDecisions as number, 10_000);
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RequestValidationError(`${name} must be a finite number`);
  }
  return value;
}

function optionalFiniteNumber(value: unknown, name: string): number | undefined {
  return value === undefined ? undefined : finiteNumber(value, name);
}

function optionalSafeInteger(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value)) {
    throw new RequestValidationError(`${name} must be a safe integer`);
  }
  return value as number;
}

function optionalDifficulty(value: unknown): AiDifficulty | undefined {
  if (value === undefined) return undefined;
  if (value !== 'easy' && value !== 'normal' && value !== 'hard') {
    throw new RequestValidationError('scenario.redDifficulty must be easy, normal, or hard');
  }
  return value;
}

function parseOptionalSpawns(value: unknown, name: string): SpawnPosition[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) throw new RequestValidationError(`${name} must be an array`);
  return value.map((item, index) => {
    const spawn = asRecord(item);
    return {
      x: finiteNumber(spawn.x, `${name}[${index}].x`),
      y: finiteNumber(spawn.y, `${name}[${index}].y`),
    };
  });
}

function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new RequestValidationError('request body must be a JSON object');
  }
  return value as Record<string, unknown>;
}
