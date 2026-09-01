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
import { snowGymCapabilities } from '../protocol/Capabilities';

export interface ServiceResponse {
  status: number;
  body: unknown;
}

/** Transport-independent request handler used by the HTTP server and tests. */
export class SnowGymService {
  private readonly bluePolicy = new SimpleBlueAgent();
  private readonly mutationCache = new Map<string, CachedMutation>();
  private environment: SnowEnvironment;

  constructor(environment = new SnowEnvironment()) {
    this.environment = environment;
  }

  handle(method: string, path: string, body: unknown = undefined): ServiceResponse {
    try {
      if (method === 'GET' && path === '/health') {
        return { status: 200, body: { ok: true } };
      }
      if (method === 'GET' && path === '/capabilities') {
        return { status: 200, body: snowGymCapabilities() };
      }
      if (method === 'GET' && path === '/status') {
        return { status: 200, body: this.snapshot() };
      }
      if (method === 'POST' && path === '/reset') {
        const request = parseReset(body);
        return this.mutate('reset', request, () => this.reset(request));
      }
      if (method === 'POST' && path === '/step') {
        const request = parseStep(body);
        return this.mutate('step', request, () => this.step(request.action));
      }
      if (method === 'POST' && path === '/step-joint') {
        const request = parseJointStep(body);
        return this.mutate('step-joint', request, () =>
          this.stepJoint(request.actions.blue, request.actions.red),
        );
      }
      if (method === 'POST' && path === '/step-scripted') {
        const request = parseGuardedRequest(body, []);
        return this.mutate('step-scripted', request, () => this.step(this.defaultBlueAction()));
      }
      if (method === 'POST' && path === '/autoplay') {
        const request = parseAutoplay(body);
        return this.mutate('autoplay', request, () => this.autoplay(request.maxDecisions));
      }
      return { status: 404, body: { error: 'not_found' } };
    } catch (error) {
      if (error instanceof EpisodeCompleteError) {
        return { status: 409, body: { error: 'episode_complete', message: error.message } };
      }
      if (error instanceof StateConflictError) {
        return {
          status: 409,
          body: { error: error.code, message: error.message, ...error.detail },
        };
      }
      if (error instanceof RequestValidationError || error instanceof RangeError) {
        return { status: 400, body: { error: 'invalid_request', message: error.message } };
      }
      throw error;
    }
  }

  private snapshot(): object {
    const blue = this.environment.observe(Team.Player);
    const red = this.environment.observe(Team.Enemy);
    return {
      status: this.environment.status(),
      observation: blue,
      observations: { blue, red },
    };
  }

  private defaultBlueAction(): TeamAction {
    return this.bluePolicy.act(this.environment.observe(Team.Player));
  }

  private step(action: TeamAction): object {
    const result = this.environment.step(action);
    return { ...result, info: { ...result.info, action } };
  }

  private stepJoint(blue: TeamAction, red: TeamAction): object {
    const result = this.environment.stepJoint(blue, red);
    return { ...result, info: { ...result.info, actions: { blue, red } } };
  }

  private reset(request: ResetRequest): object {
    if (request.map !== undefined) {
      this.environment = new SnowEnvironment({
        scenario: createMapScenario(request.map, {
          seed: request.seed,
          maxTicks: request.maxTicks,
          blueUnits: request.blueUnits,
          redUnits: request.redUnits,
        }),
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
    this.mutationCache.clear();
    return this.snapshot();
  }

  private mutate(
    operation: string,
    request: GuardedRequest,
    mutation: () => object,
  ): ServiceResponse {
    const fingerprint = JSON.stringify({ operation, ...request, idempotencyKey: undefined });
    if (request.idempotencyKey) {
      const cached = this.mutationCache.get(request.idempotencyKey);
      if (cached) {
        if (cached.fingerprint !== fingerprint) {
          throw new StateConflictError(
            'idempotency_conflict',
            `idempotencyKey ${request.idempotencyKey} was already used for a different mutation`,
          );
        }
        return cached.response;
      }
    }

    const actualStateHash = this.environment.status().stateHash;
    if (request.expectedStateHash && request.expectedStateHash !== actualStateHash) {
      throw new StateConflictError(
        'stale_state',
        'expectedStateHash does not match current state',
        {
          expectedStateHash: request.expectedStateHash,
          actualStateHash,
        },
      );
    }

    const response = { status: 200, body: mutation() };
    if (request.idempotencyKey)
      this.rememberMutation(request.idempotencyKey, fingerprint, response);
    return response;
  }

  private rememberMutation(key: string, fingerprint: string, response: ServiceResponse): void {
    this.mutationCache.set(key, { fingerprint, response });
    if (this.mutationCache.size > 256) {
      const oldest = this.mutationCache.keys().next().value as string | undefined;
      if (oldest) this.mutationCache.delete(oldest);
    }
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

class StateConflictError extends Error {
  constructor(
    readonly code: 'stale_state' | 'idempotency_conflict',
    message: string,
    readonly detail: Record<string, unknown> = {},
  ) {
    super(message);
  }
}

interface CachedMutation {
  fingerprint: string;
  response: ServiceResponse;
}

interface GuardedRequest {
  expectedStateHash?: string;
  idempotencyKey?: string;
}

interface StepRequest extends GuardedRequest {
  action: TeamAction;
}

interface JointStepRequest extends GuardedRequest {
  actions: { blue: TeamAction; red: TeamAction };
}

interface AutoplayRequest extends GuardedRequest {
  maxDecisions: number;
}

interface ResetRequest extends GuardedRequest {
  seed?: number;
  scenario?: OpenScenarioOptions;
  map?: string;
  maxTicks?: number;
  blueUnits?: number;
  redUnits?: number;
  decisionHz?: number;
  redDifficulty?: AiDifficulty;
  redController?: RedControllerType;
}

function parseReset(body: unknown): ResetRequest {
  if (body === undefined || body === null) return {};
  const record = asRecord(body);
  assertAllowedKeys(record, ['seed', 'scenario', 'expectedStateHash', 'idempotencyKey'], 'request');
  const guards = parseGuards(record);
  const seed = optionalSafeInteger(record.seed, 'seed');
  if (record.scenario === undefined) return { ...guards, seed };

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
    const conflicting = ['arenaWidth', 'arenaHeight', 'blueSpawns', 'redSpawns'].filter(
      (key) => scenario[key] !== undefined,
    );
    if (conflicting.length > 0) {
      throw new RequestValidationError(
        `scenario.map fixes terrain and native spawn positions; remove: ${conflicting.sort().join(', ')}`,
      );
    }
    return {
      ...guards,
      seed,
      map: scenario.map,
      maxTicks: optionalSafeInteger(scenario.maxTicks, 'scenario.maxTicks'),
      blueUnits: optionalSafeInteger(scenario.blueUnits, 'scenario.blueUnits'),
      redUnits: optionalSafeInteger(scenario.redUnits, 'scenario.redUnits'),
      decisionHz: optionalSafeInteger(scenario.decisionHz, 'scenario.decisionHz'),
      redDifficulty,
      redController,
    };
  }

  return {
    ...guards,
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

function parseStep(body: unknown): StepRequest {
  const record = asRecord(body);
  assertAllowedKeys(record, ['action', 'expectedStateHash', 'idempotencyKey'], 'request');
  if (record.action === undefined) {
    throw new RequestValidationError(
      'action is required; use /step-scripted for the built-in policy',
    );
  }
  return { ...parseGuards(record), action: parseTeamAction(record.action, 'action') };
}

function parseJointStep(body: unknown): JointStepRequest {
  const record = asRecord(body);
  assertAllowedKeys(record, ['actions', 'expectedStateHash', 'idempotencyKey'], 'request');
  const actions = asRecord(record.actions);
  assertAllowedKeys(actions, ['blue', 'red'], 'actions');
  if (actions.blue === undefined || actions.red === undefined) {
    throw new RequestValidationError('actions.blue and actions.red are required');
  }
  return {
    ...parseGuards(record),
    actions: {
      blue: parseTeamAction(actions.blue, 'actions.blue'),
      red: parseTeamAction(actions.red, 'actions.red'),
    },
  };
}

function parseTeamAction(value: unknown, name: string): TeamAction {
  const actionRecord = asRecord(value);
  assertAllowedKeys(actionRecord, ['actions'], name);
  if (!Array.isArray(actionRecord.actions)) {
    throw new RequestValidationError(`${name}.actions must be an array`);
  }
  return { actions: actionRecord.actions.map(parseUnitAction) };
}

function parseUnitAction(value: unknown): UnitAction {
  const action = asRecord(value);
  if (!Number.isSafeInteger(action.unitId)) {
    throw new RequestValidationError('unitId must be a safe integer');
  }
  const unitId = action.unitId as number;
  if (action.type === 'noop' || action.type === 'hold') {
    assertAllowedKeys(action, ['type', 'unitId'], 'unit action');
    return { type: action.type, unitId };
  }
  if (action.type !== 'move' && action.type !== 'throw') {
    throw new RequestValidationError('action type must be noop, hold, move, or throw');
  }
  assertAllowedKeys(
    action,
    action.type === 'move' ? ['type', 'unitId', 'x', 'y'] : ['type', 'unitId', 'x', 'y', 'power'],
    'unit action',
  );
  const x = finiteNumber(action.x, 'x');
  const y = finiteNumber(action.y, 'y');
  if (action.type === 'move') return { type: 'move', unitId, x, y };
  return { type: 'throw', unitId, x, y, power: finiteNumber(action.power, 'power') };
}

function parseAutoplay(body: unknown): AutoplayRequest {
  const record = body === undefined || body === null ? {} : asRecord(body);
  assertAllowedKeys(record, ['maxDecisions', 'expectedStateHash', 'idempotencyKey'], 'request');
  const maxDecisions = record.maxDecisions ?? 10_000;
  if (!Number.isSafeInteger(maxDecisions) || (maxDecisions as number) <= 0) {
    throw new RequestValidationError('maxDecisions must be a positive safe integer');
  }
  return { ...parseGuards(record), maxDecisions: Math.min(maxDecisions as number, 10_000) };
}

function parseGuardedRequest(body: unknown, extraAllowed: string[]): GuardedRequest {
  const record = body === undefined || body === null ? {} : asRecord(body);
  assertAllowedKeys(record, [...extraAllowed, 'expectedStateHash', 'idempotencyKey'], 'request');
  return parseGuards(record);
}

function parseGuards(record: Record<string, unknown>): GuardedRequest {
  let expectedStateHash: string | undefined;
  if (record.expectedStateHash !== undefined) {
    if (
      typeof record.expectedStateHash !== 'string' ||
      !/^fnv1a64:[0-9a-f]{16}$/.test(record.expectedStateHash)
    ) {
      throw new RequestValidationError(
        'expectedStateHash must use the fnv1a64: plus 16 lowercase hex format',
      );
    }
    expectedStateHash = record.expectedStateHash;
  }
  let idempotencyKey: string | undefined;
  if (record.idempotencyKey !== undefined) {
    if (
      typeof record.idempotencyKey !== 'string' ||
      !/^[A-Za-z0-9._:-]{1,128}$/.test(record.idempotencyKey)
    ) {
      throw new RequestValidationError(
        'idempotencyKey must be 1-128 letters, digits, dots, underscores, colons, or hyphens',
      );
    }
    idempotencyKey = record.idempotencyKey;
  }
  return { expectedStateHash, idempotencyKey };
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
    assertAllowedKeys(spawn, ['x', 'y'], `${name}[${index}]`);
    return {
      x: finiteNumber(spawn.x, `${name}[${index}].x`),
      y: finiteNumber(spawn.y, `${name}[${index}].y`),
    };
  });
}

function assertAllowedKeys(
  record: Record<string, unknown>,
  allowedKeys: readonly string[],
  name: string,
): void {
  const allowed = new Set(allowedKeys);
  const unknown = Object.keys(record).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw new RequestValidationError(`unknown ${name} fields: ${unknown.sort().join(', ')}`);
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new RequestValidationError('request body must be a JSON object');
  }
  return value as Record<string, unknown>;
}
