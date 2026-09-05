import { createHash } from 'node:crypto';
import type { TeamAction } from '../../actions/UnitAction';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import type { Observation, UnitObservation } from '../../observations/Observation';
import { createMapScenario, createOpenScenario } from '../../scenarios/Scenario';
import type {
  CommanderClient,
  CommanderRequest,
  CommanderResponseMetadata,
} from '../commander/CommanderClient';
import { summarizeStrategy } from '../commander/StrategicSummary';
import { parseCommandPlan } from '../command/PlanValidator';
import { PlanAwareTeamController } from '../execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../grounding/PlanGrounder';
import { createFallbackEnvelope } from '../lifecycle/FallbackPlan';
import { PlanLifecycle } from '../lifecycle/PlanLifecycle';
import { commanderBackend } from '../providers/CommanderBackend';
import { OpenAICommanderError, openAIRequestBody } from '../providers/OpenAICommanderClient';
import { PlanStore, type PlanSnapshot } from '../runtime/PlanStore';
import { summarizePlanOutcome, type PlanOutcomeSummary } from '../trajectory/PlanOutcome';
import { TrajectoryMonitor } from '../trajectory/TrajectoryMonitor';
import { directAdvancePlan } from '../examples/TrajectoryMockCommanderExample';

export const BACKEND_BENCHMARK_VERSION = 'snowgym.commander-backend-benchmark.v0' as const;
export interface BackendCase {
  readonly id: string;
  readonly seed: number;
  readonly blueUnits: number;
  readonly redUnits: number;
  readonly map: string | null;
  readonly prefixDecisions: number;
  readonly switchAt?: number;
}

/** Small, illustrative cases; these are not fighter training or qualification seeds. */
export const BACKEND_PILOT_CASES: readonly BackendCase[] = [
  {
    id: 'open-5v5-opening',
    seed: 610001,
    blueUnits: 5,
    redUnits: 5,
    map: null,
    prefixDecisions: 0,
  },
  {
    id: 'open-5v5-contact',
    seed: 610002,
    blueUnits: 5,
    redUnits: 5,
    map: null,
    prefixDecisions: 30,
  },
  {
    id: 'terrain-10v10',
    seed: 610003,
    blueUnits: 10,
    redUnits: 10,
    map: 'arena6.json',
    prefixDecisions: 40,
  },
  {
    id: 'terrain-6v10-history',
    seed: 610004,
    blueUnits: 6,
    redUnits: 10,
    map: 'arena6.json',
    prefixDecisions: 60,
    switchAt: 20,
  },
];

export interface BackendFixture {
  readonly format: typeof BACKEND_BENCHMARK_VERSION;
  readonly configuration: BackendCase;
  readonly prefix: readonly TeamAction[];
  readonly prefixStateHashes: readonly string[];
  readonly observation: Observation;
  readonly snapshot: PlanSnapshot;
  readonly request: CommanderRequest;
  readonly fixtureDigest: string;
}

/** All local physical data stays in the artifact. Only fixture.request reaches the provider. */
export function prepareBackendFixture(configuration: BackendCase): BackendFixture {
  const world = startWorld(configuration);
  const prefix: TeamAction[] = [];
  const prefixStateHashes = [world.environment.status().stateHash];
  for (let index = 0; index < configuration.prefixDecisions; index++) {
    if (world.environment.status().terminated || world.environment.status().truncated) {
      throw new Error(`case ${configuration.id} terminates before its scheduled request`);
    }
    transition(world, configuration, index);
    const action = world.controller.act(world.observation, 0.1);
    advance(world, action);
    prefix.push(action);
    prefixStateHashes.push(world.environment.status().stateHash);
  }
  if (!world.observation.match.blueAlive || !world.observation.match.redAlive)
    throw new Error('request must precede elimination');
  const snapshot = world.store.current();
  const request: CommanderRequest = {
    requestId: `backend-pilot-${configuration.id}`,
    triggers: [], // Explicit scheduled opportunity, not a fabricated lifecycle failure.
    summary: summarizeStrategy(world.observation, snapshot),
    currentPlan: snapshot.plan.envelope.decision,
    ...(prefix.length ? { trajectory: world.monitor.digest() } : {}),
    ...(world.previousPlanOutcome ? { previousPlanOutcome: world.previousPlanOutcome } : {}),
  };
  const body = {
    format: BACKEND_BENCHMARK_VERSION,
    configuration,
    prefix,
    prefixStateHashes,
    observation: world.observation,
    snapshot,
    request,
  };
  return { ...body, fixtureDigest: benchmarkDigest(body) };
}

export function restoreBackendFixture(fixture: BackendFixture): ReturnType<typeof startWorld> {
  const { fixtureDigest, ...body } = fixture;
  if (fixture.format !== BACKEND_BENCHMARK_VERSION || benchmarkDigest(body) !== fixtureDigest)
    throw new Error('fixture digest mismatch');
  if (fixture.prefix.length !== fixture.configuration.prefixDecisions)
    throw new Error('prefix length mismatch');
  const world = startWorld(fixture.configuration);
  if (world.environment.status().stateHash !== fixture.prefixStateHashes[0])
    throw new Error('reset hash mismatch');
  fixture.prefix.forEach((action, index) => {
    transition(world, fixture.configuration, index);
    // Check the production executor as well as replaying the archived actions.
    if (benchmarkDigest(world.controller.act(world.observation, 0.1)) !== benchmarkDigest(action))
      throw new Error('prefix policy mismatch');
    advance(world, action);
    if (world.environment.status().stateHash !== fixture.prefixStateHashes[index + 1])
      throw new Error('prefix state mismatch');
  });
  if (
    benchmarkDigest(world.observation) !== benchmarkDigest(fixture.observation) ||
    benchmarkDigest(world.store.current()) !== benchmarkDigest(fixture.snapshot) ||
    benchmarkDigest(world.store.current().plan.envelope.decision) !==
      benchmarkDigest(fixture.request.currentPlan) ||
    benchmarkDigest(summarizeStrategy(world.observation, world.store.current())) !==
      benchmarkDigest(fixture.request.summary) ||
    benchmarkDigest(world.previousPlanOutcome ?? null) !==
      benchmarkDigest(fixture.request.previousPlanOutcome ?? null) ||
    benchmarkDigest(fixture.prefix.length ? world.monitor.digest() : null) !==
      benchmarkDigest(fixture.request.trajectory ?? null)
  ) {
    throw new Error('restored physical, plan, or trajectory state mismatch');
  }
  return world;
}

/** Fixed plan, zero simulated API delay; the production reactive executor remains scripted. */
export function continueBackendPlan(
  fixture: BackendFixture,
  decision: unknown | undefined,
  horizon = 300,
) {
  positiveInteger(horizon, 'continuation horizon');
  const world = restoreBackendFixture(fixture);
  const activation =
    decision === undefined
      ? null
      : new PlanLifecycle(world.store).activateCandidate(
          {
            planId: 'backend-pilot-candidate',
            source: {
              requestId: fixture.request.requestId,
              sourceTick: fixture.observation.tick,
              sourceStateHash: fixture.request.summary.sourceStateHash,
            },
            decision,
          },
          world.observation,
        );
  const fallbackUsed = activation === null || activation.status === 'rejected';
  if (fallbackUsed)
    world.store.activate(
      new PlanGrounder().ground(
        createFallbackEnvelope(world.observation, 'backend_pilot_fallback', 1),
        world.observation,
      ),
      world.observation.tick,
    );
  const activePlan = world.store.current();
  const initialBlueHealth = totalHealth(world.observation.allies);
  const initialRedHealth = totalHealth(world.observation.enemies);
  const actions: TeamAction[] = [];
  const stateHashes = [world.environment.status().stateHash];
  let rejectedActions = 0;
  let issuedActions = 0;
  let canonicalReturn = 0;
  while (
    actions.length < horizon &&
    !world.environment.status().terminated &&
    !world.environment.status().truncated
  ) {
    const action = world.controller.act(world.observation, 0.1);
    const result = advance(world, action);
    actions.push(action);
    stateHashes.push(result.info.stateHash);
    rejectedActions += result.info.actionResults.filter(({ accepted }) => !accepted).length;
    issuedActions += result.info.actionResults.length;
    canonicalReturn += result.reward;
  }
  const status = world.environment.status();
  const damageDealt = initialRedHealth - totalHealth(world.observation.enemies);
  const damageReceived = initialBlueHealth - totalHealth(world.observation.allies);
  const blueCapacity = fixture.observation.allies.reduce((sum, unit) => sum + unit.maxHealth, 0);
  const redCapacity = fixture.observation.enemies.reduce((sum, unit) => sum + unit.maxHealth, 0);
  return {
    activationStatus: activation?.status ?? 'fallback',
    repairs: activation && activation.status !== 'rejected' ? activation.repairs : [],
    activationError: activation?.status === 'rejected' ? activation.error : null,
    fallbackUsed,
    activePlan,
    actions,
    stateHashes,
    metrics: {
      decisions: actions.length,
      finalTick: status.tick,
      winner: status.winner,
      terminated: status.terminated,
      truncated: status.truncated,
      censored: !status.terminated && !status.truncated,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      damageDealt,
      damageReceived,
      normalizedDamageDealt: damageDealt / redCapacity,
      normalizedDamageReceived: damageReceived / blueCapacity,
      netNormalizedDamage: damageDealt / redCapacity - damageReceived / blueCapacity,
      issuedActions,
      rejectedActions,
      rejectedActionRate: rejectedActions / Math.max(issuedActions, 1),
      canonicalReturn,
    },
  };
}

export type BackendArm = ReturnType<typeof commanderBackend>;
export function backendBenchmarkArms(
  backends = ['luna', 'astra'],
  reasoning = ['low', 'medium'],
): BackendArm[] {
  if (!backends.length || !reasoning.length)
    throw new Error('backend and reasoning lists cannot be empty');
  if (reasoning.some((effort) => !['light', 'low', 'medium'].includes(effort)))
    throw new Error('pilot reasoning must be low (light) or medium');
  const arms = backends.flatMap((backend) =>
    reasoning.map((effort) => commanderBackend(backend, effort)),
  );
  if (new Set(arms.map(armId)).size !== arms.length)
    throw new Error('duplicate backend/reasoning arm');
  return arms;
}

export function backendCallSchedule(caseCount: number, arms: readonly BackendArm[]) {
  positiveInteger(caseCount, 'case count');
  if (!arms.length) throw new Error('at least one backend arm is required');
  return Array.from({ length: caseCount }, (_, caseIndex) =>
    arms.map((_, position) => ({
      caseIndex,
      arm: arms[(position + caseIndex) % arms.length],
    })),
  ).flat();
}

export async function benchmarkBackendCall(
  client: CommanderClient,
  arm: BackendArm,
  fixture: BackendFixture,
  options: { horizon?: number; timeoutMs?: number; maxOutputTokens?: number } = {},
) {
  restoreBackendFixture(fixture); // Fail before a billable call on corrupted input.
  const horizon = positiveInteger(options.horizon ?? 300, 'horizon');
  const timeoutMs = positiveInteger(options.timeoutMs ?? 60_000, 'timeoutMs');
  const wireBody = openAIRequestBody(
    fixture.request,
    arm.reasoningEffort,
    options.maxOutputTokens ?? 4096,
    arm.model,
  );
  const abort = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let decision: unknown;
  let metadata: CommanderResponseMetadata | undefined;
  let error: string | null = null;
  let schemaValid = false;
  let modelMatches: boolean | null = null;
  const started = performance.now();
  try {
    const response = await Promise.race([
      client.plan(fixture.request, abort.signal),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          abort.abort();
          reject(new Error(`provider deadline exceeded (${timeoutMs} ms); no retry`));
        }, timeoutMs);
      }),
    ]);
    decision = response.decision;
    metadata = response.metadata;
    parseCommandPlan(decision);
    schemaValid = true;
    // Dated provider snapshots are permitted, but a different model family is not.
    modelMatches =
      metadata?.model === arm.model || (metadata?.model?.startsWith(`${arm.model}-`) ?? false);
    if (!modelMatches)
      throw new Error(`unexpected response model: ${metadata?.model ?? 'missing'}`);
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
    if (caught instanceof OpenAICommanderError) metadata = caught.metadata;
  } finally {
    clearTimeout(timer);
  }
  const latencyMs = performance.now() - started;
  const continuation = continueBackendPlan(fixture, error === null ? decision : undefined, horizon);
  return {
    caseId: fixture.configuration.id,
    arm: armId(arm),
    ...arm,
    fixtureDigest: fixture.fixtureDigest,
    requestDigest: benchmarkDigest(wireBody),
    latencyMs,
    schemaValid,
    modelMatches,
    error,
    timedOut: abort.signal.aborted,
    metadata: metadata ?? null,
    decision: decision ?? null,
    continuation,
  };
}

export type BackendBenchmarkRow = Awaited<ReturnType<typeof benchmarkBackendCall>>;
export function summarizeBackendRows(rows: readonly BackendBenchmarkRow[]) {
  return [...new Set(rows.map((row) => row.arm))].map((arm) => {
    const group = rows.filter((row) => row.arm === arm);
    const accepted = group.filter((row) => !row.continuation.fallbackUsed);
    return {
      arm,
      requests: group.length,
      schemaValid: group.filter((row) => row.schemaValid).length,
      accepted: accepted.length,
      repaired: group.filter((row) => row.continuation.activationStatus === 'repaired').length,
      fallback: group.length - accepted.length,
      failures: group.filter((row) => row.error !== null).length,
      meanLatencyMs: mean(group.map((row) => row.latencyMs)),
      medianLatencyMs: quantile(
        group.map((row) => row.latencyMs),
        0.5,
      ),
      p95LatencyMs: quantile(
        group.map((row) => row.latencyMs),
        0.95,
      ),
      meanInputTokens: knownMean(group.map((row) => row.metadata?.tokensIn)),
      meanOutputTokens: knownMean(group.map((row) => row.metadata?.tokensOut)),
      meanReasoningTokens: knownMean(group.map((row) => row.metadata?.reasoningTokens)),
      meanCachedInputTokens: knownMean(group.map((row) => row.metadata?.cachedInputTokens)),
      usageKnownRequests: group.filter(
        (row) => row.metadata?.tokensIn !== undefined && row.metadata?.tokensOut !== undefined,
      ).length,
      acceptedBlueWins: accepted.filter((row) => row.continuation.metrics.winner === 'blue').length,
      acceptedCensored: accepted.filter((row) => row.continuation.metrics.censored).length,
      meanAcceptedNetDamage: accepted.length
        ? mean(accepted.map((row) => row.continuation.metrics.netNormalizedDamage))
        : null,
      rejectedActions: group.reduce(
        (sum, row) => sum + row.continuation.metrics.rejectedActions,
        0,
      ),
      issuedActions: group.reduce((sum, row) => sum + row.continuation.metrics.issuedActions, 0),
    };
  });
}

export function armId(arm: BackendArm): string {
  return `${arm.backend}-${arm.reasoningEffort}`;
}
export function benchmarkDigest(value: unknown): string {
  return `sha256:${createHash('sha256')
    .update(JSON.stringify(canonical(value)))
    .digest('hex')}`;
}
function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, child]) => [key, canonical(child)]),
    );
  return value;
}
function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0)
    throw new Error(`${name} must be a positive safe integer`);
  return value;
}
function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function knownMean(values: readonly (number | undefined)[]): number | null {
  const known = values.filter((value): value is number => value !== undefined);
  return known.length ? mean(known) : null;
}
function quantile(values: readonly number[], fraction: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (sorted.length - 1) * fraction;
  return sorted[Math.floor(at)] + (at % 1) * (sorted[Math.ceil(at)] - sorted[Math.floor(at)]);
}
function totalHealth(units: readonly UnitObservation[]): number {
  return units.reduce((sum, unit) => sum + (unit.alive ? Math.max(0, unit.health) : 0), 0);
}
function startWorld(configuration: BackendCase) {
  if (!Number.isSafeInteger(configuration.prefixDecisions) || configuration.prefixDecisions < 0)
    throw new Error('invalid prefix length');
  if (
    configuration.switchAt !== undefined &&
    (!Number.isSafeInteger(configuration.switchAt) ||
      configuration.switchAt <= 0 ||
      configuration.switchAt >= configuration.prefixDecisions)
  )
    throw new Error('switch must occur inside the prefix after at least one decision');
  const scenario = configuration.map
    ? createMapScenario(configuration.map, configuration)
    : createOpenScenario(configuration);
  const environment = new SnowEnvironment({ scenario, decisionHz: 10, redDifficulty: 'normal' });
  const observation = environment.reset(configuration.seed);
  const store = new PlanStore(
    new PlanGrounder().ground(
      createFallbackEnvelope(observation, 'backend_pilot_initial', 0),
      observation,
    ),
    observation.tick,
  );
  return {
    environment,
    observation,
    store,
    controller: new PlanAwareTeamController(store, new ReactiveUnitPolicy()),
    monitor: new TrajectoryMonitor({ windowDecisions: 20, minimumProgressDecisions: 5 }),
    previousPlanOutcome: undefined as PlanOutcomeSummary | undefined,
  };
}
function transition(
  world: ReturnType<typeof startWorld>,
  configuration: BackendCase,
  index: number,
): void {
  if (index !== configuration.switchAt) return;
  world.previousPlanOutcome = summarizePlanOutcome(world.monitor.digest(), 'superseded');
  const envelope = createFallbackEnvelope(world.observation, 'backend_pilot_scheduled_switch', 1);
  world.store.activate(
    new PlanGrounder().ground({ ...envelope, decision: directAdvancePlan() }, world.observation),
    world.observation.tick,
  );
  world.monitor.reset();
}
function advance(world: ReturnType<typeof startWorld>, action: TeamAction) {
  const before = world.observation;
  const result = world.environment.step(action);
  world.monitor.record({
    before,
    after: result.observation,
    actionResults: result.info.actionResults,
    plan: world.store.current(),
  });
  world.observation = result.observation;
  return result;
}
