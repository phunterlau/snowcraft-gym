import type { CommanderClient } from '../commander/CommanderClient';
import type { CommandPlan } from '../command/CommandPlan';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import { createMapScenario } from '../../scenarios/Scenario';
import { PlanAwareTeamController } from '../execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../grounding/PlanGrounder';
import { createFallbackEnvelope } from '../lifecycle/FallbackPlan';
import { PlanLifecycle } from '../lifecycle/PlanLifecycle';
import { PlanStore } from '../runtime/PlanStore';
import { CommanderScheduler, type CommanderSchedulerEvent } from '../scheduler/CommanderScheduler';

export interface SingleRequestBattleOptions {
  readonly seed?: number;
  readonly paceMs?: number;
  readonly maxDecisions?: number;
  readonly pause?: (milliseconds: number) => Promise<void>;
}

export interface SingleRequestBattleResult {
  readonly seed: number;
  readonly paceMs: number;
  readonly schedulerEvents: readonly CommanderSchedulerEvent[];
  readonly activePlan: CommandPlan;
  readonly assignments: Readonly<Record<string, readonly number[]>>;
  readonly rejectedActions: number;
  readonly decisions: number;
  readonly finalTick: number;
  readonly wallTimeMs: number;
  readonly blueAlive: number;
  readonly redAlive: number;
  readonly winner: 'blue' | 'red' | 'draw' | null;
}

/** Executes exactly one commander request while a paced, renderer-free battle continues. */
export async function runSingleRequestCommanderBattle(
  client: CommanderClient,
  options: SingleRequestBattleOptions = {},
): Promise<SingleRequestBattleResult> {
  const seed = safeInteger(options.seed ?? 42, 'seed');
  const paceMs = nonNegativeInteger(options.paceMs ?? 100, 'paceMs');
  const maxDecisions = positiveInteger(options.maxDecisions ?? 10_000, 'maxDecisions');
  const pause = options.pause ?? sleep;
  const scenario = createMapScenario('arena6.json', {
    name: 'single-request-commanded-10v10',
    seed,
    blueUnits: 10,
    redUnits: 10,
  });
  const environment = new SnowEnvironment({ scenario, decisionHz: 10, redDifficulty: 'easy' });
  let observation = environment.reset(seed);
  let status = environment.status();
  const initial = new PlanGrounder().ground(
    createFallbackEnvelope(observation, 'commander_pending', 0),
    observation,
  );
  const store = new PlanStore(initial, observation.tick);
  const lifecycle = new PlanLifecycle(store, undefined, undefined, undefined, {
    maxPlanAgeTicks: 10_000,
  });
  const scheduler = new CommanderScheduler(client, lifecycle, {
    automaticLifecycleMonitoring: false,
    minimumRequestIntervalTicks: 10_000,
    responseTimeoutTicks: 600,
  });
  const controller = new PlanAwareTeamController(store, new ReactiveUnitPolicy());
  scheduler.notify('plan_expired', observation);
  await flushPromises();
  const startedAt = performance.now();
  let decisions = 0;
  let rejectedActions = 0;

  while (!status.terminated && !status.truncated && decisions < maxDecisions) {
    const action = controller.act(observation, 1 / environment.decisionHz);
    const result = environment.step(action);
    decisions++;
    rejectedActions += result.info.actionResults.filter(({ accepted }) => !accepted).length;
    observation = result.observation;
    scheduler.tick(observation);
    status = environment.status();
    if (!status.terminated && !status.truncated && paceMs > 0) await pause(paceMs);
  }
  if (!status.terminated && !status.truncated) {
    throw new RangeError(`episode did not complete within ${maxDecisions} decisions`);
  }
  const events = scheduler.events();
  const requests = events.filter(({ type }) => type === 'request_started');
  if (requests.length !== 1) {
    throw new Error(`single-request battle emitted ${requests.length} commander requests`);
  }

  return {
    seed,
    paceMs,
    schedulerEvents: events,
    activePlan: store.current().plan.envelope.decision,
    assignments: Object.fromEntries(
      store.current().plan.groups.map(({ role, assignment }) => [role, assignment.unitIds]),
    ),
    rejectedActions,
    decisions,
    finalTick: status.tick,
    wallTimeMs: performance.now() - startedAt,
    blueAlive: status.blueAlive,
    redAlive: status.redAlive,
    winner: status.winner,
  };
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function safeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value)) throw new RangeError(`${name} must be a safe integer`);
  return value;
}

function nonNegativeInteger(value: number, name: string): number {
  const result = safeInteger(value, name);
  if (result < 0) throw new RangeError(`${name} must be non-negative`);
  return result;
}

function positiveInteger(value: number, name: string): number {
  const result = safeInteger(value, name);
  if (result <= 0) throw new RangeError(`${name} must be positive`);
  return result;
}
