import type { TeamAction } from '../../actions/UnitAction';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import { createMapScenario } from '../../scenarios/Scenario';
import { commandedTenVsTenPlan } from './CommandedReplayExample';
import { MockCommander } from '../commander/MockCommander';
import { PlanAwareTeamController } from '../execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../grounding/PlanGrounder';
import { createFallbackEnvelope } from '../lifecycle/FallbackPlan';
import { PlanLifecycle } from '../lifecycle/PlanLifecycle';
import { PlanStore } from '../runtime/PlanStore';
import { CommanderScheduler, type CommanderSchedulerEvent } from '../scheduler/CommanderScheduler';

export interface DelayedMockCommanderOptions {
  readonly seed?: number;
  readonly latencyTicks?: number;
  readonly maxDecisions?: number;
}

export interface DelayedMockCommanderResult {
  readonly seed: number;
  readonly latencyTicks: number;
  readonly actions: readonly TeamAction[];
  readonly stateHashes: readonly string[];
  readonly schedulerEvents: readonly CommanderSchedulerEvent[];
  readonly rejectedActions: number;
  readonly decisions: number;
  readonly finalTick: number;
  readonly blueAlive: number;
  readonly redAlive: number;
  readonly winner: 'blue' | 'red' | 'draw' | null;
  readonly finalAssignments: Readonly<Record<string, readonly number[]>>;
}

/** Runs a reproducible simulated-latency commander swap without a server or renderer. */
export async function runDelayedMockCommanderTenVsTen(
  options: DelayedMockCommanderOptions = {},
): Promise<DelayedMockCommanderResult> {
  const seed = safeInteger(options.seed ?? 42, 'seed');
  const latencyTicks = nonNegativeInteger(options.latencyTicks ?? 90, 'latencyTicks');
  const maxDecisions = positiveInteger(options.maxDecisions ?? 10_000, 'maxDecisions');
  const scenario = createMapScenario('arena6.json', {
    name: 'delayed-mock-commander-10v10',
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
  const client = new MockCommander(
    () => ({
      decision: commandedTenVsTenPlan(),
      metadata: { model: 'deterministic-mock', latencyMs: 0 },
    }),
    {
      latencyMs: 0,
      sleep: async () => undefined,
    },
  );
  const scheduler = new CommanderScheduler(client, lifecycle, {
    minimumRequestIntervalTicks: 180,
    minimumResponseLatencyTicks: latencyTicks,
    responseTimeoutTicks: Math.max(latencyTicks + 60, 60),
  });
  const controller = new PlanAwareTeamController(store, new ReactiveUnitPolicy());
  scheduler.notify('plan_expired', observation);
  await flushPromises();

  const actions: TeamAction[] = [];
  const stateHashes = [status.stateHash];
  let rejectedActions = 0;
  while (!status.terminated && !status.truncated && actions.length < maxDecisions) {
    const action = controller.act(observation, 1 / environment.decisionHz);
    const result = environment.step(action);
    actions.push(action);
    rejectedActions += result.info.actionResults.filter(({ accepted }) => !accepted).length;
    observation = result.observation;
    scheduler.tick(observation);
    status = environment.status();
    stateHashes.push(status.stateHash);
  }
  if (!status.terminated && !status.truncated) {
    throw new RangeError(`episode did not complete within ${maxDecisions} decisions`);
  }

  return {
    seed,
    latencyTicks,
    actions,
    stateHashes,
    schedulerEvents: scheduler.events(),
    rejectedActions,
    decisions: actions.length,
    finalTick: status.tick,
    blueAlive: status.blueAlive,
    redAlive: status.redAlive,
    winner: status.winner,
    finalAssignments: Object.fromEntries(
      store.current().plan.groups.map(({ role, assignment }) => [role, assignment.unitIds]),
    ),
  };
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
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
