import type { CommanderClient } from '../commander/CommanderClient';
import type { CommandPlan } from '../command/CommandPlan';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import { createMapScenario } from '../../scenarios/Scenario';
import { PlanAwareTeamController } from '../execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../grounding/PlanGrounder';
import { PlanLifecycle } from '../lifecycle/PlanLifecycle';
import { PlanStore } from '../runtime/PlanStore';
import { CommanderScheduler, type CommanderSchedulerEvent } from '../scheduler/CommanderScheduler';
import { TrajectoryMonitor, type TrajectoryDigest } from '../trajectory/TrajectoryMonitor';
import { TrajectorySignalDetector } from '../trajectory/TrajectorySignals';
import { directAdvancePlan } from './TrajectoryMockCommanderExample';

export interface TrajectoryCommanderBattleOptions {
  readonly seed?: number;
  readonly paceMs?: number;
  readonly maxDecisions?: number;
  readonly maximumRequests?: number;
  readonly pause?: (milliseconds: number) => Promise<void>;
}

export interface TrajectoryCommanderBattleResult {
  readonly seed: number;
  readonly paceMs: number;
  readonly maximumRequests: number;
  readonly schedulerEvents: readonly CommanderSchedulerEvent[];
  readonly finalTrajectory: TrajectoryDigest;
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

/** Runs a paced, renderer-free battle with capped trajectory-driven commander calls. */
export async function runTrajectoryCommanderBattle(
  client: CommanderClient,
  options: TrajectoryCommanderBattleOptions = {},
): Promise<TrajectoryCommanderBattleResult> {
  const seed = safeInteger(options.seed ?? 42, 'seed');
  const paceMs = nonNegativeInteger(options.paceMs ?? 100, 'paceMs');
  const maxDecisions = positiveInteger(options.maxDecisions ?? 10_000, 'maxDecisions');
  const maximumRequests = positiveInteger(options.maximumRequests ?? 3, 'maximumRequests');
  const pause = options.pause ?? sleep;
  const scenario = createMapScenario('arena6.json', {
    name: 'trajectory-commanded-10v10',
    seed,
    blueUnits: 10,
    redUnits: 10,
  });
  const environment = new SnowEnvironment({ scenario, decisionHz: 10, redDifficulty: 'easy' });
  let observation = environment.reset(seed);
  let status = environment.status();
  const initialPlan = directAdvancePlan();
  const grounded = new PlanGrounder().ground(
    {
      planId: 'trajectory-live-initial',
      source: {
        requestId: 'trajectory-live-initial-request',
        sourceTick: observation.tick,
        sourceStateHash: status.stateHash,
      },
      decision: initialPlan,
    },
    observation,
  );
  const store = new PlanStore(grounded, observation.tick);
  const lifecycle = new PlanLifecycle(store, undefined, undefined, undefined, {
    maxPlanAgeTicks: 720,
  });
  const scheduler = new CommanderScheduler(client, lifecycle, {
    minimumRequestIntervalTicks: 240,
    responseTimeoutTicks: 600,
    maximumRequests,
    trajectorySignalDetector: new TrajectorySignalDetector({
      activationGraceTicks: 60,
      recoveryTicks: 60,
      minimumRejectedActions: 3,
      rejectedActionFraction: 0.3,
    }),
  });
  const controller = new PlanAwareTeamController(store, new ReactiveUnitPolicy());
  const monitor = new TrajectoryMonitor({
    windowDecisions: 20,
    minimumProgressDecisions: 10,
    progressEpsilon: 0.25,
    displacementEpsilon: 0.2,
    stalledFraction: 0.5,
    movementIntentFraction: 0.5,
  });
  const startedAt = performance.now();
  let decisions = 0;
  let rejectedActions = 0;
  let finalTrajectory: TrajectoryDigest | null = null;

  while (!status.terminated && !status.truncated && decisions < maxDecisions) {
    const plan = store.current();
    const before = observation;
    const action = controller.act(before, 1 / environment.decisionHz);
    const result = environment.step(action);
    decisions++;
    rejectedActions += result.info.actionResults.filter(({ accepted }) => !accepted).length;
    observation = result.observation;
    finalTrajectory = monitor.record({
      before,
      after: observation,
      plan,
      actionResults: result.info.actionResults,
    });
    scheduler.tick(observation, finalTrajectory);
    await flushPromises();
    status = environment.status();
    if (!status.terminated && !status.truncated && paceMs > 0) await pause(paceMs);
  }
  if (!status.terminated && !status.truncated) {
    scheduler.close(status.tick);
    throw new RangeError(`episode did not complete within ${maxDecisions} decisions`);
  }
  scheduler.close(status.tick);
  if (!finalTrajectory) throw new Error('trajectory battle completed without a decision');
  const events = scheduler.events();
  const requests = events.filter(({ type }) => type === 'request_started').length;
  if (requests > maximumRequests) {
    throw new Error(`trajectory battle exceeded request limit ${maximumRequests}: ${requests}`);
  }

  return {
    seed,
    paceMs,
    maximumRequests,
    schedulerEvents: events,
    finalTrajectory,
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
