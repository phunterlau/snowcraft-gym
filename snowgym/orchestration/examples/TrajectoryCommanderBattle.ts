import type { CommanderClient } from '../commander/CommanderClient';
import type { CommandPlan } from '../command/CommandPlan';
import type { AiDifficulty } from '../../../src/systems/AISystem';
import type { TeamAction } from '../../actions/UnitAction';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import {
  parseReplayRecording,
  REPLAY_FORMAT,
  type ReplayRecording,
} from '../../replay/ReplayRecording';
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
import {
  buildCommanderTrace,
  type CommanderPlanTraceEntry,
  type CommanderTraceRecording,
} from '../trace/CommanderTrace';

export interface TrajectoryCommanderBattleOptions {
  readonly seed?: number;
  readonly paceMs?: number;
  readonly maxDecisions?: number;
  readonly maximumRequests?: number;
  readonly blueUnits?: number;
  readonly redUnits?: number;
  readonly map?: string;
  readonly redDifficulty?: AiDifficulty;
  readonly pause?: (milliseconds: number) => Promise<void>;
}

export interface TrajectoryCommanderBattleResult {
  readonly seed: number;
  readonly paceMs: number;
  readonly maximumRequests: number;
  readonly replay: ReplayRecording;
  readonly commanderTrace: CommanderTraceRecording;
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
  const blueUnits = positiveInteger(options.blueUnits ?? 10, 'blueUnits');
  const redUnits = positiveInteger(options.redUnits ?? 10, 'redUnits');
  const map = options.map ?? 'arena6.json';
  const redDifficulty = options.redDifficulty ?? 'easy';
  const pause = options.pause ?? sleep;
  const scenario = createMapScenario(map, {
    name: `trajectory-commanded-${blueUnits}v${redUnits}-${map}`,
    seed,
    blueUnits,
    redUnits,
  });
  const environment = new SnowEnvironment({ scenario, decisionHz: 10, redDifficulty });
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
  const frames = [observation];
  const actions: TeamAction[] = [];
  const stateHashes = [status.stateHash];
  const trajectoryDigests: TrajectoryDigest[] = [];
  const planTraces: CommanderPlanTraceEntry[] = [planTrace(store.current())];

  while (!status.terminated && !status.truncated && decisions < maxDecisions) {
    const plan = store.current();
    const before = observation;
    const action = controller.act(before, 1 / environment.decisionHz);
    const result = environment.step(action);
    decisions++;
    actions.push(structuredClone(action));
    frames.push(result.observation);
    rejectedActions += result.info.actionResults.filter(({ accepted }) => !accepted).length;
    observation = result.observation;
    finalTrajectory = monitor.record({
      before,
      after: observation,
      plan,
      actionResults: result.info.actionResults,
    });
    trajectoryDigests.push(finalTrajectory);
    scheduler.tick(observation, finalTrajectory);
    await flushPromises();
    const currentPlan = store.current();
    if (currentPlan.version !== planTraces.at(-1)!.version) {
      planTraces.push(planTrace(currentPlan));
    }
    status = environment.status();
    stateHashes.push(status.stateHash);
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
      decisions,
      terminated: status.terminated,
      truncated: status.truncated,
      winner: status.winner,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      finalTick: status.tick,
    },
  });
  const commanderTrace = buildCommanderTrace(replay, {
    plans: planTraces,
    schedulerEvents: events,
    lifecycleEvents: lifecycle.events(),
    trajectoryDigests,
  });

  return {
    seed,
    paceMs,
    maximumRequests,
    replay,
    commanderTrace,
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

function planTrace(snapshot: ReturnType<PlanStore['current']>): CommanderPlanTraceEntry {
  return {
    tick: snapshot.activatedAtTick,
    version: snapshot.version,
    planId: snapshot.plan.envelope.planId,
    decision: structuredClone(snapshot.plan.envelope.decision),
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
