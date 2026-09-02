import type { TeamAction } from '../../actions/UnitAction';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import type { AiDifficulty } from '../../../src/systems/AISystem';
import {
  parseReplayRecording,
  REPLAY_FORMAT,
  type ReplayRecording,
} from '../../replay/ReplayRecording';
import { createMapScenario } from '../../scenarios/Scenario';
import { hashObservation } from '../../reproducibility/StateHash';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
} from '../command/CommandPlan';
import { MockCommander } from '../commander/MockCommander';
import { commandedTenVsTenPlan } from './CommandedReplayExample';
import { PlanAwareTeamController } from '../execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../grounding/PlanGrounder';
import { PlanLifecycle, type PlanLifecycleEvent } from '../lifecycle/PlanLifecycle';
import { PlanStore } from '../runtime/PlanStore';
import { CommanderScheduler, type CommanderSchedulerEvent } from '../scheduler/CommanderScheduler';
import {
  buildCommanderTrace,
  type CommanderPlanTraceEntry,
  type CommanderTraceRecording,
} from '../trace/CommanderTrace';
import { TrajectoryMonitor, type TrajectoryDigest } from '../trajectory/TrajectoryMonitor';
import { TrajectorySignalDetector } from '../trajectory/TrajectorySignals';

export interface TrajectoryMockCommanderOptions {
  readonly seed?: number;
  readonly latencyTicks?: number;
  readonly maxDecisions?: number;
  readonly blueUnits?: number;
  readonly redUnits?: number;
  readonly map?: string;
  readonly redDifficulty?: AiDifficulty;
}

export interface TrajectoryMockCommanderResult {
  readonly seed: number;
  readonly latencyTicks: number;
  readonly replay: ReplayRecording;
  readonly commanderTrace: CommanderTraceRecording;
  readonly actions: readonly TeamAction[];
  readonly stateHashes: readonly string[];
  readonly trajectoryDigests: readonly TrajectoryDigest[];
  readonly schedulerEvents: readonly CommanderSchedulerEvent[];
  readonly lifecycleEvents: readonly PlanLifecycleEvent[];
  readonly planTraces: readonly CommanderPlanTraceEntry[];
  readonly commanderRequests: number;
  readonly rejectedActions: number;
  readonly decisions: number;
  readonly finalTick: number;
  readonly blueAlive: number;
  readonly redAlive: number;
  readonly winner: 'blue' | 'red' | 'draw' | null;
}

/** Runs a reproducible multi-request commander loop driven by actual trajectory evidence. */
export async function runTrajectoryMockCommanderBattle(
  options: TrajectoryMockCommanderOptions = {},
): Promise<TrajectoryMockCommanderResult> {
  const seed = safeInteger(options.seed ?? 42, 'seed');
  const latencyTicks = nonNegativeInteger(options.latencyTicks ?? 30, 'latencyTicks');
  const maxDecisions = positiveInteger(options.maxDecisions ?? 400, 'maxDecisions');
  const blueUnits = positiveInteger(options.blueUnits ?? 10, 'blueUnits');
  const redUnits = positiveInteger(options.redUnits ?? 10, 'redUnits');
  const map = options.map ?? 'arena6.json';
  const redDifficulty = options.redDifficulty ?? 'easy';
  const scenario = createMapScenario(map, {
    name: `trajectory-mock-commander-${blueUnits}v${redUnits}-${map}`,
    seed,
    blueUnits,
    redUnits,
    maxTicks: maxDecisions * 6,
  });
  const environment = new SnowEnvironment({ scenario, decisionHz: 10, redDifficulty });
  let observation = environment.reset(seed);
  let status = environment.status();
  const initialDecision = directAdvancePlan();
  const initialEnvelope: CommandPlanEnvelope = {
    planId: 'trajectory-initial-direct',
    source: {
      requestId: 'trajectory-initial-request',
      sourceTick: observation.tick,
      sourceStateHash: status.stateHash,
    },
    decision: initialDecision,
  };
  const store = new PlanStore(
    new PlanGrounder().ground(initialEnvelope, observation),
    observation.tick,
  );
  let commanderRequests = 0;
  const client = new MockCommander(
    (_request, callIndex) => {
      commanderRequests++;
      return {
        decision: callIndex === 0 ? directAdvancePlan() : commandedTenVsTenPlan(),
        metadata: { model: 'deterministic-trajectory-mock', latencyMs: 0 },
      };
    },
    { latencyMs: 0, sleep: async () => undefined },
  );
  const lifecycle = new PlanLifecycle(store, undefined, undefined, undefined, {
    maxPlanAgeTicks: 10_000,
  });
  const scheduler = new CommanderScheduler(client, lifecycle, {
    minimumRequestIntervalTicks: 60,
    minimumResponseLatencyTicks: latencyTicks,
    responseTimeoutTicks: Math.max(latencyTicks + 120, 120),
    trajectorySignalDetector: new TrajectorySignalDetector({
      activationGraceTicks: 30,
      recoveryTicks: 30,
      minimumRejectedActions: 3,
      rejectedActionFraction: 0.3,
    }),
    maximumRequests: 3,
  });
  const controller = new PlanAwareTeamController(store, new ReactiveUnitPolicy());
  const monitor = new TrajectoryMonitor({
    windowDecisions: 10,
    minimumProgressDecisions: 5,
    progressEpsilon: 0.2,
    displacementEpsilon: 0.15,
    stalledFraction: 0.4,
  });
  const actions: TeamAction[] = [];
  const frames = [observation];
  const stateHashes = [hashObservation(observation)];
  const trajectoryDigests: TrajectoryDigest[] = [];
  const planTraces: CommanderPlanTraceEntry[] = [planTrace(store.current())];
  let rejectedActions = 0;

  while (!status.terminated && !status.truncated && actions.length < maxDecisions) {
    const plan = store.current();
    const before = observation;
    const action = controller.act(before, 1 / environment.decisionHz);
    const result = environment.step(action);
    const digest = monitor.record({
      before,
      after: result.observation,
      plan,
      actionResults: result.info.actionResults,
    });
    actions.push(structuredClone(action));
    frames.push(result.observation);
    stateHashes.push(hashObservation(result.observation));
    trajectoryDigests.push(digest);
    rejectedActions += result.info.actionResults.filter(({ accepted }) => !accepted).length;
    observation = result.observation;
    scheduler.tick(observation, digest);
    await flushPromises();
    const currentPlan = store.current();
    if (currentPlan.version !== planTraces.at(-1)!.version) {
      planTraces.push(planTrace(currentPlan));
    }
    status = environment.status();
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
      decisions: actions.length,
      terminated: status.terminated,
      truncated: status.truncated,
      winner: status.winner,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      finalTick: status.tick,
    },
  });
  const schedulerEvents = scheduler.events();
  const lifecycleEvents = lifecycle.events();
  const commanderTrace = buildCommanderTrace(replay, {
    plans: planTraces,
    schedulerEvents,
    lifecycleEvents,
    trajectoryDigests,
  });

  return {
    seed,
    latencyTicks,
    replay,
    commanderTrace,
    actions,
    stateHashes,
    trajectoryDigests,
    schedulerEvents,
    lifecycleEvents,
    planTraces,
    commanderRequests,
    rejectedActions,
    decisions: actions.length,
    finalTick: status.tick,
    blueAlive: status.blueAlive,
    redAlive: status.redAlive,
    winner: status.winner,
  };
}

/** Backward-compatible name retained for existing callers; options may now select any map roster. */
export const runTrajectoryMockCommanderTenVsTen = runTrajectoryMockCommanderBattle;

function planTrace(snapshot: ReturnType<PlanStore['current']>): CommanderPlanTraceEntry {
  return {
    tick: snapshot.activatedAtTick,
    version: snapshot.version,
    planId: snapshot.plan.envelope.planId,
    decision: structuredClone(snapshot.plan.envelope.decision),
  };
}

export function directAdvancePlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: 'Advance the entire force directly into the enemy backfield.',
    groups: [
      {
        role: 'main',
        allocationWeight: 1,
        selection: 'balanced',
        order: {
          mission: 'advance',
          objective: { kind: 'region', region: 'enemy_backfield' },
          approach: 'direct',
          engagement: {
            posture: 'balanced',
            fire: 'focus',
            preferredRange: 'medium',
            cohesion: 'tight',
          },
        },
      },
    ],
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
