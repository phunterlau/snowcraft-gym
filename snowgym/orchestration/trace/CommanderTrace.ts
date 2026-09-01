import { hashObservation } from '../../reproducibility/StateHash';
import { REPLAY_FORMAT, type ReplayRecording } from '../../replay/ReplayRecording';
import type { CommandPlan } from '../command/CommandPlan';
import { parseCommandPlan } from '../command/PlanValidator';
import type { PlanLifecycleEvent } from '../lifecycle/PlanLifecycle';
import type { CommanderSchedulerEvent } from '../scheduler/CommanderScheduler';
import { TRAJECTORY_DIGEST_VERSION, type TrajectoryDigest } from '../trajectory/TrajectoryMonitor';

export const COMMANDER_TRACE_FORMAT = 'snowgym.commander-trace.v0' as const;

export interface CommanderTraceReplayBinding {
  readonly format: typeof REPLAY_FORMAT;
  readonly scenario: string;
  readonly seed: number;
  readonly finalTick: number;
  readonly finalStateHash: string;
}

export interface CommanderPlanTraceEntry {
  readonly tick: number;
  readonly version: number;
  readonly planId: string;
  readonly decision: CommandPlan;
}

/** ID-free orchestration evidence stored separately from the visual replay. */
export interface CommanderTraceRecording {
  readonly format: typeof COMMANDER_TRACE_FORMAT;
  readonly replay: CommanderTraceReplayBinding;
  readonly plans: readonly CommanderPlanTraceEntry[];
  readonly schedulerEvents: readonly CommanderSchedulerEvent[];
  readonly lifecycleEvents: readonly PlanLifecycleEvent[];
  readonly trajectoryDigests: readonly TrajectoryDigest[];
}

export interface CommanderTraceInput {
  readonly plans: readonly CommanderPlanTraceEntry[];
  readonly schedulerEvents: readonly CommanderSchedulerEvent[];
  readonly lifecycleEvents: readonly PlanLifecycleEvent[];
  readonly trajectoryDigests: readonly TrajectoryDigest[];
}

export function buildCommanderTrace(
  replay: ReplayRecording,
  input: CommanderTraceInput,
): CommanderTraceRecording {
  const finalStateHash = replay.stateHashes?.at(-1) ?? hashObservation(replay.frames.at(-1)!);
  return parseCommanderTrace(
    {
      format: COMMANDER_TRACE_FORMAT,
      replay: {
        format: replay.format,
        scenario: replay.scenario,
        seed: replay.seed,
        finalTick: replay.outcome.finalTick,
        finalStateHash,
      },
      plans: input.plans,
      schedulerEvents: input.schedulerEvents,
      lifecycleEvents: input.lifecycleEvents,
      trajectoryDigests: input.trajectoryDigests,
    },
    replay,
  );
}

/** Validates untrusted sidecar JSON and optionally binds it to a loaded replay. */
export function parseCommanderTrace(
  value: unknown,
  replay?: ReplayRecording,
): CommanderTraceRecording {
  assertNoEntityIds(value, '$');
  const trace = record(value, 'trace');
  if (trace.format !== COMMANDER_TRACE_FORMAT) {
    throw new CommanderTraceFormatError(`expected format ${COMMANDER_TRACE_FORMAT}`);
  }
  const binding = record(trace.replay, 'replay');
  if (binding.format !== REPLAY_FORMAT) {
    throw new CommanderTraceFormatError(`expected replay format ${REPLAY_FORMAT}`);
  }
  nonEmptyString(binding.scenario, 'replay.scenario');
  integer(binding.seed, 'replay.seed');
  const finalTick = nonNegativeInteger(binding.finalTick, 'replay.finalTick');
  stateHash(binding.finalStateHash, 'replay.finalStateHash');

  const plans = array(trace.plans, 'plans');
  let previousVersion = 0;
  let previousPlanTick = -1;
  for (const [index, value] of plans.entries()) {
    const plan = record(value, `plans[${index}]`);
    const tick = boundedTick(plan.tick, `plans[${index}].tick`, finalTick);
    const version = positiveInteger(plan.version, `plans[${index}].version`);
    if (tick < previousPlanTick) {
      throw new CommanderTraceFormatError('plan ticks must not decrease');
    }
    if (version <= previousVersion) {
      throw new CommanderTraceFormatError('plan versions must increase');
    }
    previousPlanTick = tick;
    previousVersion = version;
    nonEmptyString(plan.planId, `plans[${index}].planId`);
    parseCommandPlan(plan.decision);
  }
  if (plans.length === 0) throw new CommanderTraceFormatError('plans must not be empty');

  validateTimedRecords(trace.schedulerEvents, 'schedulerEvents', finalTick, schedulerEventTypes);
  validateTimedRecords(trace.lifecycleEvents, 'lifecycleEvents', finalTick, lifecycleEventTypes);

  const digests = array(trace.trajectoryDigests, 'trajectoryDigests');
  let previousDigestEnd = -1;
  for (const [index, value] of digests.entries()) {
    const digest = record(value, `trajectoryDigests[${index}]`);
    if (digest.schemaVersion !== TRAJECTORY_DIGEST_VERSION) {
      throw new CommanderTraceFormatError(
        `trajectoryDigests[${index}] must use ${TRAJECTORY_DIGEST_VERSION}`,
      );
    }
    positiveInteger(digest.planVersion, `trajectoryDigests[${index}].planVersion`);
    const startTick = boundedTick(
      digest.startTick,
      `trajectoryDigests[${index}].startTick`,
      finalTick,
    );
    const endTick = boundedTick(digest.endTick, `trajectoryDigests[${index}].endTick`, finalTick);
    if (endTick <= startTick) {
      throw new CommanderTraceFormatError(`trajectoryDigests[${index}] has invalid tick range`);
    }
    if (endTick < previousDigestEnd) {
      throw new CommanderTraceFormatError('trajectory digest end ticks must not decrease');
    }
    previousDigestEnd = endTick;
    positiveInteger(digest.decisions, `trajectoryDigests[${index}].decisions`);
    const groups = array(digest.groups, `trajectoryDigests[${index}].groups`);
    if (groups.length === 0 || groups.length > 3) {
      throw new CommanderTraceFormatError(
        `trajectoryDigests[${index}].groups must contain one to three entries`,
      );
    }
  }

  if (replay) validateReplayBinding(binding, replay);
  return structuredClone(value) as CommanderTraceRecording;
}

export class CommanderTraceFormatError extends Error {}

const schedulerEventTypes = new Set([
  'request_cancelled',
  'request_limit_reached',
  'trajectory_signal',
  'request_started',
  'trigger_coalesced',
  'request_timed_out',
  'request_failed',
  'response_ignored',
  'response_processed',
]);

const lifecycleEventTypes = new Set([
  'candidate_activated',
  'candidate_rejected',
  'fallback_activated',
]);

function validateReplayBinding(binding: Record<string, unknown>, replay: ReplayRecording): void {
  const expectedHash = replay.stateHashes?.at(-1) ?? hashObservation(replay.frames.at(-1)!);
  if (
    binding.format !== replay.format ||
    binding.scenario !== replay.scenario ||
    binding.seed !== replay.seed ||
    binding.finalTick !== replay.outcome.finalTick ||
    binding.finalStateHash !== expectedHash
  ) {
    throw new CommanderTraceFormatError('commander trace does not match the loaded replay');
  }
}

function validateTimedRecords(
  value: unknown,
  name: string,
  finalTick: number,
  allowedTypes: ReadonlySet<string>,
): void {
  const values = array(value, name);
  let previousTick = -1;
  for (const [index, value] of values.entries()) {
    const event = record(value, `${name}[${index}]`);
    if (typeof event.type !== 'string' || !allowedTypes.has(event.type)) {
      throw new CommanderTraceFormatError(`${name}[${index}].type is invalid`);
    }
    const tick = boundedTick(event.tick, `${name}[${index}].tick`, finalTick);
    if (tick < previousTick) {
      throw new CommanderTraceFormatError(`${name} ticks must not decrease`);
    }
    previousTick = tick;
  }
}

function assertNoEntityIds(value: unknown, path: string): void {
  if (Array.isArray(value)) {
    value.forEach((child, index) => assertNoEntityIds(child, `${path}[${index}]`));
    return;
  }
  if (typeof value !== 'object' || value === null) return;
  for (const [key, child] of Object.entries(value)) {
    if (key === 'unitId' || key === 'unitIds' || key === 'enemyId') {
      throw new CommanderTraceFormatError(`${path}.${key} is forbidden in commander traces`);
    }
    assertNoEntityIds(child, `${path}.${key}`);
  }
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new CommanderTraceFormatError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new CommanderTraceFormatError(`${name} must be an array`);
  return value;
}

function integer(value: unknown, name: string): number {
  if (!Number.isSafeInteger(value)) {
    throw new CommanderTraceFormatError(`${name} must be a safe integer`);
  }
  return value as number;
}

function nonNegativeInteger(value: unknown, name: string): number {
  const result = integer(value, name);
  if (result < 0) throw new CommanderTraceFormatError(`${name} must be non-negative`);
  return result;
}

function positiveInteger(value: unknown, name: string): number {
  const result = integer(value, name);
  if (result <= 0) throw new CommanderTraceFormatError(`${name} must be positive`);
  return result;
}

function boundedTick(value: unknown, name: string, finalTick: number): number {
  const result = nonNegativeInteger(value, name);
  if (result > finalTick) throw new CommanderTraceFormatError(`${name} exceeds replay final tick`);
  return result;
}

function nonEmptyString(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new CommanderTraceFormatError(`${name} must be a non-empty string`);
  }
  return value;
}

function stateHash(value: unknown, name: string): string {
  const result = nonEmptyString(value, name);
  if (!/^fnv1a64:[0-9a-f]{16}$/.test(result)) {
    throw new CommanderTraceFormatError(`${name} is invalid`);
  }
  return result;
}
