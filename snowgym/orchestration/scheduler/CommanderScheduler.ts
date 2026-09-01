import type { Observation } from '../../observations/Observation';
import type {
  CommanderClient,
  CommanderRequest,
  CommanderResponse,
  CommanderResponseMetadata,
} from '../commander/CommanderClient';
import { summarizeStrategy } from '../commander/StrategicSummary';
import {
  PlanLifecycle,
  type LifecycleTrigger,
  type PlanActivationOutcome,
} from '../lifecycle/PlanLifecycle';
import type { CandidatePlanEnvelope } from '../lifecycle/PlanReconciler';

export interface CommanderSchedulerOptions {
  readonly minimumRequestIntervalTicks?: number;
  readonly responseTimeoutTicks?: number;
  /** Reproducible simulated inference latency; zero uses provider completion immediately. */
  readonly minimumResponseLatencyTicks?: number;
  /** Disable only for bounded one-request demos whose external-call count must be fixed. */
  readonly automaticLifecycleMonitoring?: boolean;
}

export type CommanderSchedulerEvent =
  | {
      readonly type: 'request_started';
      readonly tick: number;
      readonly requestId: string;
      readonly triggers: readonly LifecycleTrigger[];
      readonly eligibleAtTick: number;
      readonly deadlineTick: number;
    }
  | {
      readonly type: 'trigger_coalesced';
      readonly tick: number;
      readonly trigger: LifecycleTrigger;
      readonly requestId: string | null;
    }
  | {
      readonly type: 'request_timed_out';
      readonly tick: number;
      readonly requestId: string;
    }
  | {
      readonly type: 'request_failed';
      readonly tick: number;
      readonly requestId: string;
      readonly error: string;
    }
  | {
      readonly type: 'response_ignored';
      readonly tick: number;
      readonly requestId: string;
      readonly reason: 'timed_out' | 'superseded';
    }
  | {
      readonly type: 'response_processed';
      readonly tick: number;
      readonly requestId: string;
      readonly status: PlanActivationOutcome['status'];
      readonly sourceAgeTicks: number;
      readonly metadata?: CommanderResponseMetadata;
    };

interface InFlightRequest {
  readonly sequence: number;
  readonly request: CommanderRequest;
  readonly sourceTick: number;
  readonly sourceStateHash: string;
  readonly eligibleAtTick: number;
  readonly deadlineTick: number;
  readonly abortController: AbortController;
}

type Completion =
  | { readonly sequence: number; readonly response: CommanderResponse }
  | { readonly sequence: number; readonly error: unknown };

/** Starts commander work asynchronously while the synchronous team controller keeps running. */
export class CommanderScheduler {
  private readonly minimumRequestIntervalTicks: number;
  private readonly responseTimeoutTicks: number;
  private readonly minimumResponseLatencyTicks: number;
  private readonly automaticLifecycleMonitoring: boolean;
  private readonly pendingTriggers = new Set<LifecycleTrigger>();
  private readonly completions: Completion[] = [];
  private readonly trace: CommanderSchedulerEvent[] = [];
  private inFlight: InFlightRequest | null = null;
  private lastRequestTick: number | null = null;
  private lastTimedOutSequence = 0;
  private sequence = 0;

  constructor(
    private readonly client: CommanderClient,
    private readonly lifecycle: PlanLifecycle,
    options: CommanderSchedulerOptions = {},
  ) {
    this.minimumRequestIntervalTicks = nonNegativeInteger(
      options.minimumRequestIntervalTicks ?? 180,
      'minimumRequestIntervalTicks',
    );
    this.responseTimeoutTicks = positiveInteger(
      options.responseTimeoutTicks ?? 180,
      'responseTimeoutTicks',
    );
    this.minimumResponseLatencyTicks = nonNegativeInteger(
      options.minimumResponseLatencyTicks ?? 0,
      'minimumResponseLatencyTicks',
    );
    this.automaticLifecycleMonitoring = options.automaticLifecycleMonitoring ?? true;
    if (this.minimumResponseLatencyTicks >= this.responseTimeoutTicks) {
      throw new RangeError('minimumResponseLatencyTicks must be less than responseTimeoutTicks');
    }
  }

  /** Poll once per team decision. This method never awaits provider work. */
  tick(observation: Observation): void {
    this.processTimeout(observation);
    this.processCompletions(observation);
    if (this.automaticLifecycleMonitoring) this.monitorLifecycle(observation);
    this.startPendingIfEligible(observation);
  }

  /** Adds an external lifecycle signal without blocking or duplicating in-flight work. */
  notify(trigger: LifecycleTrigger, observation: Observation): void {
    if (this.inFlight || !this.requestIntervalElapsed(observation.tick)) {
      const wasPending = this.pendingTriggers.has(trigger);
      this.pendingTriggers.add(trigger);
      if (!wasPending) {
        this.trace.push({
          type: 'trigger_coalesced',
          tick: observation.tick,
          trigger,
          requestId: this.inFlight?.request.requestId ?? null,
        });
      }
      return;
    }
    this.startRequest([trigger], observation);
  }

  status(): {
    readonly inFlightRequestId: string | null;
    readonly pendingTriggers: readonly LifecycleTrigger[];
  } {
    return {
      inFlightRequestId: this.inFlight?.request.requestId ?? null,
      pendingTriggers: orderedTriggers(this.pendingTriggers),
    };
  }

  events(): readonly CommanderSchedulerEvent[] {
    return structuredClone(this.trace);
  }

  private monitorLifecycle(observation: Observation): void {
    const triggers = this.lifecycle.evaluate(observation);
    if (triggers.length === 0) return;
    this.lifecycle.maintain(observation);
    for (const trigger of triggers) this.notify(trigger, observation);
  }

  private startPendingIfEligible(observation: Observation): void {
    if (
      this.inFlight ||
      this.pendingTriggers.size === 0 ||
      !this.requestIntervalElapsed(observation.tick)
    ) {
      return;
    }
    const triggers = orderedTriggers(this.pendingTriggers);
    this.pendingTriggers.clear();
    this.startRequest(triggers, observation);
  }

  private startRequest(triggers: readonly LifecycleTrigger[], observation: Observation): void {
    const sequence = ++this.sequence;
    const requestId = `commander-request-${sequence}`;
    const snapshot = this.lifecycle.current();
    const summary = summarizeStrategy(observation, snapshot);
    const request: CommanderRequest = {
      requestId,
      triggers: [...triggers],
      summary,
      currentPlan: snapshot.plan.envelope.decision,
    };
    const abortController = new AbortController();
    const inFlight: InFlightRequest = {
      sequence,
      request,
      sourceTick: summary.sourceTick,
      sourceStateHash: summary.sourceStateHash,
      eligibleAtTick: observation.tick + this.minimumResponseLatencyTicks,
      deadlineTick: observation.tick + this.responseTimeoutTicks,
      abortController,
    };
    this.inFlight = inFlight;
    this.lastRequestTick = observation.tick;
    this.trace.push({
      type: 'request_started',
      tick: observation.tick,
      requestId,
      triggers: [...triggers],
      eligibleAtTick: inFlight.eligibleAtTick,
      deadlineTick: inFlight.deadlineTick,
    });
    void this.client.plan(request, abortController.signal).then(
      (response) => this.completions.push({ sequence, response }),
      (error) => this.completions.push({ sequence, error }),
    );
  }

  private processTimeout(observation: Observation): void {
    if (!this.inFlight || observation.tick < this.inFlight.deadlineTick) return;
    const expired = this.inFlight;
    this.inFlight = null;
    this.lastTimedOutSequence = expired.sequence;
    expired.abortController.abort();
    this.trace.push({
      type: 'request_timed_out',
      tick: observation.tick,
      requestId: expired.request.requestId,
    });
  }

  private processCompletions(observation: Observation): void {
    for (let index = 0; index < this.completions.length; ) {
      const completion = this.completions[index];
      if (this.inFlight?.sequence === completion.sequence) {
        if (observation.tick < this.inFlight.eligibleAtTick) {
          index++;
          continue;
        }
        const request = this.inFlight;
        this.inFlight = null;
        this.completions.splice(index, 1);
        if ('error' in completion) {
          this.trace.push({
            type: 'request_failed',
            tick: observation.tick,
            requestId: request.request.requestId,
            error: errorMessage(completion.error),
          });
          continue;
        }
        this.activateResponse(request, completion.response, observation);
        continue;
      }

      this.completions.splice(index, 1);
      this.trace.push({
        type: 'response_ignored',
        tick: observation.tick,
        requestId: `commander-request-${completion.sequence}`,
        reason: completion.sequence <= this.lastTimedOutSequence ? 'timed_out' : 'superseded',
      });
    }
  }

  private activateResponse(
    request: InFlightRequest,
    response: CommanderResponse,
    observation: Observation,
  ): void {
    const candidate: CandidatePlanEnvelope = {
      planId: `commander-plan-${request.sequence}`,
      source: {
        requestId: request.request.requestId,
        sourceTick: request.sourceTick,
        sourceStateHash: request.sourceStateHash,
      },
      decision: response.decision,
    };
    const outcome = this.lifecycle.activateCandidate(candidate, observation);
    this.trace.push({
      type: 'response_processed',
      tick: observation.tick,
      requestId: request.request.requestId,
      status: outcome.status,
      sourceAgeTicks:
        outcome.status === 'rejected'
          ? observation.tick - request.sourceTick
          : outcome.sourceAgeTicks,
      metadata: response.metadata,
    });
  }

  private requestIntervalElapsed(tick: number): boolean {
    return (
      this.lastRequestTick === null ||
      tick - this.lastRequestTick >= this.minimumRequestIntervalTicks
    );
  }
}

const TRIGGER_ORDER: readonly LifecycleTrigger[] = [
  'plan_expired',
  'own_force_loss_major',
  'group_eliminated',
  'objective_completed',
];

function orderedTriggers(triggers: ReadonlySet<LifecycleTrigger>): LifecycleTrigger[] {
  return TRIGGER_ORDER.filter((trigger) => triggers.has(trigger));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
  return value;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
}
