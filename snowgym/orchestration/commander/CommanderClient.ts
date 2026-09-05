import type { CommandPlan } from '../command/CommandPlan';
import type { LifecycleTrigger } from '../lifecycle/PlanLifecycle';
import type { StrategicSummary } from './StrategicSummary';
import type { TrajectoryDigest } from '../trajectory/TrajectoryMonitor';
import type { PlanOutcomeSummary } from '../trajectory/PlanOutcome';
import type { RecoveryEvidence } from '../recovery/RecoveryEvidence';

export interface CommanderRequest {
  readonly requestId: string;
  readonly triggers: readonly LifecycleTrigger[];
  readonly summary: StrategicSummary;
  readonly currentPlan: CommandPlan;
  /** Optional bounded execution evidence for the plan active at sourceTick. */
  readonly trajectory?: TrajectoryDigest;
  /** Explicit host-owned history; no provider conversation state is required. */
  readonly previousPlanOutcome?: PlanOutcomeSummary;
  /** Opt-in diagnostic input; does not alter the command schema or legacy requests. */
  readonly recoveryEvidence?: RecoveryEvidence;
}

export interface CommanderResponseMetadata {
  readonly requestedModel?: string;
  readonly reasoningEffort?: string;
  readonly model?: string;
  readonly latencyMs?: number;
  readonly tokensIn?: number;
  readonly tokensOut?: number;
  readonly reasoningTokens?: number;
  readonly cachedInputTokens?: number;
  readonly responseId?: string;
  readonly providerRequestId?: string;
}

export interface CommanderResponse {
  /** Untrusted provider output. PlanLifecycle validates it before activation. */
  readonly decision: unknown;
  readonly metadata?: CommanderResponseMetadata;
}

export interface CommanderClient {
  plan(request: CommanderRequest, signal?: AbortSignal): Promise<CommanderResponse>;
}
