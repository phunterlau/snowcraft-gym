import type { CommandPlan } from '../command/CommandPlan';
import type { LifecycleTrigger } from '../lifecycle/PlanLifecycle';
import type { StrategicSummary } from './StrategicSummary';
import type { TrajectoryDigest } from '../trajectory/TrajectoryMonitor';

export interface CommanderRequest {
  readonly requestId: string;
  readonly triggers: readonly LifecycleTrigger[];
  readonly summary: StrategicSummary;
  readonly currentPlan: CommandPlan;
  /** Optional bounded execution evidence for the plan active at sourceTick. */
  readonly trajectory?: TrajectoryDigest;
}

export interface CommanderResponseMetadata {
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
