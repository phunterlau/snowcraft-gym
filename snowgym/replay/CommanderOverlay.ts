import type {
  CommanderPlanTraceEntry,
  CommanderTraceRecording,
} from '../orchestration/trace/CommanderTrace';
import type { TrajectoryDigest } from '../orchestration/trajectory/TrajectoryMonitor';

export interface CommanderOverlayEvent {
  readonly tick: number;
  readonly label: string;
}

export interface CommanderOverlayState {
  readonly plan: CommanderPlanTraceEntry;
  readonly trajectory: TrajectoryDigest | null;
  readonly events: readonly CommanderOverlayEvent[];
}

export function commanderOverlayAtTick(
  trace: CommanderTraceRecording,
  tick: number,
): CommanderOverlayState {
  const boundedTick = Math.min(Math.max(tick, 0), trace.replay.finalTick);
  const plan =
    [...trace.plans].reverse().find((entry) => entry.tick <= boundedTick) ?? trace.plans[0];
  const trajectory =
    [...trace.trajectoryDigests].reverse().find((digest) => digest.endTick <= boundedTick) ?? null;
  const events = [...trace.schedulerEvents, ...trace.lifecycleEvents]
    .filter((event) => event.tick <= boundedTick)
    .sort((left, right) => left.tick - right.tick)
    .slice(-5)
    .map((event) => ({ tick: event.tick, label: eventLabel(event) }));
  return { plan, trajectory, events };
}

function eventLabel(
  event:
    | CommanderTraceRecording['schedulerEvents'][number]
    | CommanderTraceRecording['lifecycleEvents'][number],
): string {
  switch (event.type) {
    case 'trajectory_signal':
      return `${event.trigger}: ${event.roles.join(', ')}`;
    case 'request_started':
      return `request started: ${event.triggers.join(', ')}`;
    case 'response_processed':
      return `response ${event.status} (${event.sourceAgeTicks} ticks old)`;
    case 'request_failed':
      return 'request failed';
    case 'request_timed_out':
      return 'request timed out';
    case 'request_cancelled':
      return 'request cancelled';
    case 'request_limit_reached':
      return `request limit ${event.maximumRequests} reached`;
    case 'trigger_coalesced':
      return `trigger coalesced: ${event.trigger}`;
    case 'response_ignored':
      return `response ignored: ${event.reason}`;
    case 'candidate_activated':
      return `plan v${event.version} activated`;
    case 'candidate_rejected':
      return `candidate rejected; retained v${event.version}`;
    case 'fallback_activated':
      return `fallback v${event.version}: ${event.trigger}`;
  }
}
