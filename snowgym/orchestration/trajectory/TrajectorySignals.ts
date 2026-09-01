import type { GroupRole } from '../command/CommandPlan';
import type { SoftLifecycleTrigger } from '../lifecycle/PlanLifecycle';
import type { PlanSnapshot } from '../runtime/PlanStore';
import type { TrajectoryDigest, TrajectoryGroupDigest } from './TrajectoryMonitor';

export interface TrajectorySignal {
  readonly trigger: SoftLifecycleTrigger;
  readonly tick: number;
  readonly planVersion: number;
  readonly roles: readonly GroupRole[];
  readonly digest: TrajectoryDigest;
}

export interface TrajectorySignalDetectorOptions {
  readonly activationGraceTicks?: number;
  readonly recoveryTicks?: number;
  readonly minimumRejectedActions?: number;
  readonly rejectedActionFraction?: number;
}

interface LatchState {
  latched: boolean;
  recoveredAtTick: number | null;
}

const SIGNAL_ORDER: readonly SoftLifecycleTrigger[] = ['plan_stalled', 'action_rejection_repeated'];

/** Converts bounded trajectory evidence into debounced soft replan signals. */
export class TrajectorySignalDetector {
  private readonly activationGraceTicks: number;
  private readonly recoveryTicks: number;
  private readonly minimumRejectedActions: number;
  private readonly rejectedActionFraction: number;
  private readonly latches = new Map<string, LatchState>();
  private planVersion: number | null = null;

  constructor(options: TrajectorySignalDetectorOptions = {}) {
    this.activationGraceTicks = nonNegativeInteger(
      options.activationGraceTicks ?? 60,
      'activationGraceTicks',
    );
    this.recoveryTicks = nonNegativeInteger(options.recoveryTicks ?? 60, 'recoveryTicks');
    this.minimumRejectedActions = positiveInteger(
      options.minimumRejectedActions ?? 3,
      'minimumRejectedActions',
    );
    this.rejectedActionFraction = fraction(
      options.rejectedActionFraction ?? 0.3,
      'rejectedActionFraction',
    );
  }

  evaluate(digest: TrajectoryDigest, snapshot: PlanSnapshot): readonly TrajectorySignal[] {
    if (digest.planVersion !== snapshot.version) return [];
    if (this.planVersion !== snapshot.version) {
      this.planVersion = snapshot.version;
      this.latches.clear();
    }
    if (digest.endTick - snapshot.activatedAtTick < this.activationGraceTicks) return [];

    const matches = new Map<SoftLifecycleTrigger, GroupRole[]>();
    matches.set(
      'plan_stalled',
      digest.groups.filter(({ progress }) => progress === 'stalled').map(({ role }) => role),
    );
    matches.set(
      'action_rejection_repeated',
      digest.groups.filter((group) => this.rejectionsRepeated(group)).map(({ role }) => role),
    );

    const signals: TrajectorySignal[] = [];
    for (const trigger of SIGNAL_ORDER) {
      const roles = matches.get(trigger) ?? [];
      const activeKeys = new Set(roles.map((role) => latchKey(trigger, role)));
      this.recoverInactive(trigger, activeKeys, digest.endTick);
      const newlyLatched = roles.filter((role) => this.latch(trigger, role));
      if (newlyLatched.length === 0) continue;
      signals.push({
        trigger,
        tick: digest.endTick,
        planVersion: digest.planVersion,
        roles: newlyLatched,
        digest: structuredClone(digest),
      });
    }
    return signals;
  }

  reset(): void {
    this.planVersion = null;
    this.latches.clear();
  }

  private rejectionsRepeated(group: TrajectoryGroupDigest): boolean {
    const issued = group.issuedActions.hold + group.issuedActions.move + group.issuedActions.throw;
    const rejected =
      group.rejectedActions.hold + group.rejectedActions.move + group.rejectedActions.throw;
    return (
      rejected >= this.minimumRejectedActions &&
      rejected / Math.max(issued, 1) >= this.rejectedActionFraction
    );
  }

  private latch(trigger: SoftLifecycleTrigger, role: GroupRole): boolean {
    const key = latchKey(trigger, role);
    const state = this.latches.get(key) ?? { latched: false, recoveredAtTick: null };
    state.recoveredAtTick = null;
    if (state.latched) {
      this.latches.set(key, state);
      return false;
    }
    state.latched = true;
    this.latches.set(key, state);
    return true;
  }

  private recoverInactive(
    trigger: SoftLifecycleTrigger,
    activeKeys: ReadonlySet<string>,
    tick: number,
  ): void {
    for (const [key, state] of this.latches) {
      if (!key.startsWith(`${trigger}:`) || activeKeys.has(key) || !state.latched) continue;
      state.recoveredAtTick ??= tick;
      if (tick - state.recoveredAtTick >= this.recoveryTicks) {
        state.latched = false;
        state.recoveredAtTick = null;
      }
    }
  }
}

function latchKey(trigger: SoftLifecycleTrigger, role: GroupRole): string {
  return `${trigger}:${role}`;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`);
  }
  return value;
}

function fraction(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new RangeError(`${name} must be in (0, 1]`);
  }
  return value;
}
