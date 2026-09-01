import type { GroupRole } from '../command/CommandPlan';
import type { HardLifecycleTrigger } from '../lifecycle/PlanLifecycle';
import type { TrajectoryDigest } from './TrajectoryMonitor';

export const PLAN_OUTCOME_VERSION = 'snowgym.plan-outcome.v0' as const;

export interface PlanOutcomeSummary {
  readonly schemaVersion: typeof PLAN_OUTCOME_VERSION;
  readonly planVersion: number;
  readonly startTick: number;
  readonly endTick: number;
  readonly decisions: number;
  readonly outcome: 'superseded' | 'fallback';
  readonly trigger?: HardLifecycleTrigger;
  readonly ownCasualties: number;
  readonly enemyHealthDelta: number;
  readonly rejectedActions: number;
  readonly stalledRoles: readonly GroupRole[];
}

/** Reduces the final bounded digest for one plan into provider-safe history. */
export function summarizePlanOutcome(
  digest: TrajectoryDigest,
  outcome: PlanOutcomeSummary['outcome'],
  trigger?: HardLifecycleTrigger,
): PlanOutcomeSummary {
  const firstGroup = digest.groups[0];
  return {
    schemaVersion: PLAN_OUTCOME_VERSION,
    planVersion: digest.planVersion,
    startTick: digest.startTick,
    endTick: digest.endTick,
    decisions: digest.decisions,
    outcome,
    ...(trigger ? { trigger } : {}),
    ownCasualties: digest.groups.reduce(
      (sum, group) => sum + Math.max(0, group.livingStart - group.livingEnd),
      0,
    ),
    enemyHealthDelta: firstGroup?.enemyHealthDelta ?? 0,
    rejectedActions: digest.groups.reduce(
      (sum, group) =>
        sum + group.rejectedActions.hold + group.rejectedActions.move + group.rejectedActions.throw,
      0,
    ),
    stalledRoles: digest.groups
      .filter(({ progress }) => progress === 'stalled')
      .map(({ role }) => role),
  };
}
