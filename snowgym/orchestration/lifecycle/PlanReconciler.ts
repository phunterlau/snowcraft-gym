import type { Observation } from '../../observations/Observation';
import {
  GROUP_ROLE_ORDER,
  type CommandPlanEnvelope,
  type CommandPlanSource,
  type GroupCommand,
  type GroupRole,
} from '../command/CommandPlan';
import { parseCommandPlan } from '../command/PlanValidator';
import { PlanGrounder } from '../grounding/PlanGrounder';
import type { GroundedPlan } from '../runtime/PlanStore';
import { fallbackOrder } from './FallbackPlan';

export interface CandidatePlanEnvelope {
  readonly planId: string;
  readonly source: CommandPlanSource;
  readonly decision: unknown;
}

export interface PlanRepair {
  readonly path: string;
  readonly reason: 'roster_too_small' | 'missing_support_target' | 'enemy_force_eliminated';
  readonly before: unknown;
  readonly after: unknown;
}

export interface ReconciledPlan {
  readonly plan: GroundedPlan;
  readonly repairs: readonly PlanRepair[];
  readonly sourceAgeTicks: number;
}

/** Validates a candidate, repairs bounded feasibility drift, then grounds against current state. */
export class PlanReconciler {
  constructor(private readonly grounder = new PlanGrounder()) {}

  reconcile(candidate: CandidatePlanEnvelope, observation: Observation): ReconciledPlan {
    validateCandidateMetadata(candidate, observation);
    const parsed = parseCommandPlan(candidate.decision);
    const groups = parsed.groups.map((group) => structuredClone(group));
    const repairs: PlanRepair[] = [];
    const livingAllies = observation.allies.filter(({ alive }) => alive).length;
    if (livingAllies === 0)
      throw new PlanReconciliationError('cannot ground a plan without living allies');

    while (groups.length > livingAllies) {
      const dropIndex = groupToDrop(groups);
      const [dropped] = groups.splice(dropIndex, 1);
      repairs.push({
        path: `$.groups[role=${dropped.role}]`,
        reason: 'roster_too_small',
        before: dropped,
        after: null,
      });
    }

    repairBrokenSupport(groups, observation, repairs);
    repairEnemyObjectives(groups, observation, repairs);
    const decision = parseCommandPlan({ ...parsed, groups });
    const envelope: CommandPlanEnvelope = {
      planId: candidate.planId,
      source: candidate.source,
      decision,
    };
    return {
      plan: this.grounder.ground(envelope, observation),
      repairs,
      sourceAgeTicks: observation.tick - candidate.source.sourceTick,
    };
  }
}

export class PlanReconciliationError extends Error {}

function validateCandidateMetadata(
  candidate: CandidatePlanEnvelope,
  observation: Observation,
): void {
  if (typeof candidate.planId !== 'string' || candidate.planId.trim().length === 0) {
    throw new PlanReconciliationError('planId must not be empty');
  }
  if (
    typeof candidate.source?.requestId !== 'string' ||
    candidate.source.requestId.trim().length === 0
  ) {
    throw new PlanReconciliationError('requestId must not be empty');
  }
  if (!Number.isSafeInteger(candidate.source.sourceTick) || candidate.source.sourceTick < 0) {
    throw new PlanReconciliationError('sourceTick must be a non-negative safe integer');
  }
  if (candidate.source.sourceTick > observation.tick) {
    throw new PlanReconciliationError('sourceTick cannot be newer than current observation');
  }
}

function groupToDrop(groups: readonly GroupCommand[]): number {
  const candidates = groups
    .map((group, index) => ({ group, index }))
    .filter(({ group }) => group.role !== 'main')
    .sort((left, right) => GROUP_ROLE_ORDER[right.group.role] - GROUP_ROLE_ORDER[left.group.role]);
  if (candidates.length === 0) {
    throw new PlanReconciliationError('cannot drop the required main group');
  }
  return candidates[0].index;
}

function repairBrokenSupport(
  groups: GroupCommand[],
  observation: Observation,
  repairs: PlanRepair[],
): void {
  const roles = new Set<GroupRole>(groups.map(({ role }) => role));
  for (let index = 0; index < groups.length; index++) {
    const group = groups[index];
    if (group.order.objective.kind !== 'ally_group' || roles.has(group.order.objective.role))
      continue;
    const replacement: GroupCommand = {
      ...group,
      selection: 'balanced',
      order: fallbackOrder(observation),
    };
    repairs.push({
      path: `$.groups[role=${group.role}]`,
      reason: 'missing_support_target',
      before: group,
      after: replacement,
    });
    groups[index] = replacement;
  }
}

function repairEnemyObjectives(
  groups: GroupCommand[],
  observation: Observation,
  repairs: PlanRepair[],
): void {
  if (observation.enemies.some(({ alive }) => alive)) return;
  for (let index = 0; index < groups.length; index++) {
    const group = groups[index];
    if (group.order.objective.kind !== 'enemy_cluster') continue;
    const replacement: GroupCommand = {
      ...group,
      selection: 'balanced',
      order: fallbackOrder(observation),
    };
    repairs.push({
      path: `$.groups[role=${group.role}]`,
      reason: 'enemy_force_eliminated',
      before: group,
      after: replacement,
    });
    groups[index] = replacement;
  }
}
