import type { Observation } from '../../observations/Observation';
import {
  APPROACHES,
  COHESION_LEVELS,
  FIRE_POLICIES,
  GROUP_ROLES,
  POSTURES,
  PREFERRED_RANGES,
  type GroupRole,
} from '../../orchestration/command/CommandPlan';
import { centroid, createTacticalFrame, toRelative } from '../../orchestration/grounding/TacticalFrame';
import type { PlanSnapshot } from '../../orchestration/runtime/PlanStore';

const MISSIONS = ['engage', 'advance', 'hold', 'withdraw', 'support'] as const;
const OBJECTIVE_KINDS = ['enemy_cluster', 'region', 'current_position', 'ally_group'] as const;

export const PLAN_GROUP_SLOTS = GROUP_ROLES.length;
export const PLAN_FEATURES_PER_GROUP = 38;
export const PLAN_AGE_HORIZON_SECONDS = 30;

export const PLAN_FEATURE_LAYOUT = {
  role: { offset: 0, size: 3 },
  mission: { offset: 3, size: 5 },
  approach: { offset: 8, size: 4 },
  posture: { offset: 12, size: 3 },
  fire: { offset: 15, size: 3 },
  preferredRange: { offset: 18, size: 3 },
  cohesion: { offset: 21, size: 3 },
  objectiveKind: { offset: 24, size: 4 },
  objectiveRelative: { offset: 28, size: 2 },
  groupRelative: { offset: 30, size: 2 },
  allocationFraction: { offset: 32, size: 1 },
  assignedFraction: { offset: 33, size: 1 },
  supportRole: { offset: 34, size: 3 },
  planAge: { offset: 37, size: 1 },
} as const;

export const PLAN_FEATURE_VECTOR_SIZE = PLAN_FEATURES_PER_GROUP;

export interface EncodedPlanTensor {
  /** Row-major [PLAN_GROUP_SLOTS, PLAN_FEATURE_VECTOR_SIZE]. */
  readonly groups: Float32Array;
  readonly groupMask: Uint8Array;
}

/** Encode one grounded plan without exposing unit IDs as learnable features. */
export function encodePlanTensor(
  snapshot: PlanSnapshot,
  observation: Observation,
  currentTick: number,
): EncodedPlanTensor {
  if (!Number.isSafeInteger(currentTick) || currentTick < snapshot.activatedAtTick) {
    throw new RangeError('currentTick must be a safe integer at or after plan activation');
  }
  if (!Number.isFinite(observation.simulationHz) || observation.simulationHz <= 0) {
    throw new RangeError('observation simulationHz must be positive and finite');
  }
  const groups = new Float32Array(PLAN_GROUP_SLOTS * PLAN_FEATURE_VECTOR_SIZE);
  const groupMask = new Uint8Array(PLAN_GROUP_SLOTS);
  const frame = createTacticalFrame(observation);
  const livingAllies = observation.allies.filter(({ alive }) => alive);
  const livingById = new Map(livingAllies.map((unit) => [unit.id, unit]));
  const totalWeight = snapshot.plan.groups.reduce(
    (sum, group) => sum + group.command.allocationWeight,
    0,
  );
  const age = Math.min(
    (currentTick - snapshot.activatedAtTick) /
      (observation.simulationHz * PLAN_AGE_HORIZON_SECONDS),
    1,
  );

  for (const group of snapshot.plan.groups) {
    const slot = GROUP_ROLES.indexOf(group.role);
    if (slot < 0) throw new RangeError(`unknown grounded role ${group.role}`);
    groupMask[slot] = 1;
    const row = groups.subarray(
      slot * PLAN_FEATURE_VECTOR_SIZE,
      (slot + 1) * PLAN_FEATURE_VECTOR_SIZE,
    );
    oneHot(row, PLAN_FEATURE_LAYOUT.role.offset, GROUP_ROLES, group.role);
    oneHot(row, PLAN_FEATURE_LAYOUT.mission.offset, MISSIONS, group.command.order.mission);
    oneHot(row, PLAN_FEATURE_LAYOUT.approach.offset, APPROACHES, group.command.order.approach);
    oneHot(
      row,
      PLAN_FEATURE_LAYOUT.posture.offset,
      POSTURES,
      group.command.order.engagement.posture,
    );
    oneHot(
      row,
      PLAN_FEATURE_LAYOUT.fire.offset,
      FIRE_POLICIES,
      group.command.order.engagement.fire,
    );
    oneHot(
      row,
      PLAN_FEATURE_LAYOUT.preferredRange.offset,
      PREFERRED_RANGES,
      group.command.order.engagement.preferredRange,
    );
    oneHot(
      row,
      PLAN_FEATURE_LAYOUT.cohesion.offset,
      COHESION_LEVELS,
      group.command.order.engagement.cohesion,
    );
    oneHot(
      row,
      PLAN_FEATURE_LAYOUT.objectiveKind.offset,
      OBJECTIVE_KINDS,
      group.objective.kind,
    );
    writeRelative(row, PLAN_FEATURE_LAYOUT.objectiveRelative.offset, group.objective.anchor, frame);

    const assigned = group.assignment.unitIds
      .map((unitId) => livingById.get(unitId))
      .filter((unit) => unit !== undefined);
    writeRelative(
      row,
      PLAN_FEATURE_LAYOUT.groupRelative.offset,
      assigned.length === 0 ? frame.ownCentroid : centroid(assigned),
      frame,
    );
    row[PLAN_FEATURE_LAYOUT.allocationFraction.offset] =
      group.command.allocationWeight / totalWeight;
    row[PLAN_FEATURE_LAYOUT.assignedFraction.offset] =
      assigned.length / Math.max(livingAllies.length, 1);
    if (group.command.order.mission === 'support') {
      oneHot(
        row,
        PLAN_FEATURE_LAYOUT.supportRole.offset,
        GROUP_ROLES,
        group.command.order.objective.role,
      );
    }
    row[PLAN_FEATURE_LAYOUT.planAge.offset] = age;
  }
  return { groups, groupMask };
}

function oneHot<Value extends string>(
  row: Float32Array,
  offset: number,
  values: readonly Value[],
  selected: Value,
): void {
  const index = values.indexOf(selected);
  if (index < 0) throw new RangeError(`cannot encode unknown value ${selected}`);
  row[offset + index] = 1;
}

function writeRelative(
  row: Float32Array,
  offset: number,
  point: { readonly x: number; readonly y: number },
  frame: ReturnType<typeof createTacticalFrame>,
): void {
  const relative = toRelative(frame, point);
  row[offset] = clamp(relative.forward / Math.max(frame.forwardExtent, 1));
  row[offset + 1] = clamp(relative.lateral / Math.max(frame.lateralExtent, 1));
}

function clamp(value: number): number {
  return Math.min(1, Math.max(-1, value));
}

export function planRoleSlot(role: GroupRole): number {
  return GROUP_ROLES.indexOf(role);
}
