import type { Observation, UnitObservation } from '../../observations/Observation';
import { GROUP_ROLES, type PreferredRange } from '../../orchestration/command/CommandPlan';
import { centroid, createTacticalFrame } from '../../orchestration/grounding/TacticalFrame';
import { refreshPlanObjectives } from '../../orchestration/runtime/PlanProjection';
import type { GroundedGroupPlan, PlanSnapshot } from '../../orchestration/runtime/PlanStore';
import { PLAN_GROUP_SLOTS } from './PlanTensorEncoder';

export const PLAN_ROLE_STATE_FEATURES = 20;

export const PLAN_ROLE_STATE_LAYOUT = {
  centroid: { offset: 0, size: 2 },
  velocity: { offset: 2, size: 2 },
  spread: { offset: 4, size: 1 },
  health: { offset: 5, size: 1 },
  livingFraction: { offset: 6, size: 1 },
  readiness: { offset: 7, size: 1 },
  objectiveDisplacement: { offset: 8, size: 2 },
  objectiveDistance: { offset: 10, size: 1 },
  objectiveHealth: { offset: 11, size: 1 },
  rangeError: { offset: 12, size: 1 },
  activationDisplacement: { offset: 13, size: 2 },
  supportedDisplacement: { offset: 15, size: 2 },
  supportedHealth: { offset: 17, size: 1 },
  flankAngle: { offset: 18, size: 1 },
  missionPhase: { offset: 19, size: 1 },
} as const;

export interface EncodedRoleState {
  /** Row-major [PLAN_GROUP_SLOTS, PLAN_ROLE_STATE_FEATURES]. */
  readonly roleState: Float32Array;
  /** Per-role instantaneous mission progress in stable role order. */
  readonly missionProgress: Float32Array;
}

/** Encode physical role summaries without exposing unit IDs as model features. */
export function encodeRoleState(
  snapshot: PlanSnapshot,
  observation: Observation,
): EncodedRoleState {
  const projected = refreshPlanObjectives(snapshot, observation);
  const roleState = new Float32Array(PLAN_GROUP_SLOTS * PLAN_ROLE_STATE_FEATURES);
  const missionProgress = new Float32Array(PLAN_GROUP_SLOTS);
  const diagonal = Math.hypot(observation.arena.width, observation.arena.height);
  const frame = createTacticalFrame(observation);

  for (const group of projected.plan.groups) {
    const slot = GROUP_ROLES.indexOf(group.role);
    if (slot < 0) throw new RangeError(`unknown grounded role ${group.role}`);
    const activationGroup = snapshot.plan.groups.find(({ role }) => role === group.role);
    if (!activationGroup) throw new RangeError(`missing activation group ${group.role}`);
    const row = roleState.subarray(
      slot * PLAN_ROLE_STATE_FEATURES,
      (slot + 1) * PLAN_ROLE_STATE_FEATURES,
    );
    const assigned = assignedUnits(group, observation);
    const living = assigned.filter(({ alive }) => alive);
    const center = living.length > 0 ? centroid(living) : activationGroup.activationAnchor;
    const objective = group.objective.anchor;
    const dx = objective.x - center.x;
    const dy = objective.y - center.y;
    const distance = Math.hypot(dx, dy);
    const objectiveForward = dx * frame.forwardAxis.x + dy * frame.forwardAxis.y;
    const objectiveLateral = dx * frame.leftAxis.x + dy * frame.leftAxis.y;

    row[0] = clampSigned(center.x / Math.max(observation.arena.width / 2, 1));
    row[1] = clampSigned(center.y / Math.max(observation.arena.height / 2, 1));
    row[2] = clampSigned(mean(living, ({ vx }) => vx) / 20);
    row[3] = clampSigned(mean(living, ({ vy }) => vy) / 20);
    row[4] = clampUnit(spread(living, center) / Math.max(diagonal, 1));
    row[5] = healthFraction(assigned);
    row[6] = living.length / Math.max(assigned.length, 1);
    row[7] =
      living.filter((unit) =>
        ['idle', 'moving', 'recovering'].includes(unit.state) && (unit.stunRemaining ?? 0) <= 0,
      ).length / Math.max(living.length, 1);
    row[8] = clampSigned(dx / Math.max(observation.arena.width, 1));
    row[9] = clampSigned(dy / Math.max(observation.arena.height, 1));
    row[10] = clampUnit(distance / Math.max(diagonal, 1));
    row[11] = objectiveHealth(group, observation);
    row[12] = clampSigned(
      (distance - preferredDistance(group.command.order.engagement.preferredRange)) /
        Math.max(diagonal, 1),
    );
    row[13] = clampSigned(
      (center.x - activationGroup.activationAnchor.x) / Math.max(observation.arena.width, 1),
    );
    row[14] = clampSigned(
      (center.y - activationGroup.activationAnchor.y) / Math.max(observation.arena.height, 1),
    );
    const supported = supportedUnits(group, projected.plan.groups, observation);
    if (supported.length > 0) {
      const supportedCenter = centroid(supported.filter(({ alive }) => alive));
      row[15] = clampSigned(
        (supportedCenter.x - center.x) / Math.max(observation.arena.width, 1),
      );
      row[16] = clampSigned(
        (supportedCenter.y - center.y) / Math.max(observation.arena.height, 1),
      );
      row[17] = healthFraction(supported);
    }
    row[18] = clampSigned(Math.atan2(objectiveLateral, objectiveForward) / Math.PI);
    row[19] = missionPhase(group, activationGroup, row, distance, diagonal);
    missionProgress[slot] = row[19];
  }
  return { roleState, missionProgress };
}

function assignedUnits(group: GroundedGroupPlan, observation: Observation): UnitObservation[] {
  const ids = new Set(group.assignment.unitIds);
  return observation.allies.filter(({ id }) => ids.has(id));
}

function supportedUnits(
  group: GroundedGroupPlan,
  groups: readonly GroundedGroupPlan[],
  observation: Observation,
): UnitObservation[] {
  if (group.command.order.mission !== 'support') return [];
  const supportedRole = group.command.order.objective.role;
  const supported = groups.find(({ role }) => role === supportedRole);
  return supported ? assignedUnits(supported, observation) : [];
}

function objectiveHealth(group: GroundedGroupPlan, observation: Observation): number {
  if (group.objective.kind === 'enemy_cluster') {
    const ids = new Set(group.objective.enemyIds);
    return healthFraction(observation.enemies.filter(({ id }) => ids.has(id)));
  }
  if (group.objective.kind === 'ally_group') {
    const ids = new Set(group.objective.unitIds);
    return healthFraction(observation.allies.filter(({ id }) => ids.has(id)));
  }
  return 0;
}

function missionPhase(
  group: GroundedGroupPlan,
  activation: GroundedGroupPlan,
  row: Float32Array,
  distance: number,
  diagonal: number,
): number {
  const mission = group.command.order.mission;
  if (mission === 'engage') return clampUnit(1 - row[11]);
  if (mission === 'hold') {
    const drift = Math.hypot(row[13] * diagonal, row[14] * diagonal);
    return clampUnit(1 - drift / Math.max(diagonal * 0.08, 1e-6));
  }
  if (mission === 'support') {
    const supportDistance = Math.hypot(row[15], row[16]) * diagonal;
    return clampUnit(1 - Math.abs(supportDistance - diagonal * 0.13) / (diagonal * 0.05));
  }
  const initial = Math.hypot(
    activation.objective.anchor.x - activation.activationAnchor.x,
    activation.objective.anchor.y - activation.activationAnchor.y,
  );
  return clampUnit(1 - distance / Math.max(initial, diagonal * 0.01));
}

function healthFraction(units: readonly UnitObservation[]): number {
  const maximum = units.reduce((sum, unit) => sum + Math.max(unit.maxHealth, 1), 0);
  return maximum === 0
    ? 0
    : clampUnit(units.reduce((sum, unit) => sum + Math.max(unit.health, 0), 0) / maximum);
}

function spread(units: readonly UnitObservation[], center: { x: number; y: number }): number {
  return mean(units, (unit) => Math.hypot(unit.x - center.x, unit.y - center.y));
}

function mean(units: readonly UnitObservation[], value: (unit: UnitObservation) => number): number {
  return units.length === 0 ? 0 : units.reduce((sum, unit) => sum + value(unit), 0) / units.length;
}

function preferredDistance(range: PreferredRange): number {
  return range === 'close' ? 4.5 : range === 'medium' ? 6.5 : 8.5;
}

function clampUnit(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function clampSigned(value: number): number {
  return Math.min(1, Math.max(-1, value));
}
