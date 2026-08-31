import type { Observation, UnitObservation } from '../../observations/Observation';
import {
  GROUP_ROLE_ORDER,
  type CommandPlan,
  type GroupCommand,
  type GroupRole,
} from '../command/CommandPlan';
import { createTacticalFrame, toRelative, type Point } from './TacticalFrame';

export interface GroupAssignment {
  readonly role: GroupRole;
  readonly unitIds: readonly number[];
}

export interface GroupAllocationOptions {
  readonly objectiveAnchors?: Readonly<Partial<Record<GroupRole, Point>>>;
}

export class GroupAllocationError extends Error {}

/** Deterministically allocates every living ally exactly once. */
export class GroupAllocator {
  allocate(
    plan: CommandPlan,
    observation: Observation,
    options: GroupAllocationOptions = {},
  ): readonly GroupAssignment[] {
    const living = observation.allies.filter((unit) => unit.alive);
    if (living.length < plan.groups.length) {
      throw new GroupAllocationError(
        `cannot allocate ${plan.groups.length} groups from ${living.length} living units`,
      );
    }

    const groups = [...plan.groups].sort(
      (left, right) => GROUP_ROLE_ORDER[left.role] - GROUP_ROLE_ORDER[right.role],
    );
    const counts = allocateCounts(groups, living.length);
    const frame = createTacticalFrame(observation);
    const available = new Map(living.map((unit) => [unit.id, unit]));
    const result: GroupAssignment[] = [];

    for (let index = 0; index < groups.length; index++) {
      const group = groups[index];
      const anchor = options.objectiveAnchors?.[group.role];
      if (group.selection === 'nearest_objective' && anchor === undefined) {
        throw new GroupAllocationError(`group ${group.role} requires an objective anchor`);
      }
      const ranked = [...available.values()].sort((left, right) =>
        compareUnits(left, right, group, frame, anchor),
      );
      const selected = ranked.slice(0, counts[index]);
      for (const unit of selected) available.delete(unit.id);
      result.push({
        role: group.role,
        unitIds: selected.map((unit) => unit.id).sort((a, b) => a - b),
      });
    }

    return result;
  }
}

function allocateCounts(groups: readonly GroupCommand[], unitCount: number): number[] {
  const totalWeight = groups.reduce((sum, group) => sum + group.allocationWeight, 0);
  const quotas = groups.map((group) => (unitCount * group.allocationWeight) / totalWeight);
  const counts = quotas.map(Math.floor);
  const remaining = unitCount - counts.reduce((sum, count) => sum + count, 0);
  const remainderOrder = groups
    .map((group, index) => ({ index, remainder: quotas[index] - counts[index], role: group.role }))
    .sort(
      (left, right) =>
        right.remainder - left.remainder ||
        GROUP_ROLE_ORDER[left.role] - GROUP_ROLE_ORDER[right.role],
    );
  for (let index = 0; index < remaining; index++) counts[remainderOrder[index].index]++;

  for (let receiver = 0; receiver < counts.length; receiver++) {
    if (counts[receiver] > 0) continue;
    const donor = counts
      .map((count, index) => ({ count, index, surplus: count - quotas[index] }))
      .filter(({ count }) => count > 1)
      .sort(
        (left, right) =>
          right.count - left.count || right.surplus - left.surplus || left.index - right.index,
      )[0];
    if (!donor) throw new GroupAllocationError('could not give every group at least one unit');
    counts[donor.index]--;
    counts[receiver]++;
  }
  return counts;
}

function compareUnits(
  left: UnitObservation,
  right: UnitObservation,
  group: GroupCommand,
  frame: ReturnType<typeof createTacticalFrame>,
  anchor: Point | undefined,
): number {
  const leftRelative = toRelative(frame, left);
  const rightRelative = toRelative(frame, right);
  let delta = 0;
  switch (group.selection) {
    case 'balanced':
      delta = 0;
      break;
    case 'frontline':
      delta = rightRelative.forward - leftRelative.forward;
      break;
    case 'rearline':
      delta = leftRelative.forward - rightRelative.forward;
      break;
    case 'healthiest':
      delta = healthFraction(right) - healthFraction(left);
      break;
    case 'nearest_objective':
      delta = distanceSquared(left, anchor as Point) - distanceSquared(right, anchor as Point);
      break;
    case 'nearest_left_lane':
      delta = rightRelative.lateral - leftRelative.lateral;
      break;
    case 'nearest_right_lane':
      delta = leftRelative.lateral - rightRelative.lateral;
      break;
  }
  return delta || left.id - right.id;
}

function healthFraction(unit: UnitObservation): number {
  return unit.health / Math.max(unit.maxHealth, 1);
}

function distanceSquared(unit: UnitObservation, point: Point): number {
  return (unit.x - point.x) ** 2 + (unit.y - point.y) ** 2;
}
