import type { Observation, UnitObservation } from '../../observations/Observation';
import type { EnemyClusterSelector, GroupCommand, GroupRole, Region } from '../command/CommandPlan';
import type { GroupAssignment } from './GroupAllocator';
import {
  centroid,
  createTacticalFrame,
  fromRelative,
  toRelative,
  type Point,
  type TacticalFrame,
} from './TacticalFrame';

export type ResolvedObjective =
  | {
      readonly kind: 'enemy_cluster';
      readonly selector: EnemyClusterSelector;
      readonly anchor: Point;
      readonly enemyIds: readonly number[];
    }
  | { readonly kind: 'region'; readonly region: Region; readonly anchor: Point }
  | { readonly kind: 'current_position'; readonly anchor: Point }
  | {
      readonly kind: 'ally_group';
      readonly role: GroupRole;
      readonly anchor: Point;
      readonly unitIds: readonly number[];
    };

export class TargetResolutionError extends Error {}

interface EnemyCluster {
  readonly units: readonly UnitObservation[];
  readonly anchor: Point;
  readonly healthFraction: number;
}

export class TargetResolver {
  constructor(private readonly clusterRadius = 6) {
    if (!Number.isFinite(clusterRadius) || clusterRadius <= 0) {
      throw new RangeError('clusterRadius must be positive and finite');
    }
  }

  resolve(
    group: GroupCommand,
    observation: Observation,
    assignments: readonly GroupAssignment[] = [],
  ): ResolvedObjective {
    const objective = group.order.objective;
    if (objective.kind === 'enemy_cluster') {
      return this.resolveEnemyCluster(objective.select, observation);
    }
    if (objective.kind === 'region') {
      return {
        kind: 'region',
        region: objective.region,
        anchor: resolveRegion(objective.region, observation),
      };
    }
    if (objective.kind === 'current_position') {
      const assignment = findAssignment(group.role, assignments);
      return {
        kind: 'current_position',
        anchor: assignmentCentroid(assignment, observation),
      };
    }

    const assignment = findAssignment(objective.role, assignments);
    return {
      kind: 'ally_group',
      role: objective.role,
      anchor: assignmentCentroid(assignment, observation),
      unitIds: [...assignment.unitIds],
    };
  }

  private resolveEnemyCluster(
    selector: EnemyClusterSelector,
    observation: Observation,
  ): ResolvedObjective {
    const clusters = clusterEnemies(
      observation.enemies.filter((unit) => unit.alive),
      this.clusterRadius,
    );
    if (clusters.length === 0) throw new TargetResolutionError('no living enemy cluster exists');
    const frame = createTacticalFrame(observation);
    clusters.sort((left, right) => compareClusters(left, right, selector, frame));
    const selected = clusters[0];
    return {
      kind: 'enemy_cluster',
      selector,
      anchor: selected.anchor,
      enemyIds: selected.units.map((unit) => unit.id),
    };
  }
}

function clusterEnemies(units: readonly UnitObservation[], radius: number): EnemyCluster[] {
  const ordered = [...units].sort((left, right) => left.id - right.id);
  const remaining = new Map(ordered.map((unit) => [unit.id, unit]));
  const clusters: EnemyCluster[] = [];
  const radiusSquared = radius * radius;

  while (remaining.size > 0) {
    const first = remaining.values().next().value as UnitObservation;
    remaining.delete(first.id);
    const members = [first];
    for (let cursor = 0; cursor < members.length; cursor++) {
      const source = members[cursor];
      for (const candidate of [...remaining.values()]) {
        if (distanceSquared(source, candidate) > radiusSquared) continue;
        remaining.delete(candidate.id);
        members.push(candidate);
      }
    }
    members.sort((left, right) => left.id - right.id);
    clusters.push({
      units: members,
      anchor: centroid(members),
      healthFraction:
        members.reduce((sum, unit) => sum + unit.health / Math.max(unit.maxHealth, 1), 0) /
        members.length,
    });
  }
  return clusters;
}

function compareClusters(
  left: EnemyCluster,
  right: EnemyCluster,
  selector: EnemyClusterSelector,
  frame: TacticalFrame,
): number {
  let delta = 0;
  switch (selector) {
    case 'nearest':
      delta =
        distanceSquared(left.anchor, frame.origin) - distanceSquared(right.anchor, frame.origin);
      break;
    case 'largest':
      delta = right.units.length - left.units.length;
      break;
    case 'weakest':
      delta = left.healthFraction - right.healthFraction;
      break;
    case 'leftmost':
      delta = toRelative(frame, right.anchor).lateral - toRelative(frame, left.anchor).lateral;
      break;
    case 'rightmost':
      delta = toRelative(frame, left.anchor).lateral - toRelative(frame, right.anchor).lateral;
      break;
  }
  return delta || left.units[0].id - right.units[0].id;
}

function resolveRegion(region: Region, observation: Observation): Point {
  const frame = createTacticalFrame(observation);
  const relative =
    region === 'left_lane'
      ? { forward: frame.forwardExtent * 0.15, lateral: frame.lateralExtent * 0.55 }
      : region === 'right_lane'
        ? { forward: frame.forwardExtent * 0.15, lateral: frame.lateralExtent * -0.55 }
        : region === 'center_lane'
          ? { forward: frame.forwardExtent * 0.15, lateral: 0 }
          : region === 'own_backfield'
            ? { forward: frame.forwardExtent * -0.45, lateral: 0 }
            : { forward: frame.forwardExtent * 0.75, lateral: 0 };
  return clampToObservationArena(fromRelative(frame, relative), observation);
}

function findAssignment(role: GroupRole, assignments: readonly GroupAssignment[]): GroupAssignment {
  const assignment = assignments.find((candidate) => candidate.role === role);
  if (!assignment) throw new TargetResolutionError(`no assignment exists for group ${role}`);
  return assignment;
}

function assignmentCentroid(assignment: GroupAssignment, observation: Observation): Point {
  const ids = new Set(assignment.unitIds);
  const units = observation.allies.filter((unit) => unit.alive && ids.has(unit.id));
  if (units.length === 0) {
    throw new TargetResolutionError(`group ${assignment.role} has no living assigned units`);
  }
  return centroid(units);
}

function clampToObservationArena(point: Point, observation: Observation): Point {
  const margin = 0.5;
  return {
    x: clamp(point.x, -observation.arena.width / 2 + margin, observation.arena.width / 2 - margin),
    y: clamp(
      point.y,
      -observation.arena.height / 2 + margin,
      observation.arena.height / 2 - margin,
    ),
  };
}

function distanceSquared(left: Point, right: Point): number {
  return (left.x - right.x) ** 2 + (left.y - right.y) ** 2;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
