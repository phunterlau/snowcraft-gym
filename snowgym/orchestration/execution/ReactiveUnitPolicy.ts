import type { UnitAction } from '../../actions/UnitAction';
import type {
  Observation,
  ProjectileObservation,
  UnitObservation,
} from '../../observations/Observation';
import type { Approach, Cohesion, PreferredRange } from '../command/CommandPlan';
import { createTacticalFrame, toRelative, type Point } from '../grounding/TacticalFrame';
import type { UnitPolicy, UnitPolicyContext } from './UnitPolicy';

const DODGE_TRIGGER_RANGE = 3.5;
const DODGE_DISTANCE = 2.4;
const ARRIVAL_DISTANCE = 0.65;
const EPSILON = 1e-9;

const RANGE_DISTANCE: Readonly<Record<PreferredRange, number>> = {
  close: 4.5,
  medium: 6.5,
  long: 8.5,
};

const COHESION_SPACING: Readonly<Record<Cohesion, number>> = {
  tight: 0.7,
  normal: 1.2,
  loose: 1.8,
};

/** Fast deterministic executor. Commander intent never overrides immediate survival. */
export class ReactiveUnitPolicy implements UnitPolicy {
  act(context: UnitPolicyContext): UnitAction {
    const { self, observation } = context;
    if (!self.alive || self.state === 'defeated') return { type: 'noop', unitId: self.id };

    const incoming = findIncomingProjectile(self, observation.projectiles);
    if (incoming && canMove(self)) return dodgeAction(self, incoming, observation.arena);

    const target = selectTarget(context);
    if (target && canThrow(self)) {
      const distance = Math.hypot(target.x - self.x, target.y - self.y);
      const maximumRange = throwRange(context.group.command.order.engagement.preferredRange);
      if (distance <= maximumRange) {
        const leadTime = 0.18;
        return {
          type: 'throw',
          unitId: self.id,
          x: target.x + target.vx * leadTime,
          y: target.y + target.vy * leadTime,
          power: clamp(((distance - 1.5) / (maximumRange - 1.5)) * 0.9 + 0.1, 0.18, 1),
        };
      }
    }

    if (!canMove(self)) return { type: 'noop', unitId: self.id };
    const destination = movementDestination(context, target);
    if (distanceSquared(self, destination) <= ARRIVAL_DISTANCE * ARRIVAL_DISTANCE) {
      return { type: 'hold', unitId: self.id };
    }
    return { type: 'move', unitId: self.id, ...clampToArena(destination, observation.arena) };
  }
}

function selectTarget(context: UnitPolicyContext): UnitObservation | null {
  const candidates = context.observation.enemies
    .filter((enemy) => enemy.alive && context.group.candidateEnemyIds.includes(enemy.id))
    .sort((left, right) => left.id - right.id);
  if (candidates.length === 0) return null;
  const fire = context.group.command.order.engagement.fire;
  if (fire === 'focus' && context.group.focusTargetId !== null) {
    return (
      candidates.find(({ id }) => id === context.group.focusTargetId) ??
      nearest(context.self, candidates)
    );
  }
  if (fire === 'distributed') {
    const members = [...context.group.livingMemberIds].sort((left, right) => left - right);
    const index = Math.max(0, members.indexOf(context.self.id));
    return candidates[index % candidates.length];
  }
  return nearest(context.self, candidates);
}

function movementDestination(context: UnitPolicyContext, target: UnitObservation | null): Point {
  const { self, observation, group } = context;
  const order = group.command.order;
  let anchor = group.objective.anchor;

  if (order.mission === 'engage' && target) {
    const postureRangeOffset =
      order.engagement.posture === 'aggressive'
        ? -1
        : order.engagement.posture === 'conservative'
          ? 1.25
          : 0;
    const desiredRange = Math.max(
      2.5,
      RANGE_DISTANCE[order.engagement.preferredRange] + postureRangeOffset,
    );
    const dx = target.x - self.x;
    const dy = target.y - self.y;
    const distance = Math.max(Math.hypot(dx, dy), EPSILON);
    anchor = {
      x: target.x - (dx / distance) * desiredRange,
      y: target.y - (dy / distance) * desiredRange,
    };
  }

  const frame = createTacticalFrame(observation);
  const approachOffset = approachLateralOffset(
    order.approach,
    frame.lateralExtent,
    toRelative(frame, self).lateral,
    self.id,
  );
  const formationOffset = formationLateralOffset(context);
  const cohesionPull =
    order.engagement.cohesion === 'tight'
      ? 0.25
      : order.engagement.cohesion === 'normal'
        ? 0.12
        : 0;
  return {
    x:
      anchor.x +
      frame.leftAxis.x * (approachOffset + formationOffset) +
      (group.centroid.x - self.x) * cohesionPull,
    y:
      anchor.y +
      frame.leftAxis.y * (approachOffset + formationOffset) +
      (group.centroid.y - self.y) * cohesionPull,
  };
}

function approachLateralOffset(
  approach: Approach,
  lateralExtent: number,
  currentLateral: number,
  unitId: number,
): number {
  if (approach === 'left_flank') return Math.min(3.5, lateralExtent * 0.2);
  if (approach === 'right_flank') return -Math.min(3.5, lateralExtent * 0.2);
  if (approach === 'avoid_center') {
    const side =
      Math.abs(currentLateral) > 0.5 ? Math.sign(currentLateral) : unitId % 2 === 0 ? 1 : -1;
    return side * Math.min(3.5, lateralExtent * 0.2);
  }
  return 0;
}

function formationLateralOffset(context: UnitPolicyContext): number {
  const members = [...context.group.livingMemberIds].sort((left, right) => left - right);
  const index = Math.max(0, members.indexOf(context.self.id));
  const centered = index - (members.length - 1) / 2;
  return centered * COHESION_SPACING[context.group.command.order.engagement.cohesion];
}

function nearest(self: UnitObservation, candidates: readonly UnitObservation[]): UnitObservation {
  return [...candidates].sort(
    (left, right) =>
      distanceSquared(self, left) - distanceSquared(self, right) || left.id - right.id,
  )[0];
}

function findIncomingProjectile(
  self: UnitObservation,
  projectiles: readonly ProjectileObservation[],
): ProjectileObservation | null {
  let nearestThreat: ProjectileObservation | null = null;
  let nearestDistanceSquared = DODGE_TRIGGER_RANGE * DODGE_TRIGGER_RANGE;
  for (const projectile of projectiles) {
    if (projectile.team === self.team) continue;
    const dx = self.x - projectile.x;
    const dy = self.y - projectile.y;
    const distance = dx * dx + dy * dy;
    const approaching = projectile.vx * dx + projectile.vy * dy > 0;
    if (!approaching || distance > nearestDistanceSquared) continue;
    nearestThreat = projectile;
    nearestDistanceSquared = distance;
  }
  return nearestThreat;
}

function dodgeAction(
  self: UnitObservation,
  projectile: ProjectileObservation,
  arena: Observation['arena'],
): UnitAction {
  const speed = Math.hypot(projectile.vx, projectile.vy);
  if (speed <= EPSILON) return { type: 'hold', unitId: self.id };
  const side = self.id % 2 === 0 ? 1 : -1;
  return {
    type: 'move',
    unitId: self.id,
    ...clampToArena(
      {
        x: self.x + (-projectile.vy / speed) * DODGE_DISTANCE * side,
        y: self.y + (projectile.vx / speed) * DODGE_DISTANCE * side,
      },
      arena,
    ),
  };
}

function canMove(unit: UnitObservation): boolean {
  return unit.alive && ['idle', 'moving', 'recovering'].includes(unit.state);
}

function canThrow(unit: UnitObservation): boolean {
  return unit.throwCooldown <= 0 && ['idle', 'moving', 'preparingThrow'].includes(unit.state);
}

function throwRange(range: PreferredRange): number {
  if (range === 'close') return 8;
  if (range === 'medium') return 9;
  return 10.5;
}

function distanceSquared(left: Point, right: Point): number {
  return (left.x - right.x) ** 2 + (left.y - right.y) ** 2;
}

function clampToArena(point: Point, arena: Observation['arena']): Point {
  const margin = 0.5;
  return {
    x: clamp(point.x, -arena.width / 2 + margin, arena.width / 2 - margin),
    y: clamp(point.y, -arena.height / 2 + margin, arena.height / 2 - margin),
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
