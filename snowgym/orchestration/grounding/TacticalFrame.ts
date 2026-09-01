import type { Observation, UnitObservation } from '../../observations/Observation';

const EPSILON = 1e-9;

export interface Point {
  readonly x: number;
  readonly y: number;
}

export interface RelativePoint {
  readonly forward: number;
  readonly lateral: number;
}

export interface TacticalFrame {
  /** Stable arena-center origin used for persistent region anchors. */
  readonly origin: Point;
  readonly ownCentroid: Point;
  readonly enemyCentroid: Point;
  readonly forwardAxis: Point;
  /** Positive lateral is the team's left when facing the enemy force. */
  readonly leftAxis: Point;
  readonly forwardExtent: number;
  readonly lateralExtent: number;
}

export function createTacticalFrame(observation: Observation): TacticalFrame {
  const own = centroid(observation.allies.filter((unit) => unit.alive));
  const enemy = centroid(observation.enemies.filter((unit) => unit.alive));
  const dx = enemy.x - own.x;
  const dy = enemy.y - own.y;
  const distance = Math.hypot(dx, dy);
  const forwardAxis = distance <= EPSILON ? { x: 1, y: 0 } : { x: dx / distance, y: dy / distance };
  const leftAxis = { x: -forwardAxis.y, y: forwardAxis.x };
  return {
    origin: { x: 0, y: 0 },
    ownCentroid: own,
    enemyCentroid: enemy,
    forwardAxis,
    leftAxis,
    forwardExtent: axisExtent(observation.arena.width, observation.arena.height, forwardAxis),
    lateralExtent: axisExtent(observation.arena.width, observation.arena.height, leftAxis),
  };
}

export function toRelative(frame: TacticalFrame, point: Point): RelativePoint {
  const dx = point.x - frame.origin.x;
  const dy = point.y - frame.origin.y;
  return {
    forward: dx * frame.forwardAxis.x + dy * frame.forwardAxis.y,
    lateral: dx * frame.leftAxis.x + dy * frame.leftAxis.y,
  };
}

export function fromRelative(frame: TacticalFrame, point: RelativePoint): Point {
  return {
    x: frame.origin.x + point.forward * frame.forwardAxis.x + point.lateral * frame.leftAxis.x,
    y: frame.origin.y + point.forward * frame.forwardAxis.y + point.lateral * frame.leftAxis.y,
  };
}

export function centroid(units: readonly UnitObservation[]): Point {
  if (units.length === 0) return { x: 0, y: 0 };
  let x = 0;
  let y = 0;
  for (const unit of units) {
    x += unit.x;
    y += unit.y;
  }
  return { x: x / units.length, y: y / units.length };
}

function axisExtent(width: number, height: number, axis: Point): number {
  return (Math.abs(axis.x) * width + Math.abs(axis.y) * height) / 2;
}
