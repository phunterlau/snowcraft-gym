import type { UnitAction } from '../actions/UnitAction';
import type {
  Observation,
  ProjectileObservation,
  UnitObservation,
} from '../observations/Observation';
import type { TeamController } from './TeamController';

const ENGAGE_RANGE = 9;
const HOLD_RANGE = 6.5;
const MOVE_STEP = 4;
const DODGE_TRIGGER_RANGE = 3.5;
const DODGE_DISTANCE = 2.4;
const EPSILON = 1e-9;

/**
 * Deliberately small baseline policy: dodge an imminent projectile, otherwise
 * throw at the nearest opponent in range or close the distance.
 */
export class SimpleBlueAgent implements TeamController {
  act(observation: Observation): { actions: UnitAction[] } {
    const actions = observation.allies
      .filter((ally) => ally.alive)
      .map((ally) => this.actionFor(ally, observation));
    return { actions };
  }

  private actionFor(ally: UnitObservation, observation: Observation): UnitAction {
    const incoming = findIncomingProjectile(ally, observation.projectiles);
    if (incoming) {
      return dodgeAction(ally, incoming, observation.arena);
    }

    const target = nearestLivingEnemy(ally, observation.enemies);
    if (!target) return { type: 'noop', unitId: ally.id };

    const distance = Math.hypot(target.x - ally.x, target.y - ally.y);
    if (ally.throwCooldown <= 0 && distance <= ENGAGE_RANGE) {
      const leadTime = 0.18;
      return {
        type: 'throw',
        unitId: ally.id,
        x: target.x + target.vx * leadTime,
        y: target.y + target.vy * leadTime,
        power: clamp(((distance - 1.5) / (ENGAGE_RANGE - 1.5)) * 0.9 + 0.1, 0.18, 1),
      };
    }

    const dx = target.x - ally.x;
    const dy = target.y - ally.y;
    const length = Math.max(distance, EPSILON);
    const travel = Math.min(MOVE_STEP, Math.max(0, distance - HOLD_RANGE));
    if (travel <= EPSILON) return { type: 'noop', unitId: ally.id };

    return {
      type: 'move',
      unitId: ally.id,
      x: ally.x + (dx / length) * travel,
      y: ally.y + (dy / length) * travel,
    };
  }
}

function nearestLivingEnemy(
  ally: UnitObservation,
  enemies: readonly UnitObservation[],
): UnitObservation | null {
  let nearest: UnitObservation | null = null;
  let bestDistanceSq = Number.POSITIVE_INFINITY;
  for (const enemy of enemies) {
    if (!enemy.alive) continue;
    const dx = enemy.x - ally.x;
    const dy = enemy.y - ally.y;
    const distanceSq = dx * dx + dy * dy;
    if (
      distanceSq < bestDistanceSq ||
      (distanceSq === bestDistanceSq && enemy.id < (nearest?.id ?? Infinity))
    ) {
      nearest = enemy;
      bestDistanceSq = distanceSq;
    }
  }
  return nearest;
}

function findIncomingProjectile(
  ally: UnitObservation,
  projectiles: readonly ProjectileObservation[],
): ProjectileObservation | null {
  let nearest: ProjectileObservation | null = null;
  let bestDistanceSq = DODGE_TRIGGER_RANGE * DODGE_TRIGGER_RANGE;
  for (const projectile of projectiles) {
    if (projectile.team === ally.team) continue;
    const dx = ally.x - projectile.x;
    const dy = ally.y - projectile.y;
    const distanceSq = dx * dx + dy * dy;
    const approaching = projectile.vx * dx + projectile.vy * dy > 0;
    if (approaching && distanceSq <= bestDistanceSq) {
      nearest = projectile;
      bestDistanceSq = distanceSq;
    }
  }
  return nearest;
}

function dodgeAction(
  ally: UnitObservation,
  projectile: ProjectileObservation,
  arena: Observation['arena'],
): UnitAction {
  const speed = Math.hypot(projectile.vx, projectile.vy);
  if (speed <= EPSILON) return { type: 'noop', unitId: ally.id };

  const side = ally.id % 2 === 0 ? 1 : -1;
  const x = ally.x + (-projectile.vy / speed) * DODGE_DISTANCE * side;
  const y = ally.y + (projectile.vx / speed) * DODGE_DISTANCE * side;
  const margin = 0.5;
  return {
    type: 'move',
    unitId: ally.id,
    x: clamp(x, -arena.width / 2 + margin, arena.width / 2 - margin),
    y: clamp(y, -arena.height / 2 + margin, arena.height / 2 - margin),
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
