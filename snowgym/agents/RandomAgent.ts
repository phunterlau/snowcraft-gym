import { PLAYER } from '../../src/game/config';
import type { World } from '../../src/game/World';
import type { TeamAction, UnitAction } from '../actions/UnitAction';
import type { Observation, UnitObservation } from '../observations/Observation';
import type { TeamController } from './TeamController';

const MOVE_RANGE = 4;

/**
 * Seeded random baseline opponent: each decision every unit either holds,
 * wanders to a nearby point, or throws at a random living enemy with a random
 * power. All randomness flows through the world RNG, so episodes stay
 * reproducible from the scenario seed.
 */
export class RandomAgent implements TeamController {
  constructor(private readonly world: World) {}

  act(observation: Observation, _dt: number): TeamAction {
    void _dt; // The random policy re-decides every call.
    const actions: UnitAction[] = observation.allies
      .filter((ally) => ally.alive)
      .map((ally) => this.actionFor(ally, observation));
    return { actions };
  }

  private actionFor(ally: UnitObservation, observation: Observation): UnitAction {
    const roll = this.world.rng.next();
    const enemies = observation.enemies.filter((enemy) => enemy.alive);

    if (roll < 0.25 || enemies.length === 0) {
      return { type: 'noop', unitId: ally.id };
    }

    if (roll < 0.55) {
      return {
        type: 'move',
        unitId: ally.id,
        x: clampAxis(
          ally.x + this.world.rng.range(-MOVE_RANGE, MOVE_RANGE),
          observation.arena.width,
        ),
        y: clampAxis(
          ally.y + this.world.rng.range(-MOVE_RANGE, MOVE_RANGE),
          observation.arena.height,
        ),
      };
    }

    const target = enemies[Math.min(enemies.length - 1, this.world.rng.int(0, enemies.length - 1))];
    return {
      type: 'throw',
      unitId: ally.id,
      x: target.x,
      y: target.y,
      power: this.world.rng.range(0.2, 1),
    };
  }
}

function clampAxis(value: number, size: number): number {
  const limit = Math.max(0, size / 2 - PLAYER.radius);
  return Math.min(limit, Math.max(-limit, value));
}
