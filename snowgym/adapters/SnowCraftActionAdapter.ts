import { Team } from '../../src/game/types';
import type { World } from '../../src/game/World';
import type { MovementSystem } from '../../src/systems/MovementSystem';
import type { ThrowSystem } from '../../src/systems/ThrowSystem';
import type { TeamAction, UnitAction } from '../actions/UnitAction';

export interface ActionResult {
  action: UnitAction;
  accepted: boolean;
  reason?: 'duplicate_unit' | 'invalid_value' | 'missing_unit' | 'wrong_team' | 'unavailable';
}

/** Applies semantic SnowGym actions through generic SnowCraft system APIs. */
export class SnowCraftActionAdapter {
  constructor(
    private readonly world: World,
    private readonly movement: MovementSystem,
    private readonly throwing: ThrowSystem,
  ) {}

  apply(team: Team, teamAction: TeamAction): ActionResult[] {
    const seen = new Set<number>();
    return teamAction.actions.map((action) => {
      if (seen.has(action.unitId)) return { action, accepted: false, reason: 'duplicate_unit' };
      seen.add(action.unitId);
      return this.applyOne(team, action);
    });
  }

  private applyOne(team: Team, action: UnitAction): ActionResult {
    const unit = this.world.getPlayer(action.unitId);
    if (!unit) return { action, accepted: false, reason: 'missing_unit' };
    if (unit.team !== team) return { action, accepted: false, reason: 'wrong_team' };

    if (action.type === 'noop') return { action, accepted: true };
    if (action.type === 'hold') {
      const accepted = this.movement.tryHold(unit);
      return { action, accepted, reason: accepted ? undefined : 'unavailable' };
    }
    if (!finiteAction(action)) return { action, accepted: false, reason: 'invalid_value' };

    const accepted =
      action.type === 'move'
        ? this.movement.tryMove(unit, action.x, action.y)
        : this.throwing.tryThrow(unit, action.x, action.y, action.power);
    return { action, accepted, reason: accepted ? undefined : 'unavailable' };
  }
}

function finiteAction(action: Extract<UnitAction, { type: 'move' | 'throw' }>): boolean {
  return (
    Number.isFinite(action.x) &&
    Number.isFinite(action.y) &&
    (action.type !== 'throw' || Number.isFinite(action.power))
  );
}

export const BLUE_TEAM = Team.Player;
export const RED_TEAM = Team.Enemy;
