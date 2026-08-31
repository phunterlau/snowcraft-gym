import { SIM } from '../../src/game/config';
import { type Team } from '../../src/game/types';
import type { World } from '../../src/game/World';
import { AISystem, type AiDifficulty } from '../../src/systems/AISystem';
import type { ThrowSystem } from '../../src/systems/ThrowSystem';
import type { TeamAction, UnitAction } from '../actions/UnitAction';
import type { Observation } from '../observations/Observation';
import type { TeamController } from './TeamController';

/** Slice size for fixed-timestep advancement (matches the engine tick). */
const SIM_DT_SLICE = SIM.dt;

/**
 * Re-runs the scripted squad AI at decision cadence and reports the orders it
 * issued as semantic actions. Between decisions the underlying AI keeps its
 * per-tick reactive behavior (dodges, aim error, retargeting), so ordering the
 * world through this controller is behaviorally identical to registering the
 * AISystem directly.
 */
export class ScriptedAiAgent implements TeamController {
  private readonly ai: AISystem;

  constructor(
    private readonly world: World,
    throwSystem: ThrowSystem,
    private readonly team: Team,
    private readonly opponent: Team,
    difficulty: AiDifficulty,
  ) {
    this.ai = new AISystem(world, null, throwSystem, difficulty, {
      controlled: team,
      target: opponent,
    });
  }

  act(_observation: Observation, dt: number): TeamAction {
    void _observation; // The scripted AI perceives the world directly.
    const moveTargets = new Map(
      this.world.players.map((unit) => [unit.id, unit.moveTarget] as const),
    );

    // Fixed-timestep advance: re-running in exact dt slices preserves the
    // engine's floating-point timer cadence, keeping this bridge behaviorally
    // identical to registering the AISystem directly.
    let remaining = dt;
    while (remaining > 1e-12) {
      const slice = Math.min(SIM_DT_SLICE, remaining);
      this.ai.update(slice);
      remaining -= slice;
    }

    const actions: UnitAction[] = [];
    for (const unit of this.world.players) {
      if (unit.team !== this.team || !unit.alive) continue;

      const moveTarget = unit.moveTarget;
      if (moveTarget && moveTarget !== moveTargets.get(unit.id)) {
        // Pure report: the internal AI already applied this order, so the
        // adapter's tryMove re-issue is guaranteed to succeed or the
        // environment falls back to tryHold — neither changes behavior.
        actions.push({ type: 'move', unitId: unit.id, x: moveTarget.x, y: moveTarget.y });
        continue;
      }

      const threat = this.nearestOpponent(unit.id);
      if (threat && unit.throwCooldown <= 0) {
        // Mid-power fallback so the adapter gating (cooldown/state) matches the
        // throw the AI itself would have attempted this decision.
        actions.push({ type: 'throw', unitId: unit.id, x: threat.x, y: threat.y, power: 0.5 });
      } else {
        actions.push({ type: 'noop', unitId: unit.id });
      }
    }

    return { actions };
  }

  private nearestOpponent(unitId: number): { x: number; y: number } | null {
    const unit = this.world.getPlayer(unitId);
    if (!unit) return null;

    let nearest: { x: number; y: number } | null = null;
    let bestDistanceSq = Number.POSITIVE_INFINITY;
    for (const opponent of this.world.players) {
      if (opponent.team !== this.opponent || !opponent.alive) continue;
      const distanceSq = unit.position.distanceToSq(opponent.position);
      if (distanceSq < bestDistanceSq) {
        bestDistanceSq = distanceSq;
        nearest = { x: opponent.position.x, y: opponent.position.y };
      }
    }
    return nearest;
  }
}
