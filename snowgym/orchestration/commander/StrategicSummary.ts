import type { Observation, UnitObservation } from '../../observations/Observation';
import { hashObservation } from '../../reproducibility/StateHash';
import type { GroupRole } from '../command/CommandPlan';
import { centroid, type Point } from '../grounding/TacticalFrame';
import type { PlanSnapshot } from '../runtime/PlanStore';

export const STRATEGIC_SUMMARY_VERSION = 'snowgym.strategic-summary.v0' as const;

export interface ForceSummary {
  readonly alive: number;
  readonly healthFraction: number;
  readonly centroid: Point;
  readonly spread: number;
}

export interface StrategicGroupSummary {
  readonly role: GroupRole;
  readonly mission: PlanSnapshot['plan']['groups'][number]['command']['order']['mission'];
  readonly assigned: number;
  readonly living: number;
  readonly objectiveKind: PlanSnapshot['plan']['groups'][number]['objective']['kind'];
}

/** Compact, ID-free strategic context captured when a commander request starts. */
export interface StrategicSummary {
  readonly schemaVersion: typeof STRATEGIC_SUMMARY_VERSION;
  readonly sourceTick: number;
  readonly sourceStateHash: string;
  readonly arena: {
    readonly width: number;
    readonly height: number;
    readonly obstacleCount: number;
  };
  readonly ownForce: ForceSummary;
  readonly enemyForce: ForceSummary;
  readonly hostileProjectileCount: number;
  readonly groups: readonly StrategicGroupSummary[];
}

export function summarizeStrategy(
  observation: Observation,
  snapshot: PlanSnapshot,
): StrategicSummary {
  const livingOwnIds = new Set(observation.allies.filter(({ alive }) => alive).map(({ id }) => id));
  return {
    schemaVersion: STRATEGIC_SUMMARY_VERSION,
    sourceTick: observation.tick,
    sourceStateHash: hashObservation(observation),
    arena: {
      width: observation.arena.width,
      height: observation.arena.height,
      obstacleCount: observation.obstacles.length,
    },
    ownForce: summarizeForce(observation.allies),
    enemyForce: summarizeForce(observation.enemies),
    hostileProjectileCount: observation.projectiles.filter(
      ({ team }) => team !== observation.selfTeam,
    ).length,
    groups: snapshot.plan.groups.map(({ role, command, assignment, objective }) => ({
      role,
      mission: command.order.mission,
      assigned: assignment.unitIds.length,
      living: assignment.unitIds.filter((id) => livingOwnIds.has(id)).length,
      objectiveKind: objective.kind,
    })),
  };
}

function summarizeForce(units: readonly UnitObservation[]): ForceSummary {
  const living = units.filter(({ alive }) => alive);
  const center = centroid(living);
  return {
    alive: living.length,
    healthFraction:
      living.length === 0
        ? 0
        : living.reduce((sum, unit) => sum + unit.health / Math.max(unit.maxHealth, 1), 0) /
          living.length,
    centroid: center,
    spread:
      living.length === 0
        ? 0
        : living.reduce((sum, unit) => sum + Math.hypot(unit.x - center.x, unit.y - center.y), 0) /
          living.length,
  };
}
