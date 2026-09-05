import type { Observation, UnitObservation } from '../../observations/Observation';
import { hashObservation } from '../../reproducibility/StateHash';
import { canMove, canThrow, throwRange } from '../execution/ReactiveUnitPolicy';
import { TargetResolver } from '../grounding/TargetResolver';
import { centroid, type Point } from '../grounding/TacticalFrame';
import type { PlanSnapshot } from '../runtime/PlanStore';
import type { TrajectoryDigest } from '../trajectory/TrajectoryMonitor';

export const RECOVERY_EVIDENCE_VERSION = 'snowgym.recovery-evidence.v0' as const;
export const RECOVERY_FAMILIES = [
  'blocked_advance',
  'target_eliminated',
  'recent_casualties',
  'throws_without_damage',
] as const;
export type RecoveryFamily = (typeof RECOVERY_FAMILIES)[number];
export type RecoveryEvidence = ReturnType<typeof summarizeRecovery>;

/** Activation observation must be retained separately from the moving tactical objective. */
export function summarizeRecovery(
  observation: Observation,
  activation: Observation,
  snapshot: PlanSnapshot,
  trajectory?: TrajectoryDigest,
) {
  if (activation.tick !== snapshot.activatedAtTick || observation.tick < activation.tick)
    throw new Error('recovery activation interval mismatch');
  if (
    trajectory &&
    (trajectory.planVersion !== snapshot.version ||
      trajectory.endTick !== observation.tick ||
      trajectory.startTick < activation.tick)
  )
    throw new Error('recovery trajectory interval mismatch');
  const resolver = new TargetResolver();
  const assignments = snapshot.plan.groups.map((group) => group.assignment);
  const groups = snapshot.plan.groups.map((group) => {
    const assigned = activation.allies.filter((unit) => group.assignment.unitIds.includes(unit.id));
    const own = observation.allies.filter(
      (unit) => unit.alive && group.assignment.unitIds.includes(unit.id),
    );
    const objective = resolver.refresh(group.objective, observation, assignments);
    const targets =
      group.objective.kind === 'enemy_cluster'
        ? activation.enemies.filter(
            (unit) =>
              group.objective.kind === 'enemy_cluster' &&
              group.objective.enemyIds.includes(unit.id),
          )
        : [];
    const targetNow = observation.enemies.filter((unit) =>
      targets.some((target) => target.id === unit.id),
    );
    const tacticalTargets = observation.enemies.filter(
      (unit) =>
        unit.alive && (objective.kind !== 'enemy_cluster' || objective.enemyIds.includes(unit.id)),
    );
    const range = throwRange(group.command.order.engagement.preferredRange);
    const inRange = own.filter((unit) =>
      tacticalTargets.some((enemy) => distance(unit, enemy) <= range),
    );
    const evidence = trajectory?.groups.find((row) => row.role === group.role);
    return {
      role: group.role,
      mission: group.command.order.mission,
      assigned: assigned.length,
      living: own.length,
      livingFraction: own.length / Math.max(assigned.length, 1),
      healthFractionOfActivation: health(own) / Math.max(health(assigned), 1),
      canMoveFraction: fraction(own.filter(canMove).length, own.length),
      canThrowFraction: fraction(own.filter(canThrow).length, own.length),
      inExecutorRangeFraction: fraction(inRange.length, own.length),
      objectiveDistance: own.length ? distance(centroid(own), objective.anchor) : null,
      meanRangeExcess:
        own.length && tacticalTargets.length
          ? own.reduce(
              (sum, unit) =>
                sum +
                Math.max(
                  0,
                  Math.min(...tacticalTargets.map((enemy) => distance(unit, enemy))) - range,
                ),
              0,
            ) / own.length
          : null,
      directMovementObstructionFraction: fraction(
        own.filter((unit) =>
          segmentObstructed(unit, objective.anchor, observation, 'blocksMovement'),
        ).length,
        own.length,
      ),
      directProjectileObstructionFraction: fraction(
        inRange.filter((unit) =>
          tacticalTargets.every(
            (enemy) =>
              distance(unit, enemy) > range ||
              segmentObstructed(unit, enemy, observation, 'blocksProjectiles'),
          ),
        ).length,
        inRange.length,
      ),
      frozenTargetLiving: targets.length ? targetNow.filter((unit) => unit.alive).length : null,
      frozenTargetHealthFraction: targets.length
        ? health(targetNow) / Math.max(health(targets), 1)
        : null,
      frozenTargetEliminated: targets.length > 0 && targetNow.every((unit) => !unit.alive),
      recent: evidence
        ? {
            progress: evidence.progress,
            ownCasualties: evidence.livingStart - evidence.livingEnd,
            damageReceived: -evidence.ownHealthDelta,
            enemyHealthLoss: -evidence.enemyHealthDelta,
            acceptedThrows: evidence.issuedActions.throw - evidence.rejectedActions.throw,
            acceptedMoves: evidence.issuedActions.move - evidence.rejectedActions.move,
            stuckFraction: evidence.stuckFraction,
          }
        : null,
    };
  });
  const families: RecoveryFamily[] = [];
  for (const family of RECOVERY_FAMILIES) {
    const matched = groups.some((group) => {
      const recent = group.recent;
      if (family === 'target_eliminated')
        return group.frozenTargetEliminated && observation.enemies.some((unit) => unit.alive);
      if (!recent || (trajectory?.decisions ?? 0) < 20) return false;
      if (family === 'blocked_advance')
        return (
          recent.acceptedMoves >= 5 &&
          recent.stuckFraction >= 0.5 &&
          (group.directMovementObstructionFraction ?? 0) > 0 &&
          (group.objectiveDistance ?? 0) > 2
        );
      if (family === 'recent_casualties') return recent.ownCasualties >= 1;
      return recent.acceptedThrows >= 5 && recent.enemyHealthLoss <= 0;
    });
    if (matched) families.push(family);
  }
  return {
    schemaVersion: RECOVERY_EVIDENCE_VERSION,
    sourceTick: observation.tick,
    sourceStateHash: hashObservation(observation),
    activationTick: activation.tick,
    planVersion: snapshot.version,
    planAgeSeconds: (observation.tick - activation.tick) / observation.simulationHz,
    window: trajectory
      ? {
          startTick: trajectory.startTick,
          endTick: trajectory.endTick,
          decisions: trajectory.decisions,
        }
      : null,
    semantics: {
      executor: 'ReactiveUnitPolicy',
      geometry:
        'conservative straight segments against public AABBs; no pathfinding or ballistic arc prediction',
      readiness: 'production executor state predicates; not adapter acceptance or hit probability',
      damage: 'whole enemy force health loss, not attribution to a role or target',
      range: 'executor throw-trigger distance, not effective accuracy',
      capability: 'uncalibrated physical proxies; no mission-success probability available',
    },
    groups,
    detectedFamilies: families,
  };
}

/** Liang–Barsky segment intersection with conservative public obstacle footprints. */
export function segmentObstructed(
  a: Point,
  b: Point,
  observation: Observation,
  flag: 'blocksMovement' | 'blocksProjectiles',
): boolean {
  return observation.obstacles.some((obstacle) => {
    if (!obstacle[flag]) return false;
    let low = 0;
    let high = 1;
    for (const [origin, delta, minimum, maximum] of [
      [a.x, b.x - a.x, obstacle.x - obstacle.halfWidth, obstacle.x + obstacle.halfWidth],
      [a.y, b.y - a.y, obstacle.y - obstacle.halfHeight, obstacle.y + obstacle.halfHeight],
    ]) {
      if (Math.abs(delta) < 1e-12) {
        if (origin < minimum || origin > maximum) return false;
      } else {
        const left = (minimum - origin) / delta;
        const right = (maximum - origin) / delta;
        low = Math.max(low, Math.min(left, right));
        high = Math.min(high, Math.max(left, right));
        if (low > high) return false;
      }
    }
    return true;
  });
}
function fraction(count: number, total: number): number | null {
  return total ? count / total : null;
}
function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function health(units: readonly UnitObservation[]): number {
  return units.reduce((sum, unit) => sum + (unit.alive ? Math.max(unit.health, 0) : 0), 0);
}
