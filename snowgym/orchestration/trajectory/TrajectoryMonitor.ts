import type { ActionResult } from '../../adapters/SnowCraftActionAdapter';
import type { Observation, UnitObservation } from '../../observations/Observation';
import type { UnitAction } from '../../actions/UnitAction';
import type { GroupRole } from '../command/CommandPlan';
import { TargetResolutionError, TargetResolver } from '../grounding/TargetResolver';
import { centroid, type Point } from '../grounding/TacticalFrame';
import type { PlanSnapshot } from '../runtime/PlanStore';

export const TRAJECTORY_DIGEST_VERSION = 'snowgym.trajectory-digest.v0' as const;

export type TrajectoryProgress =
  | 'insufficient_data'
  | 'progressing'
  | 'engaging'
  | 'stable'
  | 'stalled';

export type TrajectoryActionType = UnitAction['type'];

export interface TrajectoryActionCounts {
  readonly noop: number;
  readonly hold: number;
  readonly move: number;
  readonly throw: number;
}

type MutableTrajectoryActionCounts = {
  -readonly [Key in keyof TrajectoryActionCounts]: TrajectoryActionCounts[Key];
};

export interface TrajectoryGroupDigest {
  readonly role: GroupRole;
  readonly mission: PlanSnapshot['plan']['groups'][number]['command']['order']['mission'];
  readonly assigned: number;
  readonly livingStart: number;
  readonly livingEnd: number;
  readonly progress: TrajectoryProgress;
  /** Negative values mean that the group closed on its late-bound objective. */
  readonly objectiveDistanceDelta: number | null;
  /** Negative values mean that the opposing force lost health. */
  readonly enemyHealthDelta: number;
  /** Negative values mean that this group lost health. */
  readonly ownHealthDelta: number;
  /** Positive values mean the group became more spread out. */
  readonly cohesionDelta: number;
  readonly issuedActions: TrajectoryActionCounts;
  readonly rejectedActions: TrajectoryActionCounts;
  /** Living members with accepted move intent but negligible displacement. */
  readonly stuckFraction: number;
}

/** Provider-safe, bounded evidence about execution under one active plan version. */
export interface TrajectoryDigest {
  readonly schemaVersion: typeof TRAJECTORY_DIGEST_VERSION;
  readonly planVersion: number;
  readonly startTick: number;
  readonly endTick: number;
  readonly decisions: number;
  readonly groups: readonly TrajectoryGroupDigest[];
}

export interface TrajectoryRecord {
  readonly before: Observation;
  readonly after: Observation;
  readonly actionResults: readonly ActionResult[];
  /** The immutable plan snapshot used to produce this decision. */
  readonly plan: PlanSnapshot;
}

export interface TrajectoryMonitorOptions {
  readonly windowDecisions?: number;
  readonly minimumProgressDecisions?: number;
  readonly progressEpsilon?: number;
  readonly displacementEpsilon?: number;
  readonly stalledFraction?: number;
}

interface UnitMovement {
  acceptedMoves: number;
  displacement: number;
  aliveAtEnd: boolean;
}

interface GroupSample {
  readonly role: GroupRole;
  readonly mission: TrajectoryGroupDigest['mission'];
  readonly assigned: number;
  readonly livingStart: number;
  readonly livingEnd: number;
  readonly objectiveDistanceBefore: number | null;
  readonly objectiveDistanceAfter: number | null;
  readonly ownHealthBefore: number;
  readonly ownHealthAfter: number;
  readonly enemyHealthBefore: number;
  readonly enemyHealthAfter: number;
  readonly cohesionBefore: number;
  readonly cohesionAfter: number;
  readonly issuedActions: TrajectoryActionCounts;
  readonly rejectedActions: TrajectoryActionCounts;
  readonly movements: ReadonlyMap<number, UnitMovement>;
}

interface DecisionSample {
  readonly startTick: number;
  readonly endTick: number;
  readonly planVersion: number;
  readonly groups: readonly GroupSample[];
}

/**
 * Passive, host-owned trajectory telemetry. Recording a decision never mutates
 * the environment, controller, observation, action result, or active plan.
 */
export class TrajectoryMonitor {
  private readonly windowDecisions: number;
  private readonly minimumProgressDecisions: number;
  private readonly progressEpsilon: number;
  private readonly displacementEpsilon: number;
  private readonly stalledFraction: number;
  private readonly samples: DecisionSample[] = [];
  private planVersion: number | null = null;

  constructor(
    options: TrajectoryMonitorOptions = {},
    private readonly resolver = new TargetResolver(),
  ) {
    this.windowDecisions = positiveInteger(options.windowDecisions ?? 20, 'windowDecisions');
    this.minimumProgressDecisions = positiveInteger(
      options.minimumProgressDecisions ?? 5,
      'minimumProgressDecisions',
    );
    if (this.minimumProgressDecisions > this.windowDecisions) {
      throw new RangeError('minimumProgressDecisions cannot exceed windowDecisions');
    }
    this.progressEpsilon = nonNegative(options.progressEpsilon ?? 0.25, 'progressEpsilon');
    this.displacementEpsilon = nonNegative(
      options.displacementEpsilon ?? 0.2,
      'displacementEpsilon',
    );
    this.stalledFraction = fraction(options.stalledFraction ?? 0.5, 'stalledFraction');
  }

  record(record: TrajectoryRecord): TrajectoryDigest {
    validateRecord(record);
    if (this.planVersion !== record.plan.version) {
      this.samples.length = 0;
      this.planVersion = record.plan.version;
    }
    if (this.samples.length > 0 && record.before.tick < this.samples.at(-1)!.endTick) {
      throw new RangeError('trajectory records must be ordered and non-overlapping');
    }

    this.samples.push(this.sample(record));
    while (this.samples.length > this.windowDecisions) this.samples.shift();
    return this.digest();
  }

  digest(): TrajectoryDigest {
    if (this.samples.length === 0 || this.planVersion === null) {
      throw new TrajectoryMonitorError('no trajectory decisions have been recorded');
    }
    const first = this.samples[0];
    const last = this.samples.at(-1)!;
    const roles = first.groups.map(({ role }) => role);
    return {
      schemaVersion: TRAJECTORY_DIGEST_VERSION,
      planVersion: this.planVersion,
      startTick: first.startTick,
      endTick: last.endTick,
      decisions: this.samples.length,
      groups: roles.map((role) => this.groupDigest(role)),
    };
  }

  decisionCount(): number {
    return this.samples.length;
  }

  reset(): void {
    this.samples.length = 0;
    this.planVersion = null;
  }

  private sample(record: TrajectoryRecord): DecisionSample {
    const assignments = record.plan.plan.groups.map(({ assignment }) => assignment);
    const beforeById = new Map(record.before.allies.map((unit) => [unit.id, unit]));
    const afterById = new Map(record.after.allies.map((unit) => [unit.id, unit]));
    const resultsById = new Map(
      record.actionResults.map((result) => [result.action.unitId, result]),
    );
    const enemyHealthBefore = totalLivingHealth(record.before.enemies);
    const enemyHealthAfter = totalLivingHealth(record.after.enemies);

    return {
      startTick: record.before.tick,
      endTick: record.after.tick,
      planVersion: record.plan.version,
      groups: record.plan.plan.groups.map((group): GroupSample => {
        const ids = new Set(group.assignment.unitIds);
        const beforeMembers = record.before.allies.filter((unit) => ids.has(unit.id) && unit.alive);
        const afterMembers = record.after.allies.filter((unit) => ids.has(unit.id) && unit.alive);
        const issuedActions = emptyActionCounts();
        const rejectedActions = emptyActionCounts();
        const movements = new Map<number, UnitMovement>();

        for (const unitId of group.assignment.unitIds) {
          const before = beforeById.get(unitId);
          const after = afterById.get(unitId);
          const result = resultsById.get(unitId);
          if (result) {
            issuedActions[result.action.type]++;
            if (!result.accepted) rejectedActions[result.action.type]++;
          }
          movements.set(unitId, {
            acceptedMoves: result?.accepted && result.action.type === 'move' ? 1 : 0,
            displacement: before && after ? distance(before, after) : 0,
            aliveAtEnd: after?.alive ?? false,
          });
        }

        return {
          role: group.role,
          mission: group.command.order.mission,
          assigned: group.assignment.unitIds.length,
          livingStart: beforeMembers.length,
          livingEnd: afterMembers.length,
          objectiveDistanceBefore: this.objectiveDistance(
            group.command,
            beforeMembers,
            record.before,
            assignments,
          ),
          objectiveDistanceAfter: this.objectiveDistance(
            group.command,
            afterMembers,
            record.after,
            assignments,
          ),
          ownHealthBefore: totalLivingHealth(beforeMembers),
          ownHealthAfter: totalLivingHealth(afterMembers),
          enemyHealthBefore,
          enemyHealthAfter,
          cohesionBefore: spread(beforeMembers),
          cohesionAfter: spread(afterMembers),
          issuedActions,
          rejectedActions,
          movements,
        };
      }),
    };
  }

  private objectiveDistance(
    command: PlanSnapshot['plan']['groups'][number]['command'],
    members: readonly UnitObservation[],
    observation: Observation,
    assignments: PlanSnapshot['plan']['groups'][number]['assignment'][],
  ): number | null {
    if (members.length === 0 || command.order.mission === 'hold') return null;
    try {
      const objective = this.resolver.resolve(command, observation, assignments);
      return distance(centroid(members), objective.anchor);
    } catch (error) {
      if (error instanceof TargetResolutionError) return null;
      throw error;
    }
  }

  private groupDigest(role: GroupRole): TrajectoryGroupDigest {
    const samples = this.samples.map((sample) => {
      const group = sample.groups.find((candidate) => candidate.role === role);
      if (!group) throw new TrajectoryMonitorError(`group ${role} disappeared within one plan`);
      return group;
    });
    const first = samples[0];
    const last = samples.at(-1)!;
    const issuedActions = sumActionCounts(samples.map(({ issuedActions: counts }) => counts));
    const rejectedActions = sumActionCounts(samples.map(({ rejectedActions: counts }) => counts));
    const movements = new Map<number, UnitMovement>();
    for (const sample of samples) {
      for (const [unitId, movement] of sample.movements) {
        const total = movements.get(unitId) ?? {
          acceptedMoves: 0,
          displacement: 0,
          aliveAtEnd: false,
        };
        total.acceptedMoves += movement.acceptedMoves;
        total.displacement += movement.displacement;
        total.aliveAtEnd = movement.aliveAtEnd;
        movements.set(unitId, total);
      }
    }
    const eligibleMovers = [...movements.values()].filter(
      ({ acceptedMoves, aliveAtEnd }) => acceptedMoves > 0 && aliveAtEnd,
    );
    const stuckFraction =
      eligibleMovers.length === 0
        ? 0
        : eligibleMovers.filter(({ displacement }) => displacement <= this.displacementEpsilon)
            .length / eligibleMovers.length;
    const objectiveDistanceDelta = nullableDelta(
      first.objectiveDistanceBefore,
      last.objectiveDistanceAfter,
    );
    const enemyHealthDelta = last.enemyHealthAfter - first.enemyHealthBefore;
    const ownHealthDelta = last.ownHealthAfter - first.ownHealthBefore;
    const cohesionDelta = last.cohesionAfter - first.cohesionBefore;

    return {
      role,
      mission: first.mission,
      assigned: first.assigned,
      livingStart: first.livingStart,
      livingEnd: last.livingEnd,
      progress: this.progress(
        first.mission,
        samples.length,
        objectiveDistanceDelta,
        enemyHealthDelta,
        issuedActions,
        rejectedActions,
        stuckFraction,
      ),
      objectiveDistanceDelta: roundedOrNull(objectiveDistanceDelta),
      enemyHealthDelta: rounded(enemyHealthDelta),
      ownHealthDelta: rounded(ownHealthDelta),
      cohesionDelta: rounded(cohesionDelta),
      issuedActions,
      rejectedActions,
      stuckFraction: rounded(stuckFraction),
    };
  }

  private progress(
    mission: TrajectoryGroupDigest['mission'],
    decisions: number,
    objectiveDistanceDelta: number | null,
    enemyHealthDelta: number,
    issuedActions: TrajectoryActionCounts,
    rejectedActions: TrajectoryActionCounts,
    stuckFraction: number,
  ): TrajectoryProgress {
    if (decisions < this.minimumProgressDecisions) return 'insufficient_data';
    if (mission === 'hold') return 'stable';
    if (
      mission === 'engage' &&
      (enemyHealthDelta < 0 || issuedActions.throw > rejectedActions.throw)
    ) {
      return 'engaging';
    }
    if (objectiveDistanceDelta !== null && objectiveDistanceDelta <= -this.progressEpsilon) {
      return 'progressing';
    }
    if (issuedActions.move > 0 && stuckFraction >= this.stalledFraction) return 'stalled';
    return 'stable';
  }
}

export class TrajectoryMonitorError extends Error {}

function validateRecord(record: TrajectoryRecord): void {
  if (record.before.tick < 0 || record.after.tick <= record.before.tick) {
    throw new RangeError('trajectory record ticks must advance');
  }
  if (record.before.selfTeam !== record.after.selfTeam) {
    throw new RangeError('trajectory record team cannot change');
  }
  if (record.plan.activatedAtTick > record.before.tick) {
    throw new RangeError('trajectory record plan was not active at the decision tick');
  }
}

function emptyActionCounts(): MutableTrajectoryActionCounts {
  return { noop: 0, hold: 0, move: 0, throw: 0 };
}

function sumActionCounts(counts: readonly TrajectoryActionCounts[]): MutableTrajectoryActionCounts {
  const result = emptyActionCounts();
  for (const count of counts) {
    result.noop += count.noop;
    result.hold += count.hold;
    result.move += count.move;
    result.throw += count.throw;
  }
  return result;
}

function totalLivingHealth(units: readonly UnitObservation[]): number {
  return units.filter(({ alive }) => alive).reduce((sum, unit) => sum + unit.health, 0);
}

function spread(units: readonly UnitObservation[]): number {
  if (units.length === 0) return 0;
  const center = centroid(units);
  return units.reduce((sum, unit) => sum + distance(unit, center), 0) / units.length;
}

function distance(left: Point, right: Point): number {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

function nullableDelta(before: number | null, after: number | null): number | null {
  return before === null || after === null ? null : after - before;
}

function roundedOrNull(value: number | null): number | null {
  return value === null ? null : rounded(value);
}

function rounded(value: number): number {
  return Object.is(value, -0) ? 0 : Math.round(value * 1_000_000) / 1_000_000;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
}

function nonNegative(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be non-negative`);
  return value;
}

function fraction(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new RangeError(`${name} must be in (0, 1]`);
  }
  return value;
}
