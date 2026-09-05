import type { Observation, UnitObservation } from '../../observations/Observation';
import { ReactiveUnitPolicy, throwRange } from '../execution/ReactiveUnitPolicy';
import type { UnitPolicyContext } from '../execution/UnitPolicy';
import type { GroupAssignment } from '../grounding/GroupAllocator';
import { TargetResolver, type ResolvedObjective } from '../grounding/TargetResolver';
import type { PlanSnapshot } from '../runtime/PlanStore';

/** Opt-in audit policy. Dodge, readiness and normal throws are handled by the base policy. */
export class LocalFirePolicy extends ReactiveUnitPolicy {
  readonly shots: { tick: number; unitId: number; targetId: number }[] = [];

  protected alternateThrowTarget(context: UnitPolicyContext): UnitObservation | null {
    const { self, observation, group } = context;
    const range = throwRange(group.command.order.engagement.preferredRange);
    const living = observation.enemies.filter((unit) => unit.alive);
    const assigned = new Set(group.candidateEnemyIds);
    const distance = (unit: UnitObservation) => Math.hypot(unit.x - self.x, unit.y - self.y);
    if (living.some((unit) => assigned.has(unit.id) && distance(unit) <= range)) return null;
    const alternate = living
      .filter((unit) => !assigned.has(unit.id) && distance(unit) <= range)
      .sort((a, b) => distance(a) - distance(b) || a.id - b.id)[0];
    if (!alternate) return null;
    this.shots.push({ tick: observation.tick, unitId: self.id, targetId: alternate.id });
    return alternate;
  }
}

/** Binding overlay owned by one branch; no PlanStore activation or assignment mutation. */
export class BindingRefreshResolver extends TargetResolver {
  readonly revision = 1;
  readonly interventionTick: number;
  readonly replacements: {
    role: string;
    original: ResolvedObjective;
    replacement: ResolvedObjective;
  }[];
  private readonly bindings = new Map<ResolvedObjective, ResolvedObjective>();

  constructor(snapshot: PlanSnapshot, observation: Observation) {
    super();
    this.interventionTick = observation.tick;
    const assignments = snapshot.plan.groups.map((group) => group.assignment);
    this.replacements = snapshot.plan.groups.flatMap((group) => {
      if (group.objective.kind !== 'enemy_cluster') return [];
      const replacement = observation.enemies.some((unit) => unit.alive)
        ? this.resolve(group.command, observation, assignments)
        : group.objective;
      this.bindings.set(group.objective, replacement);
      return [
        {
          role: group.role,
          original: structuredClone(group.objective),
          replacement: structuredClone(replacement),
        },
      ];
    });
  }

  refresh(
    objective: ResolvedObjective,
    observation: Observation,
    assignments: readonly GroupAssignment[] = [],
  ): ResolvedObjective {
    return super.refresh(this.bindings.get(objective) ?? objective, observation, assignments);
  }
}
