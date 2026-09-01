import type { Observation, UnitObservation } from '../../observations/Observation';
import type { GroupRole } from '../command/CommandPlan';
import { PlanGrounder } from '../grounding/PlanGrounder';
import { TargetResolutionError, TargetResolver } from '../grounding/TargetResolver';
import { centroid } from '../grounding/TacticalFrame';
import { PlanStore, type PlanSnapshot } from '../runtime/PlanStore';
import { createFallbackEnvelope } from './FallbackPlan';
import { PlanReconciler, type CandidatePlanEnvelope, type PlanRepair } from './PlanReconciler';

export type LifecycleTrigger =
  | 'plan_expired'
  | 'own_force_loss_major'
  | 'group_eliminated'
  | 'objective_completed';

export interface PlanLifecycleOptions {
  readonly maxPlanAgeTicks?: number;
  readonly majorLossFraction?: number;
  readonly completionRadius?: number;
}

export type PlanActivationOutcome =
  | {
      readonly status: 'accepted' | 'repaired';
      readonly snapshot: PlanSnapshot;
      readonly repairs: readonly PlanRepair[];
      readonly sourceAgeTicks: number;
    }
  | { readonly status: 'rejected'; readonly snapshot: PlanSnapshot; readonly error: string };

export type PlanLifecycleEvent =
  | {
      readonly type: 'candidate_activated';
      readonly tick: number;
      readonly candidatePlanId: string;
      readonly previousPlanId: string;
      readonly activePlanId: string;
      readonly version: number;
      readonly sourceAgeTicks: number;
      readonly repairs: readonly PlanRepair[];
    }
  | {
      readonly type: 'candidate_rejected';
      readonly tick: number;
      readonly candidatePlanId: string;
      readonly activePlanId: string;
      readonly version: number;
      readonly error: string;
    }
  | {
      readonly type: 'fallback_activated';
      readonly tick: number;
      readonly trigger: LifecycleTrigger;
      readonly previousPlanId: string;
      readonly activePlanId: string;
      readonly version: number;
    };

/** Owns synchronous activation policy; asynchronous scheduling is introduced in C3. */
export class PlanLifecycle {
  private readonly maxPlanAgeTicks: number;
  private readonly majorLossFraction: number;
  private readonly completionRadius: number;
  private readonly trace: PlanLifecycleEvent[] = [];
  private fallbackSequence = 0;

  constructor(
    private readonly store: PlanStore,
    private readonly reconciler = new PlanReconciler(),
    private readonly grounder = new PlanGrounder(),
    private readonly resolver = new TargetResolver(),
    options: PlanLifecycleOptions = {},
  ) {
    this.maxPlanAgeTicks = positiveInteger(options.maxPlanAgeTicks ?? 720, 'maxPlanAgeTicks');
    this.majorLossFraction = fraction(options.majorLossFraction ?? 0.5, 'majorLossFraction');
    this.completionRadius = positive(options.completionRadius ?? 1, 'completionRadius');
  }

  activateCandidate(
    candidate: CandidatePlanEnvelope,
    observation: Observation,
  ): PlanActivationOutcome {
    const previous = this.store.current();
    try {
      const reconciled = this.reconciler.reconcile(candidate, observation);
      const snapshot = this.store.activate(reconciled.plan, observation.tick);
      const status = reconciled.repairs.length === 0 ? 'accepted' : 'repaired';
      this.trace.push({
        type: 'candidate_activated',
        tick: observation.tick,
        candidatePlanId: candidate.planId,
        previousPlanId: previous.plan.envelope.planId,
        activePlanId: snapshot.plan.envelope.planId,
        version: snapshot.version,
        sourceAgeTicks: reconciled.sourceAgeTicks,
        repairs: structuredClone(reconciled.repairs),
      });
      return {
        status,
        snapshot,
        repairs: reconciled.repairs,
        sourceAgeTicks: reconciled.sourceAgeTicks,
      };
    } catch (error) {
      const message = errorMessage(error);
      this.trace.push({
        type: 'candidate_rejected',
        tick: observation.tick,
        candidatePlanId: candidate.planId,
        activePlanId: previous.plan.envelope.planId,
        version: previous.version,
        error: message,
      });
      return { status: 'rejected', snapshot: previous, error: message };
    }
  }

  evaluate(observation: Observation): readonly LifecycleTrigger[] {
    if (observation.match.blueAlive === 0 || observation.match.redAlive === 0) return [];
    const snapshot = this.store.current();
    if (observation.tick < snapshot.activatedAtTick) {
      throw new RangeError('observation tick cannot precede plan activation');
    }
    const triggers: LifecycleTrigger[] = [];
    if (observation.tick - snapshot.activatedAtTick >= this.maxPlanAgeTicks) {
      triggers.push('plan_expired');
    }

    const assignedIds = snapshot.plan.groups.flatMap(({ assignment }) => assignment.unitIds);
    const livingIds = new Set(observation.allies.filter(({ alive }) => alive).map(({ id }) => id));
    const livingAssigned = assignedIds.filter((id) => livingIds.has(id)).length;
    if (livingAssigned / Math.max(assignedIds.length, 1) <= this.majorLossFraction) {
      triggers.push('own_force_loss_major');
    }
    if (
      snapshot.plan.groups.some(
        ({ assignment }) =>
          assignment.unitIds.length > 0 && !assignment.unitIds.some((id) => livingIds.has(id)),
      )
    ) {
      triggers.push('group_eliminated');
    }
    if (snapshot.plan.groups.every((group) => this.groupCompleted(group.role, observation))) {
      triggers.push('objective_completed');
    }
    return triggers;
  }

  maintain(observation: Observation): PlanSnapshot | null {
    const trigger = this.evaluate(observation)[0];
    if (!trigger) return null;
    const previous = this.store.current();
    const fallback = createFallbackEnvelope(observation, trigger, ++this.fallbackSequence);
    const grounded = this.grounder.ground(fallback, observation);
    const snapshot = this.store.activate(grounded, observation.tick);
    this.trace.push({
      type: 'fallback_activated',
      tick: observation.tick,
      trigger,
      previousPlanId: previous.plan.envelope.planId,
      activePlanId: snapshot.plan.envelope.planId,
      version: snapshot.version,
    });
    return snapshot;
  }

  events(): readonly PlanLifecycleEvent[] {
    return structuredClone(this.trace);
  }

  private groupCompleted(role: GroupRole, observation: Observation): boolean {
    const snapshot = this.store.current();
    const group = snapshot.plan.groups.find((candidate) => candidate.role === role);
    if (!group) return false;
    const order = group.command.order;
    if (order.mission === 'hold') return false;
    if (order.mission === 'engage') return !observation.enemies.some(({ alive }) => alive);
    const assignments = snapshot.plan.groups.map(({ assignment }) => assignment);
    if (order.mission === 'support') {
      const target = assignments.find((assignment) => assignment.role === order.objective.role);
      return target === undefined || livingMembers(target.unitIds, observation).length === 0;
    }
    const members = livingMembers(group.assignment.unitIds, observation);
    if (members.length === 0) return false;
    try {
      const objective = this.resolver.resolve(group.command, observation, assignments);
      const center = centroid(members);
      return (
        Math.hypot(center.x - objective.anchor.x, center.y - objective.anchor.y) <=
        this.completionRadius
      );
    } catch (error) {
      if (error instanceof TargetResolutionError) return false;
      throw error;
    }
  }
}

function livingMembers(ids: readonly number[], observation: Observation): UnitObservation[] {
  const selected = new Set(ids);
  return observation.allies.filter((unit) => unit.alive && selected.has(unit.id));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(`${name} must be positive`);
  return value;
}

function fraction(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0 || value >= 1) {
    throw new RangeError(`${name} must be in (0, 1)`);
  }
  return value;
}

function positive(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive`);
  return value;
}
