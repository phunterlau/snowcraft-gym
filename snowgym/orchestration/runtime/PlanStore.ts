import type { CommandPlanEnvelope, GroupCommand, GroupRole } from '../command/CommandPlan';
import type { GroupAssignment } from '../grounding/GroupAllocator';
import type { ResolvedObjective } from '../grounding/TargetResolver';

export interface GroundedGroupPlan {
  readonly role: GroupRole;
  readonly command: GroupCommand;
  readonly assignment: GroupAssignment;
  readonly objective: ResolvedObjective;
  /** Centroid of this stable roster at activation, retained through casualties. */
  readonly activationAnchor: { readonly x: number; readonly y: number };
}

export interface GroundedPlan {
  readonly envelope: CommandPlanEnvelope;
  readonly groups: readonly GroundedGroupPlan[];
}

export interface PlanSnapshot {
  readonly plan: GroundedPlan;
  readonly activatedAtTick: number;
  readonly version: number;
}

/** Single-writer, synchronous-reader store for atomic plan activation. */
export class PlanStore {
  private snapshot: PlanSnapshot;

  constructor(initialPlan: GroundedPlan, activatedAtTick: number) {
    validateActivation(initialPlan, activatedAtTick);
    this.snapshot = immutableClone({ plan: initialPlan, activatedAtTick, version: 1 });
  }

  current(): PlanSnapshot {
    return this.snapshot;
  }

  activate(plan: GroundedPlan, activatedAtTick: number): PlanSnapshot {
    validateActivation(plan, activatedAtTick);
    const next = immutableClone({
      plan,
      activatedAtTick,
      version: this.snapshot.version + 1,
    });
    this.snapshot = next;
    return next;
  }
}

function validateActivation(plan: GroundedPlan, activatedAtTick: number): void {
  if (!Number.isSafeInteger(activatedAtTick) || activatedAtTick < 0) {
    throw new RangeError('activatedAtTick must be a non-negative safe integer');
  }
  if (plan.groups.length !== plan.envelope.decision.groups.length) {
    throw new RangeError('grounded groups must match command groups');
  }
  if (plan.envelope.planId.trim().length === 0) throw new RangeError('planId must not be empty');
  if (plan.envelope.source.requestId.trim().length === 0) {
    throw new RangeError('requestId must not be empty');
  }
  if (
    !Number.isSafeInteger(plan.envelope.source.sourceTick) ||
    plan.envelope.source.sourceTick < 0
  ) {
    throw new RangeError('sourceTick must be a non-negative safe integer');
  }
  if (
    plan.envelope.source.sourceStateHash !== undefined &&
    plan.envelope.source.sourceStateHash === ''
  ) {
    throw new RangeError('sourceStateHash must not be empty');
  }

  const assignedUnits = new Set<number>();
  const groundedRoles = new Set<GroupRole>();
  for (const command of plan.envelope.decision.groups) {
    const grounded = plan.groups.find((group) => group.role === command.role);
    if (
      !grounded ||
      grounded.command.role !== command.role ||
      grounded.assignment.role !== command.role
    ) {
      throw new RangeError(`missing or mismatched grounded group ${command.role}`);
    }
    if (groundedRoles.has(grounded.role))
      throw new RangeError(`duplicate grounded group ${grounded.role}`);
    groundedRoles.add(grounded.role);
    if (!sameJsonValue(grounded.command, command)) {
      throw new RangeError(`grounded command does not match envelope group ${command.role}`);
    }
    if (
      !Number.isFinite(grounded.activationAnchor.x) ||
      !Number.isFinite(grounded.activationAnchor.y)
    ) {
      throw new RangeError(`group ${command.role} activation anchor must be finite`);
    }
    for (const unitId of grounded.assignment.unitIds) {
      if (!Number.isSafeInteger(unitId))
        throw new RangeError('assigned unit IDs must be safe integers');
      if (assignedUnits.has(unitId))
        throw new RangeError(`unit ${unitId} is assigned more than once`);
      assignedUnits.add(unitId);
    }
  }
}

function sameJsonValue(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (typeof left !== 'object' || left === null || typeof right !== 'object' || right === null) {
    return false;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
    return left.every((value, index) => sameJsonValue(value, right[index]));
  }
  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftRecord).sort();
  const rightKeys = Object.keys(rightRecord).sort();
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every(
      (key, index) => key === rightKeys[index] && sameJsonValue(leftRecord[key], rightRecord[key]),
    )
  );
}

function immutableClone<Value>(value: Value): Value {
  return deepFreeze(structuredClone(value));
}

function deepFreeze<Value>(value: Value): Value {
  if (typeof value !== 'object' || value === null || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}
