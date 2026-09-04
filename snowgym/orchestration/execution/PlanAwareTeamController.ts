import type { TeamAction, UnitAction } from '../../actions/UnitAction';
import type { TeamController } from '../../agents/TeamController';
import type { Observation, UnitObservation } from '../../observations/Observation';
import type { GroupRole } from '../command/CommandPlan';
import type { GroupAssignment } from '../grounding/GroupAllocator';
import { TargetResolver } from '../grounding/TargetResolver';
import { centroid } from '../grounding/TacticalFrame';
import type { PlanStore } from '../runtime/PlanStore';
import type { GroupRuntimeSummary, UnitPolicy } from './UnitPolicy';

/** Synchronous physical controller over the latest atomically activated plan. */
export class PlanAwareTeamController implements TeamController {
  constructor(
    private readonly planStore: PlanStore,
    private readonly unitPolicy: UnitPolicy,
    private readonly targetResolver = new TargetResolver(),
  ) {}

  act(observation: Observation, _dt = 0): TeamAction {
    void _dt;
    const snapshot = this.planStore.current();
    const assignments: GroupAssignment[] = snapshot.plan.groups.map(({ assignment }) => assignment);
    const runtimes = new Map<GroupRole, GroupRuntimeSummary>();

    for (const grounded of snapshot.plan.groups) {
      const livingMembers = members(grounded.assignment, observation);
      const objective = this.targetResolver.refresh(
        grounded.objective,
        observation,
        assignments,
      );
      const candidateEnemies = candidatesFor(objective, observation);
      runtimes.set(grounded.role, {
        role: grounded.role,
        command: grounded.command,
        memberIds: grounded.assignment.unitIds,
        livingMemberIds: livingMembers.map(({ id }) => id),
        centroid: centroid(livingMembers),
        objective,
        candidateEnemyIds: candidateEnemies.map(({ id }) => id),
        focusTargetId: focusTarget(candidateEnemies)?.id ?? null,
      });
    }

    const actions: UnitAction[] = [];
    for (const ally of observation.allies) {
      if (!ally.alive) continue;
      const runtime = findRuntime(ally.id, runtimes);
      actions.push(
        runtime
          ? this.unitPolicy.act({ self: ally, observation, group: runtime })
          : { type: 'hold', unitId: ally.id },
      );
    }
    return { actions };
  }
}

function members(assignment: GroupAssignment, observation: Observation): UnitObservation[] {
  const ids = new Set(assignment.unitIds);
  return observation.allies.filter((unit) => unit.alive && ids.has(unit.id));
}

function candidatesFor(
  objective: GroupRuntimeSummary['objective'],
  observation: Observation,
): UnitObservation[] {
  const living = observation.enemies.filter((enemy) => enemy.alive);
  if (objective.kind !== 'enemy_cluster') return living;
  const ids = new Set(objective.enemyIds);
  const selected = living.filter(({ id }) => ids.has(id));
  return selected.length > 0 ? selected : living;
}

function focusTarget(candidates: readonly UnitObservation[]): UnitObservation | null {
  return (
    [...candidates].sort(
      (left, right) =>
        left.health / Math.max(left.maxHealth, 1) - right.health / Math.max(right.maxHealth, 1) ||
        left.id - right.id,
    )[0] ?? null
  );
}

function findRuntime(
  unitId: number,
  runtimes: ReadonlyMap<GroupRole, GroupRuntimeSummary>,
): GroupRuntimeSummary | null {
  for (const runtime of runtimes.values()) {
    if (runtime.memberIds.includes(unitId)) return runtime;
  }
  return null;
}
