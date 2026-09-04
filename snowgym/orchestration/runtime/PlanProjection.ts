import type { Observation } from '../../observations/Observation';
import { TargetResolver } from '../grounding/TargetResolver';
import type { PlanSnapshot } from './PlanStore';

/** Project immutable activation state onto current entity-backed objectives. */
export function refreshPlanObjectives(
  snapshot: PlanSnapshot,
  observation: Observation,
  resolver = new TargetResolver(),
): PlanSnapshot {
  const assignments = snapshot.plan.groups.map(({ assignment }) => assignment);
  return {
    ...snapshot,
    plan: {
      ...snapshot.plan,
      groups: snapshot.plan.groups.map((group) => ({
        ...group,
        objective: resolver.refresh(group.objective, observation, assignments),
      })),
    },
  };
}
