import type { UnitAction } from '../../actions/UnitAction';
import type { Observation, UnitObservation } from '../../observations/Observation';
import type { GroupCommand, GroupRole } from '../command/CommandPlan';
import type { ResolvedObjective } from '../grounding/TargetResolver';
import type { Point } from '../grounding/TacticalFrame';

export interface GroupRuntimeSummary {
  readonly role: GroupRole;
  readonly command: GroupCommand;
  readonly memberIds: readonly number[];
  readonly livingMemberIds: readonly number[];
  readonly centroid: Point;
  readonly objective: ResolvedObjective;
  readonly candidateEnemyIds: readonly number[];
  readonly focusTargetId: number | null;
}

export interface UnitPolicyContext {
  readonly self: UnitObservation;
  readonly observation: Observation;
  readonly group: GroupRuntimeSummary;
}

export interface UnitPolicy {
  act(context: UnitPolicyContext): UnitAction;
}
