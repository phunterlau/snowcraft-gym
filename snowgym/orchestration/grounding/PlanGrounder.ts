import type { Observation } from '../../observations/Observation';
import type { CommandPlanEnvelope, GroupRole } from '../command/CommandPlan';
import type { GroundedPlan } from '../runtime/PlanStore';
import { GroupAllocator } from './GroupAllocator';
import { TargetResolver, type ResolvedObjective } from './TargetResolver';
import { centroid } from './TacticalFrame';

/** Converts one validated symbolic plan into stable assignments and initial objectives. */
export class PlanGrounder {
  constructor(
    private readonly allocator = new GroupAllocator(),
    private readonly resolver = new TargetResolver(),
  ) {}

  ground(envelope: CommandPlanEnvelope, observation: Observation): GroundedPlan {
    const objectiveAnchors: Partial<Record<GroupRole, ResolvedObjective['anchor']>> = {};
    for (const group of envelope.decision.groups) {
      if (group.selection !== 'nearest_objective') continue;
      const objective = group.order.objective;
      if (objective.kind === 'ally_group' || objective.kind === 'current_position') {
        throw new PlanGroundingError(
          `group ${group.role} cannot allocate nearest an objective that depends on assignments`,
        );
      }
      objectiveAnchors[group.role] = this.resolver.resolve(group, observation).anchor;
    }

    const assignments = this.allocator.allocate(envelope.decision, observation, {
      objectiveAnchors,
    });
    return {
      envelope,
      groups: envelope.decision.groups.map((command) => {
        const assignment = assignments.find(({ role }) => role === command.role);
        if (!assignment) throw new PlanGroundingError(`missing assignment for ${command.role}`);
        return {
          role: command.role,
          command,
          assignment,
          objective: this.resolver.resolve(command, observation, assignments),
          activationAnchor: centroid(
            observation.allies.filter(
              (unit) => unit.alive && assignment.unitIds.includes(unit.id),
            ),
          ),
        };
      }),
    };
  }
}

export class PlanGroundingError extends Error {}
