import type { TeamAction, UnitAction } from '../../actions/UnitAction';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import {
  parseReplayRecording,
  REPLAY_FORMAT,
  type ReplayRecording,
} from '../../replay/ReplayRecording';
import { createMapScenario } from '../../scenarios/Scenario';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
  type GroupRole,
} from '../command/CommandPlan';
import { parseCommandPlan } from '../command/PlanValidator';
import { PlanAwareTeamController } from '../execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../grounding/PlanGrounder';
import { PlanStore, type GroundedPlan } from '../runtime/PlanStore';

export interface CommandedReplayOptions {
  readonly seed?: number;
  readonly maxDecisions?: number;
}

export type ActionCounts = Readonly<Record<UnitAction['type'], number>>;

export interface CommandedReplayResult {
  readonly replay: ReplayRecording;
  readonly plan: GroundedPlan;
  readonly actionsByRole: Readonly<Record<GroupRole, ActionCounts>>;
  readonly rejectedActions: number;
}

/** Runs the first deterministic split-force commander scenario without a server or renderer. */
export function buildCommandedTenVsTenReplay(
  options: CommandedReplayOptions = {},
): CommandedReplayResult {
  const seed = options.seed ?? 42;
  const scenario = createMapScenario('arena6.json', {
    name: 'commanded-winter-front-10v10',
    seed,
    blueUnits: 10,
    redUnits: 10,
  });
  const environment = new SnowEnvironment({ scenario, decisionHz: 10, redDifficulty: 'easy' });
  let observation = environment.reset(seed);
  let status = environment.status();
  const decision = commandedTenVsTenPlan();
  const envelope: CommandPlanEnvelope = {
    planId: `c1-winter-front-seed-${seed}`,
    source: {
      requestId: `rule-c1-seed-${seed}`,
      sourceTick: observation.tick,
      sourceStateHash: status.stateHash,
    },
    decision,
  };
  const grounded = new PlanGrounder().ground(envelope, observation);
  const controller = new PlanAwareTeamController(
    new PlanStore(grounded, observation.tick),
    new ReactiveUnitPolicy(),
  );
  const rolesByUnit = new Map<number, GroupRole>();
  for (const group of grounded.groups) {
    for (const unitId of group.assignment.unitIds) rolesByUnit.set(unitId, group.role);
  }

  const frames = [observation];
  const actions: TeamAction[] = [];
  const stateHashes = [status.stateHash];
  const actionsByRole = emptyActionCounts();
  let rejectedActions = 0;
  const maxDecisions = positiveInteger(options.maxDecisions ?? 10_000, 'maxDecisions');

  while (!status.terminated && !status.truncated && actions.length < maxDecisions) {
    const action = controller.act(observation, 1 / environment.decisionHz);
    tallyActions(action, rolesByUnit, actionsByRole);
    const result = environment.step(action);
    rejectedActions += result.info.actionResults.filter(({ accepted }) => !accepted).length;
    actions.push(action);
    observation = result.observation;
    frames.push(observation);
    status = environment.status();
    stateHashes.push(status.stateHash);
  }
  if (!status.terminated && !status.truncated) {
    throw new RangeError(`episode did not complete within ${maxDecisions} decisions`);
  }

  const replay = parseReplayRecording({
    format: REPLAY_FORMAT,
    apiVersion: status.apiVersion,
    simulationVersion: status.simulationVersion,
    stateHashVersion: status.stateHashVersion,
    upstreamBaseCommit: status.upstreamBaseCommit,
    scenario: status.scenario,
    seed: status.seed,
    simulationHz: status.simulationHz,
    decisionHz: status.decisionHz,
    ticksPerDecision: status.ticksPerDecision,
    configuration: status.configuration,
    frames,
    actions,
    stateHashes,
    outcome: {
      decisions: actions.length,
      terminated: status.terminated,
      truncated: status.truncated,
      winner: status.winner,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      finalTick: status.tick,
    },
  });
  return { replay, plan: grounded, actionsByRole, rejectedActions };
}

export function commandedTenVsTenPlan(): CommandPlan {
  return parseCommandPlan({
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: 'Fix the center, maneuver left, and retain a supporting reserve.',
    groups: [
      {
        role: 'main',
        allocationWeight: 6,
        selection: 'balanced',
        order: {
          mission: 'engage',
          objective: { kind: 'enemy_cluster', select: 'largest' },
          approach: 'direct',
          engagement: {
            posture: 'balanced',
            fire: 'focus',
            preferredRange: 'medium',
            cohesion: 'normal',
          },
        },
      },
      {
        role: 'maneuver',
        allocationWeight: 3,
        selection: 'nearest_left_lane',
        order: {
          mission: 'engage',
          objective: { kind: 'enemy_cluster', select: 'weakest' },
          approach: 'left_flank',
          engagement: {
            posture: 'aggressive',
            fire: 'distributed',
            preferredRange: 'close',
            cohesion: 'loose',
          },
        },
      },
      {
        role: 'reserve',
        allocationWeight: 1,
        selection: 'rearline',
        order: {
          mission: 'support',
          objective: { kind: 'ally_group', role: 'main' },
          approach: 'direct',
          engagement: {
            posture: 'conservative',
            fire: 'opportunistic',
            preferredRange: 'long',
            cohesion: 'tight',
          },
        },
      },
    ],
  });
}

function emptyActionCounts(): Record<GroupRole, Record<UnitAction['type'], number>> {
  const one = (): Record<UnitAction['type'], number> => ({
    noop: 0,
    hold: 0,
    move: 0,
    throw: 0,
  });
  return { main: one(), maneuver: one(), reserve: one() };
}

function tallyActions(
  action: TeamAction,
  rolesByUnit: ReadonlyMap<number, GroupRole>,
  counts: Record<GroupRole, Record<UnitAction['type'], number>>,
): void {
  for (const unitAction of action.actions) {
    const role = rolesByUnit.get(unitAction.unitId);
    if (role) counts[role][unitAction.type]++;
  }
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
}
