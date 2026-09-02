import type { Observation, UnitObservation } from '../../observations/Observation';
import {
  APPROACHES,
  COHESION_LEVELS,
  COMMAND_PLAN_VERSION,
  FIRE_POLICIES,
  POSTURES,
  PREFERRED_RANGES,
  type CommandPlan,
  type EngagementPolicy,
  type GroupCommand,
  type GroupSelection,
} from '../../orchestration/command/CommandPlan';
import { parseCommandPlan } from '../../orchestration/command/PlanValidator';
import { PlanGrounder } from '../../orchestration/grounding/PlanGrounder';

export const SYNTHETIC_PLAN_CURRICULUM_FORMAT =
  'snowgym.synthetic-plan-curriculum.v0' as const;

export interface SyntheticPlanCurriculumOptions {
  readonly baseSeed: number;
  readonly sampleCount: number;
  readonly sourceStateHash?: string;
}

export interface SyntheticPlanSample {
  readonly sourceSeed: number;
  readonly planId: string;
  readonly plan: CommandPlan;
  readonly assignments: readonly {
    readonly role: GroupCommand['role'];
    readonly unitIds: readonly number[];
  }[];
}

export interface SyntheticPlanCurriculum {
  readonly format: typeof SYNTHETIC_PLAN_CURRICULUM_FORMAT;
  readonly baseSeed: number;
  readonly sampleCount: number;
  readonly source: {
    readonly tick: number;
    readonly stateHash?: string;
    readonly arena: Observation['arena'];
    readonly allies: readonly SyntheticUnitSource[];
    readonly enemies: readonly SyntheticUnitSource[];
  };
  readonly samples: readonly SyntheticPlanSample[];
}

interface SyntheticUnitSource {
  readonly id: number;
  readonly x: number;
  readonly y: number;
  readonly health: number;
  readonly alive: boolean;
}

/** Generate schema-valid symbolic plans and deterministic grounded assignments. */
export function generateSyntheticPlanCurriculum(
  observation: Observation,
  options: SyntheticPlanCurriculumOptions,
): SyntheticPlanCurriculum {
  validateInputs(observation, options);
  const grounder = new PlanGrounder();
  const samples = Array.from({ length: options.sampleCount }, (_, index) => {
    const sourceSeed = options.baseSeed + index;
    const plan = syntheticPlan(sourceSeed);
    const planId = `synthetic-plan-${sourceSeed}`;
    const grounded = grounder.ground(
      {
        planId,
        source: {
          requestId: `synthetic-request-${sourceSeed}`,
          sourceTick: observation.tick,
          ...(options.sourceStateHash === undefined
            ? {}
            : { sourceStateHash: options.sourceStateHash }),
        },
        decision: plan,
      },
      observation,
    );
    return {
      sourceSeed,
      planId,
      plan,
      assignments: grounded.groups.map(({ role, assignment }) => ({
        role,
        unitIds: [...assignment.unitIds],
      })),
    };
  });
  return {
    format: SYNTHETIC_PLAN_CURRICULUM_FORMAT,
    baseSeed: options.baseSeed,
    sampleCount: options.sampleCount,
    source: {
      tick: observation.tick,
      ...(options.sourceStateHash === undefined
        ? {}
        : { stateHash: options.sourceStateHash }),
      arena: { ...observation.arena },
      allies: observation.allies.map(unitSource),
      enemies: observation.enemies.map(unitSource),
    },
    samples,
  };
}

function syntheticPlan(seed: number): CommandPlan {
  const random = mulberry32(seed);
  const engagement = (): EngagementPolicy => ({
    posture: pick(POSTURES, random),
    fire: pick(FIRE_POLICIES, random),
    preferredRange: pick(PREFERRED_RANGES, random),
    cohesion: pick(COHESION_LEVELS, random),
  });
  const selection = (): GroupSelection =>
    pick(
      ['balanced', 'frontline', 'rearline', 'healthiest', 'nearest_left_lane', 'nearest_right_lane'],
      random,
    );
  const variant = Math.abs(seed) % 6;
  let groups: readonly GroupCommand[] = [];
  switch (variant) {
    case 0:
      groups = [
        {
          role: 'main',
          allocationWeight: 1,
          selection: selection(),
          order: {
            mission: 'engage',
            objective: { kind: 'enemy_cluster', select: 'nearest' },
            approach: 'direct',
            engagement: engagement(),
          },
        },
      ];
      break;
    case 1:
      groups = [
        {
          role: 'main',
          allocationWeight: 2,
          selection: 'nearest_left_lane',
          order: {
            mission: 'engage',
            objective: { kind: 'enemy_cluster', select: 'weakest' },
            approach: pick(['left_flank', 'right_flank'] as const, random),
            engagement: { ...engagement(), fire: 'focus' },
          },
        },
        {
          role: 'maneuver',
          allocationWeight: 1,
          selection: 'nearest_right_lane',
          order: {
            mission: 'engage',
            objective: { kind: 'enemy_cluster', select: 'largest' },
            approach: pick(['right_flank', 'avoid_center'] as const, random),
            engagement: { ...engagement(), fire: 'distributed' },
          },
        },
      ];
      break;
    case 2:
      groups = [
        {
          role: 'main',
          allocationWeight: 1,
          selection: selection(),
          order: {
            mission: 'advance',
            objective: {
              kind: 'region',
              region: pick(['left_lane', 'center_lane', 'right_lane', 'enemy_backfield'] as const, random),
            },
            approach: pick(APPROACHES, random),
            engagement: engagement(),
          },
        },
      ];
      break;
    case 3:
      groups = [
        {
          role: 'main',
          allocationWeight: 1,
          selection: selection(),
          order: {
            mission: 'hold',
            objective: { kind: 'current_position' },
            approach: 'direct',
            engagement: engagement(),
          },
        },
      ];
      break;
    case 4:
      groups = [
        {
          role: 'main',
          allocationWeight: 1,
          selection: selection(),
          order: {
            mission: 'withdraw',
            objective: { kind: 'region', region: 'own_backfield' },
            approach: pick(['direct', 'avoid_center'] as const, random),
            engagement: { ...engagement(), posture: 'conservative' },
          },
        },
      ];
      break;
    case 5:
      groups = [
        {
          role: 'main',
          allocationWeight: 6,
          selection: 'frontline',
          order: {
            mission: 'engage',
            objective: { kind: 'enemy_cluster', select: 'nearest' },
            approach: 'direct',
            engagement: engagement(),
          },
        },
        {
          role: 'maneuver',
          allocationWeight: 3,
          selection: 'nearest_right_lane',
          order: {
            mission: 'advance',
            objective: { kind: 'region', region: 'right_lane' },
            approach: 'right_flank',
            engagement: engagement(),
          },
        },
        {
          role: 'reserve',
          allocationWeight: 1,
          selection: 'healthiest',
          order: {
            mission: 'support',
            objective: { kind: 'ally_group', role: 'main' },
            approach: 'avoid_center',
            engagement: engagement(),
          },
        },
      ];
      break;
  }
  return parseCommandPlan({
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: `Synthetic curriculum variant ${variant}`,
    groups,
  });
}

function validateInputs(
  observation: Observation,
  options: SyntheticPlanCurriculumOptions,
): void {
  if (!Number.isSafeInteger(options.baseSeed)) {
    throw new RangeError('baseSeed must be a safe integer');
  }
  if (!Number.isSafeInteger(options.sampleCount) || options.sampleCount <= 0) {
    throw new RangeError('sampleCount must be a positive safe integer');
  }
  if (options.sampleCount - 1 > Number.MAX_SAFE_INTEGER - options.baseSeed) {
    throw new RangeError('source seed range exceeds safe integers');
  }
  if (observation.allies.filter(({ alive }) => alive).length < 3) {
    throw new RangeError('synthetic plan curriculum requires at least three living allies');
  }
  if (!observation.enemies.some(({ alive }) => alive)) {
    throw new RangeError('synthetic plan curriculum requires a living enemy');
  }
  if (options.sourceStateHash !== undefined && options.sourceStateHash.length === 0) {
    throw new RangeError('sourceStateHash must not be empty');
  }
}

function unitSource(unit: UnitObservation): SyntheticUnitSource {
  return {
    id: unit.id,
    x: unit.x,
    y: unit.y,
    health: unit.health,
    alive: unit.alive,
  };
}

function pick<Value>(values: readonly Value[], random: () => number): Value {
  return values[Math.floor(random() * values.length)];
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
