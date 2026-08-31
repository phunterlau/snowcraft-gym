import { describe, expect, it } from 'vitest';
import schema from '../orchestration/command/command-plan.schema.json';
import {
  COMMAND_PLAN_VERSION,
  GROUP_ROLES,
  GROUP_SELECTIONS,
  type CommandPlan,
  type GroupCommand,
} from '../orchestration/command/CommandPlan';
import {
  CommandPlanValidationError,
  parseCommandPlan,
  validateCommandPlan,
} from '../orchestration/command/PlanValidator';

describe('CommandPlan validator', () => {
  it('accepts a bounded group plan and canonicalizes role order', () => {
    const parsed = parseCommandPlan({
      schemaVersion: COMMAND_PLAN_VERSION,
      intentSummary: 'Pin the center and preserve a reserve.',
      groups: [group('reserve', 1), group('main', 6), group('maneuver', 3)],
    });

    expect(parsed.groups.map(({ role }) => role)).toEqual(['main', 'maneuver', 'reserve']);
    expect(parsed.groups.map(({ allocationWeight }) => allocationWeight)).toEqual([6, 3, 1]);
  });

  it('rejects unknown fields, including attempts to issue individual commands', () => {
    const input = validPlanInput();
    const result = validateCommandPlan({
      ...input,
      planId: 'model-owned-id',
      groups: [{ ...input.groups[0], unitId: 7 }],
    });

    expect(result).toMatchObject({
      ok: false,
      issues: expect.arrayContaining([
        { path: '$.planId', message: 'is not allowed' },
        { path: '$.groups[0].unitId', message: 'is not allowed' },
      ]),
    });
  });

  it('enforces mission-specific objectives and approaches', () => {
    const input = validPlanInput();
    const result = validateCommandPlan({
      ...input,
      groups: [
        {
          ...input.groups[0],
          order: {
            ...input.groups[0].order,
            mission: 'withdraw',
            objective: { kind: 'region', region: 'enemy_backfield' },
            approach: 'left_flank',
          },
        },
      ],
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.issues.map(({ path }) => path)).toEqual(
        expect.arrayContaining([
          '$.groups[0].order.objective.region',
          '$.groups[0].order.approach',
        ]),
      );
    }
  });

  it('requires unique roles, a main group, and an acyclic support graph', () => {
    const duplicate = validateCommandPlan({
      schemaVersion: COMMAND_PLAN_VERSION,
      intentSummary: null,
      groups: [group('main', 1), group('main', 1)],
    });
    expect(duplicate.ok).toBe(false);

    const noMain = validateCommandPlan({
      schemaVersion: COMMAND_PLAN_VERSION,
      intentSummary: null,
      groups: [group('reserve', 1)],
    });
    expect(noMain).toMatchObject({
      ok: false,
      issues: expect.arrayContaining([{ path: '$.groups', message: 'must contain the main role' }]),
    });

    const cycle = validateCommandPlan({
      schemaVersion: COMMAND_PLAN_VERSION,
      intentSummary: null,
      groups: [supportGroup('main', 'reserve'), supportGroup('reserve', 'main')],
    });
    expect(cycle).toMatchObject({
      ok: false,
      issues: expect.arrayContaining([
        { path: '$.groups', message: 'support relationships must not form a cycle' },
      ]),
    });
  });

  it('bounds weights, group count, and trace summary length', () => {
    expect(() =>
      parseCommandPlan({
        schemaVersion: COMMAND_PLAN_VERSION,
        intentSummary: ` ${'x'.repeat(160)}`,
        groups: [group('main', 0), group('maneuver', 1), group('reserve', 1), group('main', 1)],
      }),
    ).toThrow(CommandPlanValidationError);
  });

  it('keeps the checked-in strict schema aligned with public enums', () => {
    const definitions = schema.$defs;
    expect(schema.additionalProperties).toBe(false);
    expect(schema.required).toEqual(['schemaVersion', 'intentSummary', 'groups']);
    expect(definitions.group.additionalProperties).toBe(false);
    expect(definitions.group.properties.role.enum).toEqual(GROUP_ROLES);
    expect(definitions.group.properties.selection.enum).toEqual(GROUP_SELECTIONS);
    expect(definitions.group.properties.allocationWeight).toMatchObject({
      type: 'integer',
      minimum: 1,
      maximum: 10,
    });
    for (const name of [
      'engagement',
      'enemyClusterObjective',
      'regionObjective',
      'currentPositionObjective',
      'allyGroupObjective',
      'engageOrder',
      'advanceOrder',
      'holdOrder',
      'withdrawOrder',
      'supportOrder',
    ] as const) {
      expect(definitions[name].additionalProperties).toBe(false);
    }
  });
});

function validPlanInput(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [group('main', 1)],
  };
}

function group(role: GroupCommand['role'], allocationWeight: number): GroupCommand {
  return {
    role,
    allocationWeight,
    selection: 'balanced',
    order: {
      mission: 'engage',
      objective: { kind: 'enemy_cluster', select: 'largest' },
      approach: 'direct',
      engagement: engagement(),
    },
  };
}

function supportGroup(role: GroupCommand['role'], target: GroupCommand['role']): GroupCommand {
  return {
    ...group(role, 1),
    order: {
      mission: 'support',
      objective: { kind: 'ally_group', role: target },
      approach: 'direct',
      engagement: engagement(),
    },
  };
}

function engagement(): GroupCommand['order']['engagement'] {
  return {
    posture: 'balanced',
    fire: 'focus',
    preferredRange: 'medium',
    cohesion: 'normal',
  };
}
