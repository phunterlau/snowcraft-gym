import {
  APPROACHES,
  COHESION_LEVELS,
  COMMAND_PLAN_VERSION,
  ENEMY_CLUSTER_SELECTORS,
  FIRE_POLICIES,
  GROUP_ROLE_ORDER,
  GROUP_ROLES,
  GROUP_SELECTIONS,
  POSTURES,
  PREFERRED_RANGES,
  REGIONS,
  type CommandPlan,
  type EngagementPolicy,
  type GroupCommand,
  type GroupOrder,
  type GroupRole,
} from './CommandPlan';

export interface CommandPlanValidationIssue {
  readonly path: string;
  readonly message: string;
}

export type CommandPlanValidationResult =
  | { readonly ok: true; readonly value: CommandPlan }
  | { readonly ok: false; readonly issues: readonly CommandPlanValidationIssue[] };

export class CommandPlanValidationError extends Error {
  constructor(readonly issues: readonly CommandPlanValidationIssue[]) {
    super(issues.map(({ path, message }) => `${path}: ${message}`).join('; '));
    this.name = 'CommandPlanValidationError';
  }
}

export function validateCommandPlan(value: unknown): CommandPlanValidationResult {
  try {
    return { ok: true, value: parseCommandPlan(value) };
  } catch (error) {
    if (!(error instanceof CommandPlanValidationError)) throw error;
    return { ok: false, issues: error.issues };
  }
}

/** Parses, validates, and canonicalizes group ordering without repairing model intent. */
export function parseCommandPlan(value: unknown): CommandPlan {
  const issues: CommandPlanValidationIssue[] = [];
  const plan = readObject(value, '$', issues);
  assertKeys(plan, ['schemaVersion', 'intentSummary', 'groups'], '$', issues);

  if (plan.schemaVersion !== COMMAND_PLAN_VERSION) {
    issue(issues, '$.schemaVersion', `must equal ${COMMAND_PLAN_VERSION}`);
  }

  let intentSummary: string | null = null;
  if (plan.intentSummary !== null) {
    intentSummary = readString(plan.intentSummary, '$.intentSummary', issues, 1, 160);
    if (intentSummary !== intentSummary.trim()) {
      issue(issues, '$.intentSummary', 'must not have leading or trailing whitespace');
    }
  }

  const groupsValue = plan.groups;
  const groups: GroupCommand[] = [];
  if (!Array.isArray(groupsValue)) {
    issue(issues, '$.groups', 'must be an array');
  } else {
    if (groupsValue.length < 1 || groupsValue.length > 3) {
      issue(issues, '$.groups', 'must contain between 1 and 3 groups');
    }
    for (let index = 0; index < groupsValue.length; index++) {
      groups.push(parseGroup(groupsValue[index], `$.groups[${index}]`, issues));
    }
  }

  validateGroupGraph(groups, issues);
  if (issues.length > 0) throw new CommandPlanValidationError(issues);

  groups.sort((left, right) => GROUP_ROLE_ORDER[left.role] - GROUP_ROLE_ORDER[right.role]);
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary,
    groups,
  };
}

function parseGroup(
  value: unknown,
  path: string,
  issues: CommandPlanValidationIssue[],
): GroupCommand {
  const group = readObject(value, path, issues);
  assertKeys(group, ['role', 'allocationWeight', 'selection', 'order'], path, issues);
  const role = readEnum(group.role, GROUP_ROLES, `${path}.role`, issues);
  const allocationWeight = readInteger(
    group.allocationWeight,
    `${path}.allocationWeight`,
    issues,
    1,
    10,
  );
  const selection = readEnum(group.selection, GROUP_SELECTIONS, `${path}.selection`, issues);
  const order = parseOrder(group.order, `${path}.order`, issues);
  return { role, allocationWeight, selection, order };
}

function parseOrder(
  value: unknown,
  path: string,
  issues: CommandPlanValidationIssue[],
): GroupOrder {
  const order = readObject(value, path, issues);
  assertKeys(order, ['mission', 'objective', 'approach', 'engagement'], path, issues);
  const mission = readEnum(
    order.mission,
    ['engage', 'advance', 'hold', 'withdraw', 'support'] as const,
    `${path}.mission`,
    issues,
  );
  const engagement = parseEngagement(order.engagement, `${path}.engagement`, issues);
  const objectivePath = `${path}.objective`;
  const objective = readObject(order.objective, objectivePath, issues);

  switch (mission) {
    case 'engage': {
      assertKeys(objective, ['kind', 'select'], objectivePath, issues);
      expectLiteral(objective.kind, 'enemy_cluster', `${objectivePath}.kind`, issues);
      return {
        mission,
        objective: {
          kind: 'enemy_cluster',
          select: readEnum(
            objective.select,
            ENEMY_CLUSTER_SELECTORS,
            `${objectivePath}.select`,
            issues,
          ),
        },
        approach: readEnum(order.approach, APPROACHES, `${path}.approach`, issues),
        engagement,
      };
    }
    case 'advance': {
      const region = parseRegionObjective(objective, objectivePath, issues);
      return {
        mission,
        objective: { kind: 'region', region },
        approach: readEnum(order.approach, APPROACHES, `${path}.approach`, issues),
        engagement,
      };
    }
    case 'hold': {
      const kind = readEnum(
        objective.kind,
        ['region', 'current_position'] as const,
        `${objectivePath}.kind`,
        issues,
      );
      let holdObjective:
        | { kind: 'region'; region: (typeof REGIONS)[number] }
        | { kind: 'current_position' };
      if (kind === 'region') {
        const region = parseRegionObjective(objective, objectivePath, issues);
        holdObjective = { kind: 'region', region };
      } else {
        assertKeys(objective, ['kind'], objectivePath, issues);
        holdObjective = { kind: 'current_position' };
      }
      return {
        mission,
        objective: holdObjective,
        approach: readLiteral(order.approach, 'direct', `${path}.approach`, issues),
        engagement,
      };
    }
    case 'withdraw': {
      assertKeys(objective, ['kind', 'region'], objectivePath, issues);
      expectLiteral(objective.kind, 'region', `${objectivePath}.kind`, issues);
      return {
        mission,
        objective: {
          kind: 'region',
          region: readLiteral(objective.region, 'own_backfield', `${objectivePath}.region`, issues),
        },
        approach: readEnum(
          order.approach,
          ['direct', 'avoid_center'] as const,
          `${path}.approach`,
          issues,
        ),
        engagement,
      };
    }
    case 'support': {
      assertKeys(objective, ['kind', 'role'], objectivePath, issues);
      expectLiteral(objective.kind, 'ally_group', `${objectivePath}.kind`, issues);
      return {
        mission,
        objective: {
          kind: 'ally_group',
          role: readEnum(objective.role, GROUP_ROLES, `${objectivePath}.role`, issues),
        },
        approach: readEnum(order.approach, APPROACHES, `${path}.approach`, issues),
        engagement,
      };
    }
  }
}

function parseRegionObjective(
  objective: Record<string, unknown>,
  path: string,
  issues: CommandPlanValidationIssue[],
): (typeof REGIONS)[number] {
  assertKeys(objective, ['kind', 'region'], path, issues);
  expectLiteral(objective.kind, 'region', `${path}.kind`, issues);
  return readEnum(objective.region, REGIONS, `${path}.region`, issues);
}

function parseEngagement(
  value: unknown,
  path: string,
  issues: CommandPlanValidationIssue[],
): EngagementPolicy {
  const engagement = readObject(value, path, issues);
  assertKeys(engagement, ['posture', 'fire', 'preferredRange', 'cohesion'], path, issues);
  return {
    posture: readEnum(engagement.posture, POSTURES, `${path}.posture`, issues),
    fire: readEnum(engagement.fire, FIRE_POLICIES, `${path}.fire`, issues),
    preferredRange: readEnum(
      engagement.preferredRange,
      PREFERRED_RANGES,
      `${path}.preferredRange`,
      issues,
    ),
    cohesion: readEnum(engagement.cohesion, COHESION_LEVELS, `${path}.cohesion`, issues),
  };
}

function validateGroupGraph(
  groups: readonly GroupCommand[],
  issues: CommandPlanValidationIssue[],
): void {
  const roles = new Map<GroupRole, number>();
  for (let index = 0; index < groups.length; index++) {
    const role = groups[index].role;
    if (roles.has(role)) issue(issues, `$.groups[${index}].role`, `duplicate role ${role}`);
    else roles.set(role, index);
  }
  if (!roles.has('main')) issue(issues, '$.groups', 'must contain the main role');

  const supportEdges = new Map<GroupRole, GroupRole>();
  for (let index = 0; index < groups.length; index++) {
    const group = groups[index];
    if (
      group.selection === 'nearest_objective' &&
      (group.order.objective.kind === 'ally_group' ||
        group.order.objective.kind === 'current_position')
    ) {
      issue(
        issues,
        `$.groups[${index}].selection`,
        `nearest_objective cannot be used with ${group.order.objective.kind}`,
      );
    }
    if (group.order.mission !== 'support') continue;
    const target = group.order.objective.role;
    if (!roles.has(target)) {
      issue(issues, `$.groups[${index}].order.objective.role`, `unknown group role ${target}`);
    }
    if (target === group.role) {
      issue(issues, `$.groups[${index}].order.objective.role`, 'a group cannot support itself');
    }
    supportEdges.set(group.role, target);
  }

  for (const role of supportEdges.keys()) {
    const visited = new Set<GroupRole>();
    let cursor: GroupRole | undefined = role;
    while (cursor !== undefined) {
      if (visited.has(cursor)) {
        issue(issues, '$.groups', 'support relationships must not form a cycle');
        break;
      }
      visited.add(cursor);
      cursor = supportEdges.get(cursor);
    }
  }
}

function readObject(
  value: unknown,
  path: string,
  issues: CommandPlanValidationIssue[],
): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    issue(issues, path, 'must be an object');
    return {};
  }
  return value as Record<string, unknown>;
}

function assertKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  path: string,
  issues: CommandPlanValidationIssue[],
): void {
  const allowedSet = new Set(allowed);
  for (const key of Object.keys(value)) {
    if (!allowedSet.has(key)) issue(issues, `${path}.${key}`, 'is not allowed');
  }
  for (const key of allowed) {
    if (!(key in value)) issue(issues, `${path}.${key}`, 'is required');
  }
}

function readString(
  value: unknown,
  path: string,
  issues: CommandPlanValidationIssue[],
  minLength: number,
  maxLength: number,
): string {
  if (typeof value !== 'string') {
    issue(issues, path, 'must be a string');
    return '';
  }
  if (value.length < minLength || value.length > maxLength) {
    issue(issues, path, `length must be between ${minLength} and ${maxLength}`);
  }
  return value;
}

function readInteger(
  value: unknown,
  path: string,
  issues: CommandPlanValidationIssue[],
  minimum: number,
  maximum: number,
): number {
  if (!Number.isSafeInteger(value)) {
    issue(issues, path, 'must be a safe integer');
    return minimum;
  }
  const result = value as number;
  if (result < minimum || result > maximum) {
    issue(issues, path, `must be between ${minimum} and ${maximum}`);
  }
  return result;
}

function readEnum<const Values extends readonly string[]>(
  value: unknown,
  allowed: Values,
  path: string,
  issues: CommandPlanValidationIssue[],
): Values[number] {
  if (typeof value !== 'string' || !allowed.includes(value)) {
    issue(issues, path, `must be one of: ${allowed.join(', ')}`);
    return allowed[0];
  }
  return value as Values[number];
}

function readLiteral<const Value extends string>(
  value: unknown,
  expected: Value,
  path: string,
  issues: CommandPlanValidationIssue[],
): Value {
  expectLiteral(value, expected, path, issues);
  return expected;
}

function expectLiteral(
  value: unknown,
  expected: string,
  path: string,
  issues: CommandPlanValidationIssue[],
): void {
  if (value !== expected) issue(issues, path, `must equal ${expected}`);
}

function issue(issues: CommandPlanValidationIssue[], path: string, message: string): void {
  issues.push({ path, message });
}
