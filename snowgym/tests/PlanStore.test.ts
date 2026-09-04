import { describe, expect, it } from 'vitest';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlanEnvelope,
  type GroupCommand,
} from '../orchestration/command/CommandPlan';
import { PlanStore, type GroundedPlan } from '../orchestration/runtime/PlanStore';

describe('PlanStore', () => {
  it('atomically swaps immutable snapshots with monotonic versions', () => {
    const initial = groundedPlan('plan-1', 10);
    const store = new PlanStore(initial, 12);
    const first = store.current();

    expect(first).toMatchObject({ activatedAtTick: 12, version: 1 });
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.plan.groups[0].assignment.unitIds)).toBe(true);

    const second = store.activate(groundedPlan('plan-2', 20), 25);
    expect(second).toMatchObject({ activatedAtTick: 25, version: 2 });
    expect(store.current()).toBe(second);
    expect(first.plan.envelope.planId).toBe('plan-1');
    expect(store.current().plan.envelope.planId).toBe('plan-2');
  });

  it('detaches stored state from caller-owned objects', () => {
    const source = groundedPlan('plan-1', 10);
    const store = new PlanStore(source, 10);
    const mutable = source.groups[0].assignment.unitIds as number[];
    mutable.push(99);

    expect(store.current().plan.groups[0].assignment.unitIds).toEqual([1]);
  });

  it('accepts an equivalent grounded command regardless of object key insertion order', () => {
    const source = groundedPlan('plan-1', 10);
    const command = source.groups[0].command;
    const reordered: GroupCommand = {
      selection: command.selection,
      order: command.order,
      allocationWeight: command.allocationWeight,
      role: command.role,
    };

    expect(
      () => new PlanStore({ ...source, groups: [{ ...source.groups[0], command: reordered }] }, 10),
    ).not.toThrow();
  });

  it('rejects invalid activation ticks and incomplete grounding', () => {
    expect(() => new PlanStore(groundedPlan('plan-1', 10), -1)).toThrow(RangeError);
    expect(() => new PlanStore({ ...groundedPlan('plan-1', 10), groups: [] }, 10)).toThrow(
      'grounded groups must match command groups',
    );

    const invalidSource = groundedPlan('plan-1', 10);
    const emptyRequest = {
      ...invalidSource,
      envelope: {
        ...invalidSource.envelope,
        source: { ...invalidSource.envelope.source, requestId: '' },
      },
    };
    expect(() => new PlanStore(emptyRequest, 10)).toThrow('requestId must not be empty');
  });

  it('rejects duplicate unit assignment and command/envelope mismatches', () => {
    const source = groundedPlan('plan-1', 10);
    const reserveCommand: GroupCommand = { ...groupCommand(), role: 'reserve' };
    const twoGroupPlan: GroundedPlan = {
      envelope: {
        ...source.envelope,
        decision: {
          ...source.envelope.decision,
          groups: [source.envelope.decision.groups[0], reserveCommand],
        },
      },
      groups: [
        source.groups[0],
        {
          role: 'reserve',
          command: reserveCommand,
          assignment: { role: 'reserve', unitIds: [1] },
          objective: source.groups[0].objective,
          activationAnchor: { x: 0, y: 0 },
        },
      ],
    };
    expect(() => new PlanStore(twoGroupPlan, 10)).toThrow('unit 1 is assigned more than once');

    const mismatched = groundedPlan('plan-2', 20);
    const alteredCommand: GroupCommand = {
      ...mismatched.groups[0].command,
      allocationWeight: 2,
    };
    expect(
      () =>
        new PlanStore(
          {
            ...mismatched,
            groups: [{ ...mismatched.groups[0], command: alteredCommand }],
          },
          20,
        ),
    ).toThrow('grounded command does not match envelope group main');
  });
});

function groundedPlan(planId: string, sourceTick: number): GroundedPlan {
  const command = groupCommand();
  const envelope: CommandPlanEnvelope = {
    planId,
    source: { requestId: `request-${planId}`, sourceTick },
    decision: {
      schemaVersion: COMMAND_PLAN_VERSION,
      intentSummary: null,
      groups: [command],
    },
  };
  return {
    envelope,
    groups: [
      {
        role: 'main',
        command,
        assignment: { role: 'main', unitIds: [1] },
        objective: {
          kind: 'enemy_cluster',
          selector: 'nearest',
          anchor: { x: 4, y: 0 },
          enemyIds: [2],
        },
        activationAnchor: { x: -10, y: 0 },
      },
    ],
  };
}

function groupCommand(): GroupCommand {
  return {
    role: 'main',
    allocationWeight: 1,
    selection: 'balanced',
    order: {
      mission: 'engage',
      objective: { kind: 'enemy_cluster', select: 'nearest' },
      approach: 'direct',
      engagement: {
        posture: 'balanced',
        fire: 'focus',
        preferredRange: 'medium',
        cohesion: 'normal',
      },
    },
  };
}
