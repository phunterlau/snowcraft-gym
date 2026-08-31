import { describe, expect, it } from 'vitest';
import type { Observation, UnitObservation } from '../observations/Observation';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
  type GroupCommand,
} from '../orchestration/command/CommandPlan';
import { PlanAwareTeamController } from '../orchestration/execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../orchestration/execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../orchestration/grounding/PlanGrounder';
import { PlanStore } from '../orchestration/runtime/PlanStore';
import { observationWith } from './orchestrationTestHelpers';

describe('PlanAwareTeamController', () => {
  it('shares one focus-fire target across a group', () => {
    const observation = combatObservation();
    observation.enemies[1].health = 20;
    const actions = controller(plan(engageOrder('focus'))).act(observation).actions;

    expect(actions).toHaveLength(2);
    expect(actions.every(({ type }) => type === 'throw')).toBe(true);
    expect(actions).toEqual([
      expect.objectContaining({ type: 'throw', unitId: 1, x: 4 }),
      expect.objectContaining({ type: 'throw', unitId: 2, x: 4 }),
    ]);
  });

  it('distributes targets deterministically by living-member order', () => {
    const actions = controller(plan(engageOrder('distributed'))).act(combatObservation()).actions;
    expect(actions).toEqual([
      expect.objectContaining({ type: 'throw', unitId: 1, x: 3 }),
      expect.objectContaining({ type: 'throw', unitId: 2, x: 4 }),
    ]);
  });

  it('dodges an immediate projectile before executing the group mission', () => {
    const observation = combatObservation();
    observation.projectiles.push({
      id: 30,
      ownerId: 10,
      team: 'red',
      x: -4,
      y: 0,
      vx: 10,
      vy: 0,
      height: 1,
      heightVelocity: 0,
    });
    const action = controller(plan(engageOrder('focus'))).act(observation).actions[0];
    expect(action).toEqual({ type: 'move', unitId: 1, x: -2, y: -2.4 });
  });

  it('uses explicit hold at a current-position objective', () => {
    const observation = observationWith({
      allies: [unit(1, 'blue', -2, 0)],
      enemies: [unit(10, 'red', 15, 0)],
    });
    const action = controller(plan(holdOrder())).act(observation).actions[0];
    expect(action).toEqual({ type: 'hold', unitId: 1 });
  });

  it('turns approach and posture commands into distinct movement targets', () => {
    const observation = observationWith({
      allies: [unit(1, 'blue', -10, 0)],
      enemies: [unit(10, 'red', 10, 0)],
    });
    const leftAggressive: GroupCommand['order'] = {
      mission: 'engage',
      objective: { kind: 'enemy_cluster', select: 'nearest' },
      approach: 'left_flank',
      engagement: {
        posture: 'aggressive',
        fire: 'focus',
        preferredRange: 'medium',
        cohesion: 'normal',
      },
    };
    const directConservative: GroupCommand['order'] = {
      ...leftAggressive,
      approach: 'direct',
      engagement: { ...leftAggressive.engagement, posture: 'conservative' },
    };
    const leftAction = controllerForObservation(plan(leftAggressive), observation).act(observation)
      .actions[0];
    const directAction = controllerForObservation(plan(directConservative), observation).act(
      observation,
    ).actions[0];

    expect(leftAction).toMatchObject({ type: 'move', unitId: 1 });
    expect(directAction).toMatchObject({ type: 'move', unitId: 1 });
    if (leftAction.type !== 'move' || directAction.type !== 'move') {
      throw new Error('expected movement actions');
    }
    expect(leftAction.y).toBeGreaterThan(directAction.y);
    expect(leftAction.x).toBeGreaterThan(directAction.x);
  });

  it('does not reassign stable group membership after a casualty', () => {
    const observation = observationWith({
      allies: [unit(1, 'blue', -4, -2), unit(2, 'blue', -4, 0), unit(3, 'blue', -4, 2)],
      enemies: [unit(10, 'red', 8, 0)],
    });
    const decision: CommandPlan = {
      schemaVersion: COMMAND_PLAN_VERSION,
      intentSummary: null,
      groups: [group('main', 2, engageOrder('focus')), group('reserve', 1, holdOrder())],
    };
    const envelope = hostEnvelope(decision);
    const grounded = new PlanGrounder().ground(envelope, observation);
    const original = grounded.groups.map(({ role, assignment }) => [role, assignment.unitIds]);
    const store = new PlanStore(grounded, 0);
    const policy = new PlanAwareTeamController(store, new ReactiveUnitPolicy());
    const casualtyId = grounded.groups[0].assignment.unitIds[0];
    const casualty = observation.allies.find(({ id }) => id === casualtyId);
    if (!casualty) throw new Error('missing casualty fixture');
    casualty.alive = false;
    casualty.state = 'defeated';

    const actions = policy.act(observation).actions;
    expect(actions.some(({ unitId }) => unitId === casualtyId)).toBe(false);
    expect(
      store.current().plan.groups.map(({ role, assignment }) => [role, assignment.unitIds]),
    ).toEqual(original);
  });
});

function controller(decision: CommandPlan): PlanAwareTeamController {
  const initial =
    decision.groups.length === 1 && decision.groups[0].order.mission === 'hold'
      ? observationWith({ allies: [unit(1, 'blue', -2, 0)], enemies: [unit(10, 'red', 15, 0)] })
      : combatObservation();
  const grounded = new PlanGrounder().ground(hostEnvelope(decision), initial);
  return new PlanAwareTeamController(new PlanStore(grounded, 0), new ReactiveUnitPolicy());
}

function controllerForObservation(
  decision: CommandPlan,
  observation: Observation,
): PlanAwareTeamController {
  const grounded = new PlanGrounder().ground(hostEnvelope(decision), observation);
  return new PlanAwareTeamController(new PlanStore(grounded, 0), new ReactiveUnitPolicy());
}

function hostEnvelope(decision: CommandPlan): CommandPlanEnvelope {
  return {
    planId: 'controller-test',
    source: { requestId: 'controller-test-request', sourceTick: 0 },
    decision,
  };
}

function plan(order: GroupCommand['order']): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [group('main', 1, order)],
  };
}

function group(
  role: GroupCommand['role'],
  allocationWeight: number,
  order: GroupCommand['order'],
): GroupCommand {
  return { role, allocationWeight, selection: 'balanced', order };
}

function engageOrder(fire: 'focus' | 'distributed'): GroupCommand['order'] {
  return {
    mission: 'engage',
    objective: { kind: 'enemy_cluster', select: 'largest' },
    approach: 'direct',
    engagement: {
      posture: 'balanced',
      fire,
      preferredRange: 'medium',
      cohesion: 'normal',
    },
  };
}

function holdOrder(): GroupCommand['order'] {
  return {
    mission: 'hold',
    objective: { kind: 'current_position' },
    approach: 'direct',
    engagement: {
      posture: 'conservative',
      fire: 'opportunistic',
      preferredRange: 'long',
      cohesion: 'tight',
    },
  };
}

function combatObservation(): Observation {
  return observationWith({
    allies: [unit(1, 'blue', -2, 0), unit(2, 'blue', -2, 1)],
    enemies: [unit(10, 'red', 3, 0), unit(11, 'red', 4, 0)],
  });
}

function unit(id: number, team: UnitObservation['team'], x: number, y: number): UnitObservation {
  return {
    id,
    team,
    x,
    y,
    vx: 0,
    vy: 0,
    health: 100,
    maxHealth: 100,
    alive: true,
    state: 'idle',
    throwCooldown: 0,
    charge: 0,
  };
}
