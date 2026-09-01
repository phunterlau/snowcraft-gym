import { describe, expect, it } from 'vitest';
import type { ActionResult } from '../adapters/SnowCraftActionAdapter';
import type { UnitAction } from '../actions/UnitAction';
import { SnowEnvironment } from '../core/SnowEnvironment';
import type { Observation } from '../observations/Observation';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
} from '../orchestration/command/CommandPlan';
import { PlanAwareTeamController } from '../orchestration/execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../orchestration/execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../orchestration/grounding/PlanGrounder';
import { PlanStore } from '../orchestration/runtime/PlanStore';
import {
  TRAJECTORY_DIGEST_VERSION,
  TrajectoryMonitor,
} from '../orchestration/trajectory/TrajectoryMonitor';
import { hashObservation } from '../reproducibility/StateHash';
import { createOpenScenario } from '../scenarios/Scenario';
import { observationWith } from './orchestrationTestHelpers';

describe('TrajectoryMonitor', () => {
  it('is passive and reproduces identical actions, state hashes, and digests', () => {
    const baseline = runEpisode(false);
    const first = runEpisode(true);
    const second = runEpisode(true);

    expect(first.actions).toEqual(baseline.actions);
    expect(first.hashes).toEqual(baseline.hashes);
    expect(second).toEqual(first);
    expect(first.digest).toMatchObject({
      schemaVersion: TRAJECTORY_DIGEST_VERSION,
      planVersion: 1,
      decisions: 12,
      groups: [{ role: 'main', assigned: 3 }],
    });
  });

  it('keeps a bounded, detached, provider-safe group digest and resets on plan activation', () => {
    const initial = observationWith();
    const store = storeWith(advancePlan(), initial);
    const monitor = new TrajectoryMonitor({ windowDecisions: 3, minimumProgressDecisions: 2 });

    for (let decision = 0; decision < 5; decision++) {
      const before = atTick(initial, decision * 6);
      const after = atTick(initial, decision * 6 + 6);
      monitor.record({ before, after, plan: store.current(), actionResults: [acceptedMove()] });
    }

    const digest = monitor.digest();
    expect(digest).toMatchObject({ startTick: 12, endTick: 30, decisions: 3 });
    expect(JSON.stringify(digest)).not.toMatch(/unitId|enemyId|planId/);
    (digest.groups as unknown as Array<{ progress: string }>)[0].progress = 'changed';
    expect(monitor.digest().groups[0].progress).toBe('stalled');

    const nextPlan = groundEnvelope(advancePlan(), atTick(initial, 30), 'next-plan');
    store.activate(nextPlan, 30);
    monitor.record({
      before: atTick(initial, 30),
      after: atTick(initial, 36),
      plan: store.current(),
      actionResults: [acceptedMove()],
    });
    expect(monitor.digest()).toMatchObject({ planVersion: 2, startTick: 30, decisions: 1 });
  });

  it('detects sustained accepted movement without displacement and records rejections', () => {
    const initial = observationWith();
    const store = storeWith(advancePlan(), initial);
    const monitor = new TrajectoryMonitor({
      windowDecisions: 5,
      minimumProgressDecisions: 5,
      displacementEpsilon: 0.1,
    });

    for (let decision = 0; decision < 5; decision++) {
      const before = atTick(initial, decision * 6);
      const after = atTick(initial, decision * 6 + 6);
      monitor.record({ before, after, plan: store.current(), actionResults: [acceptedMove()] });
    }

    expect(monitor.digest().groups[0]).toMatchObject({
      progress: 'stalled',
      stuckFraction: 1,
      issuedActions: { move: 5 },
      rejectedActions: { move: 0 },
    });

    monitor.reset();
    for (let decision = 0; decision < 5; decision++) {
      const before = atTick(initial, decision * 6);
      const after = atTick(initial, decision * 6 + 6);
      monitor.record({ before, after, plan: store.current(), actionResults: [rejectedMove()] });
    }
    expect(monitor.digest().groups[0]).toMatchObject({
      progress: 'stable',
      stuckFraction: 0,
      issuedActions: { move: 5 },
      rejectedActions: { move: 5 },
    });
  });

  it('classifies objective progress and combat activity without treating hold as stalled', () => {
    const initial = observationWith();
    const advanceStore = storeWith(advancePlan(), initial);
    const advanceMonitor = new TrajectoryMonitor({
      windowDecisions: 5,
      minimumProgressDecisions: 5,
      progressEpsilon: 0.1,
    });
    let before = initial;
    for (let decision = 0; decision < 5; decision++) {
      const after = movedAlly(before, 0.5, 6);
      advanceMonitor.record({
        before,
        after,
        plan: advanceStore.current(),
        actionResults: [acceptedMove()],
      });
      before = after;
    }
    expect(advanceMonitor.digest().groups[0].progress).toBe('progressing');

    const engageStore = storeWith(engagePlan(), initial);
    const engageMonitor = new TrajectoryMonitor({ minimumProgressDecisions: 1 });
    engageMonitor.record({
      before: initial,
      after: damagedEnemy(initial, 10, 6),
      plan: engageStore.current(),
      actionResults: [{ action: throwAction(), accepted: true }],
    });
    expect(engageMonitor.digest().groups[0].progress).toBe('engaging');

    const holdStore = storeWith(holdPlan(), initial);
    const holdMonitor = new TrajectoryMonitor({ minimumProgressDecisions: 1 });
    holdMonitor.record({
      before: initial,
      after: atTick(initial, 6),
      plan: holdStore.current(),
      actionResults: [acceptedMove()],
    });
    expect(holdMonitor.digest().groups[0].progress).toBe('stable');
  });

  it('validates windows and monotonically ordered records', () => {
    expect(
      () => new TrajectoryMonitor({ windowDecisions: 1, minimumProgressDecisions: 2 }),
    ).toThrow('cannot exceed');
    const initial = observationWith();
    const store = storeWith(advancePlan(), initial);
    const monitor = new TrajectoryMonitor();
    expect(() => monitor.digest()).toThrow('no trajectory decisions');
    monitor.record({
      before: initial,
      after: atTick(initial, 6),
      plan: store.current(),
      actionResults: [acceptedMove()],
    });
    expect(() =>
      monitor.record({
        before: atTick(initial, 5),
        after: atTick(initial, 11),
        plan: store.current(),
        actionResults: [acceptedMove()],
      }),
    ).toThrow('ordered and non-overlapping');
  });
});

function runEpisode(monitored: boolean): {
  actions: UnitAction[][];
  hashes: string[];
  digest: ReturnType<TrajectoryMonitor['digest']> | null;
} {
  const environment = new SnowEnvironment({
    scenario: createOpenScenario({ seed: 91, blueUnits: 3, redUnits: 3 }),
    decisionHz: 10,
  });
  let observation = environment.reset(91);
  const store = storeWith(engagePlan(), observation);
  const controller = new PlanAwareTeamController(store, new ReactiveUnitPolicy());
  const monitor = monitored ? new TrajectoryMonitor() : null;
  const actions: UnitAction[][] = [];
  const hashes = [hashObservation(observation)];
  for (let decision = 0; decision < 12; decision++) {
    const plan = store.current();
    const before = observation;
    const action = controller.act(before);
    const result = environment.step(action);
    actions.push(structuredClone(action.actions));
    hashes.push(hashObservation(result.observation));
    if (monitor) {
      monitor.record({
        before,
        after: result.observation,
        plan,
        actionResults: result.info.actionResults,
      });
    }
    observation = result.observation;
  }
  return { actions, hashes, digest: monitor?.digest() ?? null };
}

function storeWith(decision: CommandPlan, observation: Observation): PlanStore {
  return new PlanStore(groundEnvelope(decision, observation, 'initial-plan'), observation.tick);
}

function groundEnvelope(
  decision: CommandPlan,
  observation: Observation,
  planId: string,
): ReturnType<PlanGrounder['ground']> {
  const envelope: CommandPlanEnvelope = {
    planId,
    source: { requestId: `${planId}-request`, sourceTick: observation.tick },
    decision,
  };
  return new PlanGrounder().ground(envelope, observation);
}

function advancePlan(): CommandPlan {
  return oneGroupPlan({
    mission: 'advance',
    objective: { kind: 'region', region: 'center_lane' },
    approach: 'direct',
    engagement: engagement(),
  });
}

function engagePlan(): CommandPlan {
  return oneGroupPlan({
    mission: 'engage',
    objective: { kind: 'enemy_cluster', select: 'nearest' },
    approach: 'direct',
    engagement: engagement(),
  });
}

function holdPlan(): CommandPlan {
  return oneGroupPlan({
    mission: 'hold',
    objective: { kind: 'current_position' },
    approach: 'direct',
    engagement: engagement(),
  });
}

function oneGroupPlan(order: CommandPlan['groups'][number]['order']): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [{ role: 'main', allocationWeight: 1, selection: 'balanced', order }],
  };
}

function engagement(): CommandPlan['groups'][number]['order']['engagement'] {
  return {
    posture: 'balanced' as const,
    fire: 'focus' as const,
    preferredRange: 'medium' as const,
    cohesion: 'normal' as const,
  };
}

function acceptedMove(): ActionResult {
  return { action: moveAction(), accepted: true };
}

function rejectedMove(): ActionResult {
  return { action: moveAction(), accepted: false, reason: 'unavailable' };
}

function moveAction(): UnitAction {
  return { type: 'move', unitId: 1, x: 0, y: 0 };
}

function throwAction(): UnitAction {
  return { type: 'throw', unitId: 1, x: 10, y: 0, power: 0.5 };
}

function atTick(observation: Observation, tick: number): Observation {
  return { ...structuredClone(observation), tick };
}

function movedAlly(observation: Observation, deltaX: number, deltaTick: number): Observation {
  const next = atTick(observation, observation.tick + deltaTick);
  next.allies[0].x += deltaX;
  return next;
}

function damagedEnemy(observation: Observation, damage: number, deltaTick: number): Observation {
  const next = atTick(observation, observation.tick + deltaTick);
  next.enemies[0].health -= damage;
  return next;
}
