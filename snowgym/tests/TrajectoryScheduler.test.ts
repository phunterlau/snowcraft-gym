import { describe, expect, it } from 'vitest';
import type { Observation } from '../observations/Observation';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
} from '../orchestration/command/CommandPlan';
import type {
  CommanderClient,
  CommanderRequest,
  CommanderResponse,
} from '../orchestration/commander/CommanderClient';
import { PlanGrounder } from '../orchestration/grounding/PlanGrounder';
import { PlanLifecycle } from '../orchestration/lifecycle/PlanLifecycle';
import { PlanStore } from '../orchestration/runtime/PlanStore';
import { CommanderScheduler } from '../orchestration/scheduler/CommanderScheduler';
import {
  TRAJECTORY_DIGEST_VERSION,
  type TrajectoryDigest,
} from '../orchestration/trajectory/TrajectoryMonitor';
import { TrajectorySignalDetector } from '../orchestration/trajectory/TrajectorySignals';
import { observationWith } from './orchestrationTestHelpers';

describe('trajectory-aware CommanderScheduler', () => {
  it('requests a soft replan without replacing the current plan with fallback', () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new DeferredCommander();
    const scheduler = trajectoryScheduler(client, store);
    const trajectory = digest({ endTick: 60, progress: 'stalled' });

    scheduler.tick(atTick(initial, 60), trajectory);

    expect(store.current()).toMatchObject({
      version: 1,
      plan: { envelope: { planId: 'initial-plan' } },
    });
    expect(client.requests).toHaveLength(1);
    expect(client.requests[0]).toMatchObject({
      triggers: ['plan_stalled'],
      trajectory: { schemaVersion: TRAJECTORY_DIGEST_VERSION, endTick: 60 },
    });
    expect(scheduler.events()).toEqual([
      expect.objectContaining({
        type: 'trajectory_signal',
        trigger: 'plan_stalled',
        roles: ['main'],
      }),
      expect.objectContaining({
        type: 'request_started',
        triggers: ['plan_stalled'],
      }),
    ]);
  });

  it('coalesces a second trajectory problem and starts a follow-up after activation', async () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new DeferredCommander();
    const scheduler = trajectoryScheduler(client, store);

    scheduler.tick(atTick(initial, 60), digest({ endTick: 60, progress: 'stalled' }));
    scheduler.tick(atTick(initial, 66), digest({ endTick: 66, issuedMoves: 10, rejectedMoves: 5 }));
    expect(client.requests).toHaveLength(1);
    expect(scheduler.status().pendingTriggers).toEqual(['action_rejection_repeated']);

    client.resolve(0, { decision: oneGroupPlan() });
    await flushPromises();
    scheduler.tick(atTick(initial, 72), digest({ endTick: 72 }));

    expect(store.current().version).toBe(2);
    expect(client.requests).toHaveLength(2);
    expect(client.requests[1]).toMatchObject({
      requestId: 'commander-request-2',
      triggers: ['action_rejection_repeated'],
      summary: { sourceTick: 72 },
      trajectory: { endTick: 66 },
      previousPlanOutcome: {
        schemaVersion: 'snowgym.plan-outcome.v0',
        planVersion: 1,
        outcome: 'superseded',
        stalledRoles: ['main'],
      },
    });
  });

  it('retains hard lifecycle fallback semantics while attaching prior trajectory evidence', () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new DeferredCommander();
    const lifecycle = new PlanLifecycle(store, undefined, undefined, undefined, {
      maxPlanAgeTicks: 60,
    });
    const scheduler = new CommanderScheduler(client, lifecycle, {
      responseTimeoutTicks: 600,
      trajectorySignalDetector: new TrajectorySignalDetector(),
    });

    scheduler.tick(atTick(initial, 60), digest({ endTick: 60 }));

    expect(store.current()).toMatchObject({
      version: 2,
      plan: { envelope: { planId: 'fallback-60-1' } },
    });
    expect(client.requests[0]).toMatchObject({
      triggers: ['plan_expired'],
      trajectory: { planVersion: 1, endTick: 60 },
    });
  });

  it('caps provider attempts even when an earlier request fails', async () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new FailingCommander();
    const scheduler = new CommanderScheduler(
      client,
      new PlanLifecycle(store, undefined, undefined, undefined, { maxPlanAgeTicks: 1_000 }),
      { responseTimeoutTicks: 600, maximumRequests: 1 },
    );

    scheduler.notify('plan_stalled', initial);
    await flushPromises();
    scheduler.tick(atTick(initial, 6));
    scheduler.notify('action_rejection_repeated', atTick(initial, 12));

    expect(client.calls).toBe(1);
    expect(scheduler.status()).toMatchObject({ requestsStarted: 1, maximumRequests: 1 });
    expect(scheduler.events()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'request_failed' }),
        expect.objectContaining({
          type: 'request_limit_reached',
          maximumRequests: 1,
          droppedTriggers: ['action_rejection_repeated'],
        }),
      ]),
    );
  });
});

class DeferredCommander implements CommanderClient {
  readonly requests: CommanderRequest[] = [];
  private readonly pending: Array<(response: CommanderResponse) => void> = [];

  plan(request: CommanderRequest): Promise<CommanderResponse> {
    this.requests.push(structuredClone(request));
    return new Promise((resolve) => this.pending.push(resolve));
  }

  resolve(index: number, response: CommanderResponse): void {
    this.pending[index](response);
  }
}

class FailingCommander implements CommanderClient {
  calls = 0;

  plan(): Promise<CommanderResponse> {
    this.calls++;
    return Promise.reject(new Error('mock provider failure'));
  }
}

function trajectoryScheduler(client: CommanderClient, store: PlanStore): CommanderScheduler {
  return new CommanderScheduler(
    client,
    new PlanLifecycle(store, undefined, undefined, undefined, { maxPlanAgeTicks: 1_000 }),
    {
      minimumRequestIntervalTicks: 0,
      responseTimeoutTicks: 600,
      trajectorySignalDetector: new TrajectorySignalDetector({ activationGraceTicks: 0 }),
    },
  );
}

function storeWith(decision: CommandPlan, observation: Observation): PlanStore {
  const envelope: CommandPlanEnvelope = {
    planId: 'initial-plan',
    source: { requestId: 'initial-request', sourceTick: observation.tick },
    decision,
  };
  return new PlanStore(new PlanGrounder().ground(envelope, observation), observation.tick);
}

function oneGroupPlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [
      {
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
      },
    ],
  };
}

function digest(options: {
  endTick: number;
  progress?: 'stable' | 'stalled';
  issuedMoves?: number;
  rejectedMoves?: number;
}): TrajectoryDigest {
  return {
    schemaVersion: TRAJECTORY_DIGEST_VERSION,
    planVersion: 1,
    startTick: Math.max(0, options.endTick - 30),
    endTick: options.endTick,
    decisions: 5,
    groups: [
      {
        role: 'main',
        mission: 'engage',
        assigned: 1,
        livingStart: 1,
        livingEnd: 1,
        progress: options.progress ?? 'stable',
        objectiveDistanceDelta: 0,
        enemyHealthDelta: 0,
        ownHealthDelta: 0,
        cohesionDelta: 0,
        issuedActions: {
          noop: 0,
          hold: 0,
          move: options.issuedMoves ?? 0,
          throw: 0,
        },
        rejectedActions: {
          noop: 0,
          hold: 0,
          move: options.rejectedMoves ?? 0,
          throw: 0,
        },
        stuckFraction: options.progress === 'stalled' ? 1 : 0,
      },
    ],
  };
}

function atTick(observation: Observation, tick: number): Observation {
  return { ...structuredClone(observation), tick };
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}
