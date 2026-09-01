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
import { MockCommander } from '../orchestration/commander/MockCommander';
import { PlanAwareTeamController } from '../orchestration/execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../orchestration/execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../orchestration/grounding/PlanGrounder';
import { PlanLifecycle } from '../orchestration/lifecycle/PlanLifecycle';
import { PlanStore } from '../orchestration/runtime/PlanStore';
import { CommanderScheduler } from '../orchestration/scheduler/CommanderScheduler';
import { SnowEnvironment } from '../core/SnowEnvironment';
import { hashObservation } from '../reproducibility/StateHash';
import { createOpenScenario } from '../scenarios/Scenario';
import { observationWith } from './orchestrationTestHelpers';

describe('CommanderScheduler', () => {
  it('keeps the synchronous controller and simulation advancing while a request is unresolved', () => {
    const environment = new SnowEnvironment({
      scenario: createOpenScenario({ seed: 42, blueUnits: 3, redUnits: 3 }),
      decisionHz: 10,
    });
    let observation = environment.reset(42);
    const store = storeWith(oneGroupPlan(), observation);
    const controller = new PlanAwareTeamController(store, new ReactiveUnitPolicy());
    const client = new DeferredCommander();
    const scheduler = new CommanderScheduler(client, new PlanLifecycle(store), {
      responseTimeoutTicks: 1_000,
    });
    scheduler.notify('plan_expired', observation);

    for (let decision = 0; decision < 20; decision++) {
      const action = controller.act(observation, 1 / environment.decisionHz);
      expect(action.actions).toHaveLength(3);
      observation = environment.step(action).observation;
      scheduler.tick(observation);
    }

    expect(observation.tick).toBe(120);
    expect(client.requests).toHaveLength(1);
    expect(scheduler.status().inFlightRequestId).toBe('commander-request-1');
    expect(store.current().version).toBe(1);
  });

  it('holds an already completed response until deterministic simulated latency elapses', async () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new MockCommander(() => ({ decision: oneGroupPlan() }), {
      latencyMs: 0,
      sleep: async () => undefined,
    });
    const scheduler = new CommanderScheduler(client, new PlanLifecycle(store), {
      minimumResponseLatencyTicks: 30,
      responseTimeoutTicks: 60,
    });
    scheduler.notify('objective_completed', initial);
    await flushPromises();

    scheduler.tick(atTick(initial, 29));
    expect(store.current().version).toBe(1);
    scheduler.tick(atTick(initial, 30));

    expect(store.current()).toMatchObject({
      version: 2,
      activatedAtTick: 30,
      plan: { envelope: { planId: 'commander-plan-1' } },
    });
    expect(scheduler.events().at(-1)).toMatchObject({
      type: 'response_processed',
      tick: 30,
      status: 'accepted',
      sourceAgeTicks: 30,
    });
  });

  it('coalesces duplicate in-flight triggers into one cooldown-governed follow-up', async () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new DeferredCommander();
    const scheduler = new CommanderScheduler(client, new PlanLifecycle(store), {
      minimumRequestIntervalTicks: 10,
      responseTimeoutTicks: 100,
    });
    scheduler.notify('plan_expired', initial);
    scheduler.notify('group_eliminated', atTick(initial, 1));
    scheduler.notify('group_eliminated', atTick(initial, 2));

    expect(client.requests).toHaveLength(1);
    expect(scheduler.status().pendingTriggers).toEqual(['group_eliminated']);
    expect(scheduler.events().filter(({ type }) => type === 'trigger_coalesced')).toHaveLength(1);

    client.resolve(0, { decision: oneGroupPlan() });
    await flushPromises();
    scheduler.tick(atTick(initial, 3));
    scheduler.tick(atTick(initial, 9));
    expect(client.requests).toHaveLength(1);

    scheduler.tick(atTick(initial, 10));
    expect(client.requests).toHaveLength(2);
    expect(client.requests[1]).toMatchObject({
      requestId: 'commander-request-2',
      triggers: ['group_eliminated'],
      summary: { sourceTick: 10 },
    });
  });

  it('times out and ignores a late provider response without replacing the plan', async () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new DeferredCommander();
    const lifecycle = new PlanLifecycle(store, undefined, undefined, undefined, {
      maxPlanAgeTicks: 100,
    });
    const scheduler = new CommanderScheduler(client, lifecycle, {
      responseTimeoutTicks: 20,
    });
    scheduler.tick(atTick(initial, 100));
    const fallback = store.current();
    expect(fallback.plan.envelope.planId).toBe('fallback-100-1');
    scheduler.tick(atTick(initial, 120));

    expect(scheduler.status().inFlightRequestId).toBeNull();
    expect(scheduler.events().at(-1)).toMatchObject({
      type: 'request_timed_out',
      requestId: 'commander-request-1',
    });

    client.resolve(0, { decision: oneGroupPlan() });
    await flushPromises();
    scheduler.tick(atTick(initial, 121));

    expect(store.current()).toBe(fallback);
    expect(scheduler.events().at(-1)).toMatchObject({
      type: 'response_ignored',
      reason: 'timed_out',
    });
  });

  it('rejects invalid provider output while retaining the active snapshot', async () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const before = store.current();
    const client = new MockCommander(
      () => ({ decision: { schemaVersion: 'not-a-command-plan' } }),
      { latencyMs: 0, sleep: async () => undefined },
    );
    const scheduler = new CommanderScheduler(client, new PlanLifecycle(store), {
      responseTimeoutTicks: 20,
    });
    scheduler.notify('objective_completed', initial);
    await flushPromises();
    scheduler.tick(atTick(initial, 1));

    expect(store.current()).toBe(before);
    expect(scheduler.events().at(-1)).toMatchObject({
      type: 'response_processed',
      status: 'rejected',
    });
  });

  it('records provider failure without blocking or replacing the plan', async () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new MockCommander(
      () => {
        throw new Error('mock provider unavailable');
      },
      { latencyMs: 0, sleep: async () => undefined },
    );
    const scheduler = new CommanderScheduler(client, new PlanLifecycle(store), {
      responseTimeoutTicks: 20,
    });
    scheduler.notify('plan_expired', initial);
    await flushPromises();
    scheduler.tick(atTick(initial, 1));

    expect(store.current().version).toBe(1);
    expect(scheduler.status().inFlightRequestId).toBeNull();
    expect(scheduler.events().at(-1)).toMatchObject({
      type: 'request_failed',
      error: 'mock provider unavailable',
    });
  });

  it('automatically falls back and requests a replan when lifecycle monitoring expires a plan', () => {
    const initial = observationWith();
    const store = storeWith(oneGroupPlan(), initial);
    const client = new DeferredCommander();
    const lifecycle = new PlanLifecycle(store, undefined, undefined, undefined, {
      maxPlanAgeTicks: 10,
    });
    const scheduler = new CommanderScheduler(client, lifecycle, { responseTimeoutTicks: 100 });

    scheduler.tick(atTick(initial, 10));

    expect(store.current()).toMatchObject({
      version: 2,
      plan: { envelope: { planId: 'fallback-10-1' } },
    });
    expect(client.requests).toHaveLength(1);
    expect(client.requests[0]).toMatchObject({
      triggers: ['plan_expired'],
      summary: { sourceTick: 10 },
      currentPlan: { groups: [{ role: 'main' }] },
    });
  });

  it('replays the same simulated-latency schedule with exact states and traces', async () => {
    const first = await scheduledRun();
    const second = await scheduledRun();

    expect(second).toEqual(first);
    expect(first.events).toContainEqual(
      expect.objectContaining({
        type: 'response_processed',
        tick: 30,
        status: 'accepted',
      }),
    );
  });
});

class DeferredCommander implements CommanderClient {
  readonly requests: CommanderRequest[] = [];
  private readonly pending: Array<{
    resolve: (response: CommanderResponse) => void;
    reject: (error: unknown) => void;
  }> = [];

  plan(request: CommanderRequest): Promise<CommanderResponse> {
    this.requests.push(structuredClone(request));
    return new Promise((resolve, reject) => this.pending.push({ resolve, reject }));
  }

  resolve(index: number, response: CommanderResponse): void {
    this.pending[index].resolve(response);
  }
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

function atTick(observation: Observation, tick: number): Observation {
  return { ...observation, tick };
}

async function flushPromises(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

async function scheduledRun(): Promise<{
  hashes: string[];
  events: ReturnType<CommanderScheduler['events']>;
}> {
  const environment = new SnowEnvironment({
    scenario: createOpenScenario({ seed: 7, blueUnits: 3, redUnits: 3 }),
    decisionHz: 10,
  });
  let observation = environment.reset(7);
  const store = storeWith(oneGroupPlan(), observation);
  const controller = new PlanAwareTeamController(store, new ReactiveUnitPolicy());
  const client = new MockCommander(() => ({ decision: oneGroupPlan() }), {
    latencyMs: 0,
    sleep: async () => undefined,
  });
  const scheduler = new CommanderScheduler(client, new PlanLifecycle(store), {
    minimumResponseLatencyTicks: 30,
    responseTimeoutTicks: 120,
  });
  scheduler.notify('objective_completed', observation);
  await flushPromises();
  const hashes = [hashObservation(observation)];
  for (let decision = 0; decision < 12; decision++) {
    observation = environment.step(controller.act(observation)).observation;
    scheduler.tick(observation);
    hashes.push(hashObservation(observation));
  }
  return { hashes, events: scheduler.events() };
}
