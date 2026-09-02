import { describe, expect, it } from 'vitest';
import { MockCommander } from '../orchestration/commander/MockCommander';
import { commandedTenVsTenPlan } from '../orchestration/examples/CommandedReplayExample';
import { runTrajectoryCommanderBattle } from '../orchestration/examples/TrajectoryCommanderBattle';

describe('trajectory-aware commander battle', () => {
  it('runs multiple capped trajectory requests while physical control continues', async () => {
    const client = new MockCommander(
      () => ({ decision: commandedTenVsTenPlan(), metadata: { model: 'mock-live' } }),
      { latencyMs: 0, sleep: async () => undefined },
    );
    const result = await runTrajectoryCommanderBattle(client, {
      seed: 42,
      paceMs: 0,
      maximumRequests: 2,
    });

    const events = result.schedulerEvents;
    expect(events.filter(({ type }) => type === 'request_started')).toHaveLength(2);
    expect(events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'trajectory_signal', trigger: 'plan_stalled' }),
        expect.objectContaining({ type: 'response_processed', status: 'accepted' }),
      ]),
    );
    expect(result.rejectedActions).toBe(0);
    expect(result.decisions).toBeGreaterThan(0);
    expect(result.replay.frames).toHaveLength(result.decisions + 1);
    expect(result.commanderTrace.schedulerEvents).toEqual(result.schedulerEvents);
    expect(result.commanderTrace.plans.length).toBeGreaterThan(1);
    expect(result.commanderTrace.trajectoryDigests).toHaveLength(result.decisions);
    expect(JSON.stringify(result.commanderTrace)).not.toMatch(/"(?:unitId|unitIds|enemyId)"/);
  });

  it('finishes under provider failure without exceeding the attempt cap', async () => {
    const client = new MockCommander(
      () => {
        throw new Error('provider unavailable');
      },
      { latencyMs: 0, sleep: async () => undefined },
    );
    const result = await runTrajectoryCommanderBattle(client, {
      seed: 7,
      paceMs: 0,
      maximumRequests: 2,
    });

    expect(result.schedulerEvents.filter(({ type }) => type === 'request_started')).toHaveLength(2);
    expect(result.schedulerEvents.filter(({ type }) => type === 'request_failed')).toHaveLength(2);
    expect(result.rejectedActions).toBe(0);
    expect(result.commanderTrace.schedulerEvents).toEqual(result.schedulerEvents);
  });

  it('runs a configured roster and bundled map through the provider-neutral boundary', async () => {
    const client = new MockCommander(
      () => ({ decision: commandedTenVsTenPlan(), metadata: { model: 'mock-configured' } }),
      { latencyMs: 0, sleep: async () => undefined },
    );
    const result = await runTrajectoryCommanderBattle(client, {
      seed: 5,
      paceMs: 0,
      maximumRequests: 2,
      blueUnits: 3,
      redUnits: 3,
      map: 'arena4.json',
      redDifficulty: 'normal',
    });

    expect(result.replay.configuration).toMatchObject({
      blueUnits: 3,
      redUnits: 3,
      map: 'arena4.json',
      redDifficulty: 'normal',
    });
    expect(result.rejectedActions).toBe(0);
    expect(result.commanderTrace.replay.finalStateHash).toBe(result.replay.stateHashes?.at(-1));
  });
});
