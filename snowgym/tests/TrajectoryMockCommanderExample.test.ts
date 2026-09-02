import { describe, expect, it } from 'vitest';
import {
  runTrajectoryMockCommanderBattle,
  runTrajectoryMockCommanderTenVsTen,
} from '../orchestration/examples/TrajectoryMockCommanderExample';
import {
  CommanderTraceFormatError,
  parseCommanderTrace,
} from '../orchestration/trace/CommanderTrace';
import { commanderOverlayAtTick } from '../replay/CommanderOverlay';

describe('trajectory-aware mock commander 10v10 example', () => {
  it('uses real trajectory evidence to make multiple non-blocking commander requests', async () => {
    const result = await runTrajectoryMockCommanderTenVsTen({
      seed: 42,
      latencyTicks: 30,
      maxDecisions: 300,
    });

    expect(result.commanderRequests).toBe(3);
    expect(result.schedulerEvents.filter(({ type }) => type === 'trajectory_signal')).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'trajectory_signal', trigger: 'plan_stalled' }),
      ]),
    );
    expect(result.schedulerEvents.filter(({ type }) => type === 'request_started').length).toBe(
      result.commanderRequests,
    );
    expect(result.schedulerEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'response_processed', status: 'accepted' }),
        expect.objectContaining({
          type: 'request_limit_reached',
          maximumRequests: 3,
        }),
      ]),
    );
    expect(result.trajectoryDigests.length).toBe(result.decisions);
    expect(result.planTraces[0].planId).toBe('trajectory-initial-direct');
    expect(result.replay.frames).toHaveLength(result.decisions + 1);
    expect(result.commanderTrace.replay).toMatchObject({
      format: result.replay.format,
      scenario: result.replay.scenario,
      seed: result.replay.seed,
      finalTick: result.replay.outcome.finalTick,
      finalStateHash: result.stateHashes.at(-1),
    });
    expect(parseCommanderTrace(structuredClone(result.commanderTrace), result.replay)).toEqual(
      result.commanderTrace,
    );
    expect(JSON.stringify(result.commanderTrace)).not.toMatch(/"(?:unitId|unitIds|enemyId)"/);

    const opening = commanderOverlayAtTick(result.commanderTrace, 0);
    expect(opening.plan.version).toBe(1);
    expect(opening.trajectory).toBeNull();
    const ending = commanderOverlayAtTick(result.commanderTrace, result.finalTick);
    expect(ending.plan.version).toBeGreaterThan(1);
    expect(ending.trajectory?.endTick).toBe(result.finalTick);
    expect(ending.events.length).toBeGreaterThan(0);

    const mismatched = structuredClone(result.commanderTrace) as unknown as Record<string, unknown>;
    (mismatched.replay as Record<string, unknown>).seed = 999;
    expect(() => parseCommanderTrace(mismatched, result.replay)).toThrow(CommanderTraceFormatError);
  });

  it('replays identical actions, states, trajectory digests, and scheduler events', async () => {
    const first = await runTrajectoryMockCommanderTenVsTen({
      seed: 7,
      latencyTicks: 30,
      maxDecisions: 250,
    });
    const second = await runTrajectoryMockCommanderTenVsTen({
      seed: 7,
      latencyTicks: 30,
      maxDecisions: 250,
    });

    expect(second.actions).toEqual(first.actions);
    expect(second.stateHashes).toEqual(first.stateHashes);
    expect(second.trajectoryDigests).toEqual(first.trajectoryDigests);
    expect(second.schedulerEvents).toEqual(first.schedulerEvents);
    expect(second.lifecycleEvents).toEqual(first.lifecycleEvents);
    expect(second.planTraces).toEqual(first.planTraces);
    expect(second.commanderTrace).toEqual(first.commanderTrace);
  });

  it('uses the understrength opening to win a bound 6v10 commander trace', async () => {
    const result = await runTrajectoryMockCommanderBattle({
      seed: 14,
      blueUnits: 6,
      redUnits: 10,
      map: 'arena6.json',
      redDifficulty: 'easy',
      latencyTicks: 30,
      maxDecisions: 400,
    });

    expect(result.replay.configuration).toMatchObject({
      blueUnits: 6,
      redUnits: 10,
      map: 'arena6.json',
      redDifficulty: 'easy',
    });
    expect(result.winner).toBe('blue');
    expect(result.blueAlive).toBe(1);
    expect(result.redAlive).toBe(0);
    expect(result.commanderRequests).toBe(2);
    expect(result.rejectedActions).toBe(0);
    expect(result.planTraces[0].planId).toBe('trajectory-initial-economy-of-force');
    expect(result.planTraces[0].decision.intentSummary).toContain('outnumbered force');
    expect(result.commanderTrace.replay.finalStateHash).toBe(result.stateHashes.at(-1));
  });

  it('rejects a roster larger than the selected map before running', async () => {
    await expect(
      runTrajectoryMockCommanderBattle({
        blueUnits: 3,
        redUnits: 10,
        map: 'arena1.json',
      }),
    ).rejects.toThrow('redUnits must be at most 3');
  });
});
