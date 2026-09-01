import { describe, expect, it } from 'vitest';
import { runTrajectoryMockCommanderTenVsTen } from '../orchestration/examples/TrajectoryMockCommanderExample';

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
  });
});
