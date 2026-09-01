import { describe, expect, it } from 'vitest';
import { runDelayedMockCommanderTenVsTen } from '../orchestration/examples/DelayedMockCommanderExample';

describe('delayed mock commander 10v10 example', () => {
  it('keeps fighting through simulated latency, activates at tick 90, and wins headlessly', async () => {
    const result = await runDelayedMockCommanderTenVsTen({ seed: 42, latencyTicks: 90 });

    expect(result.schedulerEvents).toEqual([
      expect.objectContaining({
        type: 'request_started',
        tick: 0,
        eligibleAtTick: 90,
        deadlineTick: 150,
      }),
      expect.objectContaining({
        type: 'response_processed',
        tick: 90,
        status: 'accepted',
        sourceAgeTicks: 90,
      }),
    ]);
    expect(result.finalAssignments).toEqual({
      main: [1, 2, 3, 4, 5, 6],
      maneuver: [8, 9, 10],
      reserve: [7],
    });
    expect(result).toMatchObject({
      rejectedActions: 0,
      decisions: 152,
      finalTick: 908,
      blueAlive: 9,
      redAlive: 0,
      winner: 'blue',
    });
  });

  it('replays an identical seed and latency schedule exactly', async () => {
    const first = await runDelayedMockCommanderTenVsTen({ seed: 7, latencyTicks: 60 });
    const second = await runDelayedMockCommanderTenVsTen({ seed: 7, latencyTicks: 60 });

    expect(second.stateHashes).toEqual(first.stateHashes);
    expect(second.actions).toEqual(first.actions);
    expect(second.schedulerEvents).toEqual(first.schedulerEvents);
  });
});
