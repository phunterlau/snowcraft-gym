import { describe, expect, it } from 'vitest';
import { MockCommander } from '../orchestration/commander/MockCommander';
import { commandedTenVsTenPlan } from '../orchestration/examples/CommandedReplayExample';
import { runSingleRequestCommanderBattle } from '../orchestration/examples/OpenAICommandedBattle';

describe('single-request commanded battle', () => {
  it('enforces one request while activating asynchronously and completing headlessly', async () => {
    const client = new MockCommander(() => ({ decision: commandedTenVsTenPlan() }), {
      latencyMs: 0,
      sleep: async () => undefined,
    });
    const result = await runSingleRequestCommanderBattle(client, { seed: 42, paceMs: 0 });

    expect(result.schedulerEvents.filter(({ type }) => type === 'request_started')).toHaveLength(1);
    expect(result.schedulerEvents).toContainEqual(
      expect.objectContaining({
        type: 'response_processed',
        tick: 6,
        status: 'accepted',
      }),
    );
    expect(result.assignments).toEqual({
      main: [1, 2, 3, 4, 5, 6],
      maneuver: [8, 9, 10],
      reserve: [7],
    });
    expect(result).toMatchObject({
      rejectedActions: 0,
      blueAlive: 9,
      redAlive: 0,
      winner: 'blue',
    });
  });
});
