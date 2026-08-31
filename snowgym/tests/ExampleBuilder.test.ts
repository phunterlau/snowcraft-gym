import { describe, expect, it } from 'vitest';
import { buildReplayExample } from '../examples/ExampleBuilder';

describe('replay example builder', () => {
  it('builds deterministic configurable open-arena examples', () => {
    const options = {
      blueUnits: 2,
      redUnits: 4,
      seed: 7,
      maxTicks: 1_200,
    } as const;
    const first = buildReplayExample(options);
    const second = buildReplayExample(options);

    expect(second).toEqual(first);
    expect(first.configuration).toMatchObject({ blueUnits: 2, redUnits: 4, map: null });
    expect(first.outcome.terminated || first.outcome.truncated).toBe(true);
    expect(first.stateHashes).toHaveLength(first.frames.length);
  });

  it('selects native map spawns for an asymmetric mapped example', () => {
    const replay = buildReplayExample({
      blueUnits: 5,
      redUnits: 2,
      map: 'arena6.json',
      seed: 9,
      maxTicks: 1_200,
    });

    expect(replay.configuration).toMatchObject({
      blueUnits: 5,
      redUnits: 2,
      map: 'arena6.json',
    });
    expect(replay.frames[0].allies).toHaveLength(5);
    expect(replay.frames[0].enemies).toHaveLength(2);
    expect(replay.frames[0].obstacles).toHaveLength(27);
  });

  it('rejects map rosters above native spawn capacity', () => {
    expect(() => buildReplayExample({ blueUnits: 4, redUnits: 3, map: 'arena1.json' })).toThrow(
      'blueUnits must be at most 3 on map "arena1.json"',
    );
  });
});
