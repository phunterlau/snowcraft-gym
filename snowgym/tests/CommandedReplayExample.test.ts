import { describe, expect, it } from 'vitest';
import { buildCommandedTenVsTenReplay } from '../orchestration/examples/CommandedReplayExample';

describe('commanded 10v10 replay example', () => {
  it('runs split group missions to a deterministic blue victory without a renderer', () => {
    const result = buildCommandedTenVsTenReplay({ seed: 42 });

    expect(result.replay.configuration).toMatchObject({
      blueUnits: 10,
      redUnits: 10,
      map: 'arena6.json',
      redDifficulty: 'easy',
    });
    expect(
      result.plan.groups.map(({ role, assignment, command }) => ({
        role,
        units: assignment.unitIds.length,
        mission: command.order.mission,
      })),
    ).toEqual([
      { role: 'main', units: 6, mission: 'engage' },
      { role: 'maneuver', units: 3, mission: 'engage' },
      { role: 'reserve', units: 1, mission: 'support' },
    ]);
    expect(result.replay.outcome).toMatchObject({
      decisions: 183,
      finalTick: 1096,
      terminated: true,
      truncated: false,
      winner: 'blue',
      blueAlive: 8,
      redAlive: 0,
    });
    for (const role of ['main', 'maneuver', 'reserve'] as const) {
      expect(result.actionsByRole[role].move).toBeGreaterThan(0);
      expect(result.actionsByRole[role].throw).toBeGreaterThan(0);
    }
    expect(result.rejectedActions).toBe(0);
  });

  it('replays the same seed to identical public-state hashes and actions', () => {
    const first = buildCommandedTenVsTenReplay({ seed: 7 });
    const replay = buildCommandedTenVsTenReplay({ seed: 7 });

    expect(replay.replay.stateHashes).toEqual(first.replay.stateHashes);
    expect(replay.replay.actions).toEqual(first.replay.actions);
    expect(replay.plan.groups.map(({ assignment }) => assignment.unitIds)).toEqual(
      first.plan.groups.map(({ assignment }) => assignment.unitIds),
    );
  });
});
