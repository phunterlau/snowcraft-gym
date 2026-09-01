import { describe, expect, it } from 'vitest';
import {
  PLAN_OUTCOME_VERSION,
  summarizePlanOutcome,
} from '../orchestration/trajectory/PlanOutcome';
import {
  TRAJECTORY_DIGEST_VERSION,
  type TrajectoryDigest,
} from '../orchestration/trajectory/TrajectoryMonitor';

describe('summarizePlanOutcome', () => {
  it('reduces group execution into bounded ID-free plan history', () => {
    const outcome = summarizePlanOutcome(digest(), 'fallback', 'group_eliminated');

    expect(outcome).toEqual({
      schemaVersion: PLAN_OUTCOME_VERSION,
      planVersion: 3,
      startTick: 60,
      endTick: 180,
      decisions: 20,
      outcome: 'fallback',
      trigger: 'group_eliminated',
      ownCasualties: 2,
      enemyHealthDelta: -40,
      rejectedActions: 3,
      stalledRoles: ['maneuver'],
    });
    expect(JSON.stringify(outcome)).not.toMatch(/unitId|enemyId|planId/);
  });
});

function digest(): TrajectoryDigest {
  return {
    schemaVersion: TRAJECTORY_DIGEST_VERSION,
    planVersion: 3,
    startTick: 60,
    endTick: 180,
    decisions: 20,
    groups: [
      group('main', 'engaging', 6, 5, -40, 1),
      group('maneuver', 'stalled', 3, 2, -40, 2),
      group('reserve', 'stable', 1, 1, -40, 0),
    ],
  };
}

function group(
  role: 'main' | 'maneuver' | 'reserve',
  progress: 'engaging' | 'stalled' | 'stable',
  livingStart: number,
  livingEnd: number,
  enemyHealthDelta: number,
  rejectedMoves: number,
): TrajectoryDigest['groups'][number] {
  return {
    role,
    mission: role === 'reserve' ? 'support' : 'engage',
    assigned: livingStart,
    livingStart,
    livingEnd,
    progress,
    objectiveDistanceDelta: 0,
    enemyHealthDelta,
    ownHealthDelta: 0,
    cohesionDelta: 0,
    issuedActions: { noop: 0, hold: 0, move: 10, throw: 2 },
    rejectedActions: { noop: 0, hold: 0, move: rejectedMoves, throw: 0 },
    stuckFraction: progress === 'stalled' ? 1 : 0,
  };
}
