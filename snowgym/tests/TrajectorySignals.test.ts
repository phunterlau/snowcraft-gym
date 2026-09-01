import { describe, expect, it } from 'vitest';
import type { PlanSnapshot } from '../orchestration/runtime/PlanStore';
import {
  TRAJECTORY_DIGEST_VERSION,
  type TrajectoryDigest,
  type TrajectoryProgress,
} from '../orchestration/trajectory/TrajectoryMonitor';
import { TrajectorySignalDetector } from '../orchestration/trajectory/TrajectorySignals';

describe('TrajectorySignalDetector', () => {
  it('debounces stalls through activation grace and recovery hysteresis', () => {
    const detector = new TrajectorySignalDetector({
      activationGraceTicks: 60,
      recoveryTicks: 60,
    });
    const snapshot = planSnapshot(1, 0);

    expect(detector.evaluate(digest({ endTick: 59, progress: 'stalled' }), snapshot)).toEqual([]);
    expect(detector.evaluate(digest({ endTick: 60, progress: 'stalled' }), snapshot)).toEqual([
      expect.objectContaining({
        trigger: 'plan_stalled',
        tick: 60,
        planVersion: 1,
        roles: ['main'],
      }),
    ]);
    expect(detector.evaluate(digest({ endTick: 66, progress: 'stalled' }), snapshot)).toEqual([]);

    expect(detector.evaluate(digest({ endTick: 70, progress: 'stable' }), snapshot)).toEqual([]);
    expect(detector.evaluate(digest({ endTick: 129, progress: 'stable' }), snapshot)).toEqual([]);
    expect(detector.evaluate(digest({ endTick: 130, progress: 'stable' }), snapshot)).toEqual([]);
    expect(detector.evaluate(digest({ endTick: 131, progress: 'stalled' }), snapshot)).toEqual([
      expect.objectContaining({ trigger: 'plan_stalled', roles: ['main'] }),
    ]);
  });

  it('requires both a minimum rejection count and rejection fraction', () => {
    const detector = new TrajectorySignalDetector({
      activationGraceTicks: 0,
      minimumRejectedActions: 3,
      rejectedActionFraction: 0.3,
    });
    const snapshot = planSnapshot(1, 0);

    expect(
      detector.evaluate(digest({ endTick: 6, issuedMoves: 10, rejectedMoves: 2 }), snapshot),
    ).toEqual([]);
    expect(
      detector.evaluate(digest({ endTick: 12, issuedMoves: 20, rejectedMoves: 3 }), snapshot),
    ).toEqual([]);
    expect(
      detector.evaluate(digest({ endTick: 18, issuedMoves: 10, rejectedMoves: 3 }), snapshot),
    ).toEqual([expect.objectContaining({ trigger: 'action_rejection_repeated', roles: ['main'] })]);
  });

  it('ignores stale evidence and clears all latches on a new plan version', () => {
    const detector = new TrajectorySignalDetector({ activationGraceTicks: 0 });
    const first = planSnapshot(1, 0);
    expect(detector.evaluate(digest({ endTick: 6, progress: 'stalled' }), first)).toHaveLength(1);
    expect(detector.evaluate(digest({ endTick: 12, progress: 'stalled' }), first)).toEqual([]);

    const second = planSnapshot(2, 12);
    expect(detector.evaluate(digest({ endTick: 18, progress: 'stalled' }), second)).toEqual([]);
    expect(
      detector.evaluate(digest({ endTick: 18, planVersion: 2, progress: 'stalled' }), second),
    ).toHaveLength(1);
  });

  it('validates debounce thresholds', () => {
    expect(() => new TrajectorySignalDetector({ activationGraceTicks: -1 })).toThrow(
      'activationGraceTicks',
    );
    expect(() => new TrajectorySignalDetector({ minimumRejectedActions: 0 })).toThrow(
      'minimumRejectedActions',
    );
    expect(() => new TrajectorySignalDetector({ rejectedActionFraction: 1.1 })).toThrow(
      'rejectedActionFraction',
    );
  });
});

function digest(options: {
  endTick: number;
  planVersion?: number;
  progress?: TrajectoryProgress;
  issuedMoves?: number;
  rejectedMoves?: number;
}): TrajectoryDigest {
  return {
    schemaVersion: TRAJECTORY_DIGEST_VERSION,
    planVersion: options.planVersion ?? 1,
    startTick: Math.max(0, options.endTick - 30),
    endTick: options.endTick,
    decisions: 5,
    groups: [
      {
        role: 'main',
        mission: 'advance',
        assigned: 3,
        livingStart: 3,
        livingEnd: 3,
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

function planSnapshot(version: number, activatedAtTick: number): PlanSnapshot {
  return {
    version,
    activatedAtTick,
    plan: {
      envelope: {
        planId: `plan-${version}`,
        source: { requestId: `request-${version}`, sourceTick: activatedAtTick },
        decision: {
          schemaVersion: 'snowgym.command-plan.v0',
          intentSummary: null,
          groups: [
            {
              role: 'main',
              allocationWeight: 1,
              selection: 'balanced',
              order: {
                mission: 'advance',
                objective: { kind: 'region', region: 'center_lane' },
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
        },
      },
      groups: [],
    },
  };
}
