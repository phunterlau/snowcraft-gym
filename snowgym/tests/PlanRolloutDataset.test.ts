import { describe, expect, it } from 'vitest';
import { createOpenScenario } from '../scenarios/Scenario';
import {
  auditPlanRolloutDataset,
  buildPlanRolloutDataset,
} from '../training/plan/PlanRolloutDataset';

describe('PlanRolloutDataset', () => {
  const options = {
    scenario: createOpenScenario({
      name: 'paired-plan-rollout-test',
      seed: 77,
      blueUnits: 6,
      redUnits: 3,
      maxTicks: 600,
    }),
    environmentSeed: 77,
    basePlanSeed: 120,
    sampleCount: 6,
    maxDecisions: 8,
    redDifficulty: 'easy' as const,
  };

  it('restarts every plan from one state and records deterministic plan-caused actions', () => {
    const first = buildPlanRolloutDataset(options);
    const second = buildPlanRolloutDataset(options);

    expect(first).toEqual(second);
    expect(auditPlanRolloutDataset(first)).toBe(first);
    expect(auditPlanRolloutDataset(JSON.parse(JSON.stringify(first)))).toEqual(first);
    expect(new Set(first.episodes.map(({ initialStateHash }) => initialStateHash))).toEqual(
      new Set([first.sourceStateHash]),
    );
    expect(first.episodes.every(({ transitions }) => transitions.length > 0)).toBe(true);
    expect(
      first.episodes.every(({ transitions }) =>
        transitions.every(({ actionResults }) => actionResults.every(({ accepted }) => accepted)),
      ),
    ).toBe(true);

    const firstActions = new Set(
      first.episodes.map(({ transitions }) => JSON.stringify(transitions[0].action)),
    );
    expect(firstActions.size).toBeGreaterThan(1);
    const firstPlanTensors = new Set(
      first.episodes.map(({ transitions }) => JSON.stringify(transitions[0].planGroups)),
    );
    expect(firstPlanTensors.size).toBe(6);
  });

  it('rejects state, tensor, action-result, and digest corruption', () => {
    const dataset = buildPlanRolloutDataset({ ...options, sampleCount: 1 });

    const stateDrift = structuredClone(dataset) as unknown as {
      episodes: { transitions: { preStateHash: string }[] }[];
    };
    stateDrift.episodes[0].transitions[0].preStateHash = 'fnv1a64:0000000000000000';
    expect(() => auditPlanRolloutDataset(stateDrift)).toThrow('state hash');

    const tensorDrift = structuredClone(dataset) as unknown as {
      episodes: { transitions: { planGroups: number[] }[] }[];
    };
    tensorDrift.episodes[0].transitions[0].planGroups[0] = 2;
    expect(() => auditPlanRolloutDataset(tensorDrift)).toThrow('plan tensor');

    const rejection = structuredClone(dataset) as unknown as {
      episodes: { transitions: { actionResults: { accepted: boolean }[] }[] }[];
    };
    rejection.episodes[0].transitions[0].actionResults[0].accepted = false;
    expect(() => auditPlanRolloutDataset(rejection)).toThrow('action result');

    const digestDrift = { ...structuredClone(dataset), scenario: 'changed' };
    expect(() => auditPlanRolloutDataset(digestDrift)).toThrow('digest mismatch');
  });
});
