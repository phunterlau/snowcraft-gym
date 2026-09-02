import { describe, expect, it } from 'vitest';
import { createMapScenario } from '../scenarios/Scenario';
import {
  auditPlanTensorDataset,
  buildPlanTensorDataset,
} from '../training/plan/PlanTensorDataset';

describe('PlanTensorDataset', () => {
  it('builds and audits an exactly reproducible headless dataset', () => {
    const options = {
      scenario: createMapScenario('arena6.json', {
        seed: 42,
        blueUnits: 10,
        redUnits: 10,
      }),
      environmentSeed: 42,
      basePlanSeed: 1000,
      sampleCount: 12,
    };
    const first = buildPlanTensorDataset(options);
    const second = buildPlanTensorDataset(options);

    expect(first).toEqual(second);
    expect(auditPlanTensorDataset(first)).toBe(first);
    expect(first.tensors).toHaveLength(12);
    expect(first.tensors[0].groups).toHaveLength(3 * 38);
  });

  it('rejects tensor corruption and digest drift', () => {
    const dataset = buildPlanTensorDataset({
      scenario: createMapScenario('arena6.json', {
        seed: 9,
        blueUnits: 5,
        redUnits: 5,
      }),
      environmentSeed: 9,
      basePlanSeed: 2000,
      sampleCount: 6,
    });
    const corrupt = structuredClone(dataset) as unknown as {
      tensors: { groups: number[] }[];
    };
    corrupt.tensors[0].groups[0] = 2;
    expect(() => auditPlanTensorDataset(corrupt)).toThrow('tensor value');

    const drift = { ...structuredClone(dataset), scenario: 'changed' };
    expect(() => auditPlanTensorDataset(drift)).toThrow('digest mismatch');
  });
});
