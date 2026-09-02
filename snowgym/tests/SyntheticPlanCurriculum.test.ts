import { describe, expect, it } from 'vitest';
import type { UnitObservation } from '../observations/Observation';
import { validateCommandPlan } from '../orchestration/command/PlanValidator';
import {
  SYNTHETIC_PLAN_CURRICULUM_FORMAT,
  generateSyntheticPlanCurriculum,
} from '../training/plan/SyntheticPlanCurriculum';
import { observationWith } from './orchestrationTestHelpers';

describe('SyntheticPlanCurriculum', () => {
  it('is deterministic, schema-valid, grounded, and directive-diverse', () => {
    const observation = observationWith({
      allies: units(10, 'blue', -10),
      enemies: units(10, 'red', 10, 100),
    });
    const options = {
      baseSeed: 120,
      sampleCount: 12,
      sourceStateHash: 'fnv1a64:synthetic',
    };
    const first = generateSyntheticPlanCurriculum(observation, options);
    const second = generateSyntheticPlanCurriculum(observation, options);

    expect(first).toEqual(second);
    expect(first.format).toBe(SYNTHETIC_PLAN_CURRICULUM_FORMAT);
    expect(first.samples.map(({ sourceSeed }) => sourceSeed)).toEqual(
      Array.from({ length: 12 }, (_, index) => 120 + index),
    );
    expect(first.source.stateHash).toBe(options.sourceStateHash);
    const missions = new Set<string>();
    const approaches = new Set<string>();
    const firePolicies = new Set<string>();
    for (const sample of first.samples) {
      expect(validateCommandPlan(sample.plan)).toEqual({ ok: true, value: sample.plan });
      expect(sample.assignments.flatMap(({ unitIds }) => unitIds).sort((a, b) => a - b)).toEqual(
        Array.from({ length: 10 }, (_, index) => index + 1),
      );
      for (const group of sample.plan.groups) {
        missions.add(group.order.mission);
        approaches.add(group.order.approach);
        firePolicies.add(group.order.engagement.fire);
      }
    }
    expect(missions).toEqual(new Set(['engage', 'advance', 'hold', 'withdraw', 'support']));
    expect(approaches).toEqual(new Set(['direct', 'left_flank', 'right_flank', 'avoid_center']));
    expect(firePolicies).toEqual(new Set(['focus', 'distributed', 'opportunistic']));
  });

  it('rejects invalid seed ranges and rosters that cannot ground all variants', () => {
    const small = observationWith({
      allies: units(2, 'blue', -10),
      enemies: units(1, 'red', 10, 100),
    });
    expect(() =>
      generateSyntheticPlanCurriculum(small, { baseSeed: 1, sampleCount: 6 }),
    ).toThrow('at least three living allies');
    const valid = observationWith({
      allies: units(3, 'blue', -10),
      enemies: units(1, 'red', 10, 100),
    });
    expect(() =>
      generateSyntheticPlanCurriculum(valid, { baseSeed: 1, sampleCount: 0 }),
    ).toThrow('sampleCount');
    expect(() =>
      generateSyntheticPlanCurriculum(valid, {
        baseSeed: Number.MAX_SAFE_INTEGER,
        sampleCount: 2,
      }),
    ).toThrow('source seed range');
  });
});

function units(
  count: number,
  team: UnitObservation['team'],
  x: number,
  idOffset = 0,
): UnitObservation[] {
  return Array.from({ length: count }, (_, index) => ({
    id: idOffset + index + 1,
    team,
    x,
    y: index * 1.2 - ((count - 1) * 1.2) / 2,
    vx: 0,
    vy: 0,
    health: 100,
    maxHealth: 100,
    alive: true,
    state: 'idle',
    throwCooldown: 0,
    charge: 0,
  }));
}
