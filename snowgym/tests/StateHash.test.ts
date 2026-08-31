import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/state-hash-v1.json';
import type { Observation } from '../observations/Observation';
import { hashObservation } from '../reproducibility/StateHash';

describe('public observation state hash', () => {
  it('matches the shared TypeScript/Python golden fixture', () => {
    expect(hashObservation(fixture.observation as Observation)).toBe(fixture.expected);
  });

  it('is independent of entity array order', () => {
    const observation = structuredClone(fixture.observation) as Observation;
    observation.allies.reverse();
    observation.enemies.reverse();
    observation.projectiles.reverse();
    observation.obstacles.reverse();

    expect(hashObservation(observation)).toBe(fixture.expected);
  });

  it('changes when public simulation state changes', () => {
    const observation = structuredClone(fixture.observation) as Observation;
    observation.allies[0].health -= 1;

    expect(hashObservation(observation)).not.toBe(fixture.expected);
  });
});
