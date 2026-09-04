import { describe, expect, it } from 'vitest';
import fixture from '../fixtures/state-hash-v1.json';
import fixtureV2 from '../fixtures/state-hash-v2.json';
import type { Observation } from '../observations/Observation';
import { LEGACY_STATE_HASH_VERSION } from '../protocol/Version';
import { hashObservation } from '../reproducibility/StateHash';

describe('public observation state hash', () => {
  it('matches the shared TypeScript/Python golden fixture', () => {
    expect(hashObservation(fixture.observation as Observation, LEGACY_STATE_HASH_VERSION)).toBe(
      fixture.expected,
    );
  });

  it('is independent of entity array order', () => {
    const observation = structuredClone(fixture.observation) as Observation;
    observation.allies.reverse();
    observation.enemies.reverse();
    observation.projectiles.reverse();
    observation.obstacles.reverse();

    expect(hashObservation(observation, LEGACY_STATE_HASH_VERSION)).toBe(fixture.expected);
  });

  it('changes when public simulation state changes', () => {
    const observation = structuredClone(fixture.observation) as Observation;
    observation.allies[0].health -= 1;

    expect(hashObservation(observation, LEGACY_STATE_HASH_VERSION)).not.toBe(fixture.expected);
  });

  it('includes persistent controller state in the current hash', () => {
    const observation = structuredClone(fixture.observation) as Observation;
    const first = hashObservation(observation);
    observation.allies[0].moveTarget = { x: 4, y: -2 };
    observation.allies[0].steeringTarget = { x: 1, y: -1 };

    expect(hashObservation(observation)).not.toBe(first);
  });

  it('matches the shared actuator-complete v2 fixture', () => {
    expect(hashObservation(fixtureV2.observation as Observation)).toBe(fixtureV2.expected);
  });
});
