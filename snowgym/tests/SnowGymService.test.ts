import { describe, expect, it } from 'vitest';
import type { EnvironmentStatus, StepResult } from '../core/SnowEnvironment';
import { SnowGymService } from '../server/SnowGymService';

describe('SnowGymService', () => {
  it('returns server-side status and a detached observation', () => {
    const service = new SnowGymService();
    const response = service.handle('GET', '/status');
    const body = response.body as {
      status: EnvironmentStatus;
      observation: { allies: unknown[]; enemies: unknown[] };
    };

    expect(response.status).toBe(200);
    expect(body.status).toMatchObject({ tick: 0, blueAlive: 3, redAlive: 3 });
    expect(body.observation.allies).toHaveLength(3);
    expect(body.observation.enemies).toHaveLength(3);
  });

  it('resets by seed and advances through the explicit scripted-policy endpoint', () => {
    const service = new SnowGymService();
    const reset = service.handle('POST', '/reset', { seed: 42 });
    const step = service.handle('POST', '/step-scripted', {});
    const resetBody = reset.body as { status: EnvironmentStatus };
    const stepBody = step.body as StepResult & { info: { action: { actions: unknown[] } } };

    expect(resetBody.status.seed).toBe(42);
    expect(step.status).toBe(200);
    expect(stepBody.observation.tick).toBe(6);
    expect(stepBody.info.actionResults).toHaveLength(3);
    expect(stepBody.info.action.actions).toHaveLength(3);
  });

  it('runs autoplay to a server-readable terminal result', () => {
    const service = new SnowGymService();
    const response = service.handle('POST', '/autoplay', { maxDecisions: 2_000 });
    const body = response.body as { decisions: number; status: EnvironmentStatus };

    expect(response.status).toBe(200);
    expect(body.decisions).toBeGreaterThan(0);
    expect(body.status.terminated || body.status.truncated).toBe(true);
    expect(body.status.winner).not.toBeNull();
  });

  it('validates actions received over the transport boundary', () => {
    const service = new SnowGymService();
    const response = service.handle('POST', '/step', {
      action: { actions: [{ type: 'move', unitId: 1, x: 'bad', y: 0 }] },
    });

    expect(response).toMatchObject({
      status: 400,
      body: { error: 'invalid_request', message: 'x must be a finite number' },
    });
  });

  it('accepts hold as an explicit movement-cancellation action', () => {
    const service = new SnowGymService();
    const move = service.handle('POST', '/step', {
      action: { actions: [{ type: 'move', unitId: 1, x: 8, y: 0 }] },
    });
    const moveBody = move.body as StepResult;

    expect(move.status).toBe(200);
    expect(moveBody.observation.allies[0].state).toBe('moving');

    const hold = service.handle('POST', '/step', {
      action: { actions: [{ type: 'hold', unitId: 1 }] },
    });
    const holdBody = hold.body as StepResult;

    expect(hold.status).toBe(200);
    expect(holdBody.info.actionResults[0]).toMatchObject({ accepted: true });
    expect(holdBody.observation.allies[0].state).toBe('idle');
  });

  it('resets into a configurable N-blue versus M-red fight', () => {
    const service = new SnowGymService();
    const response = service.handle('POST', '/reset', {
      seed: 9,
      scenario: {
        blueUnits: 5,
        redUnits: 2,
        arenaWidth: 50,
        arenaHeight: 24,
        decisionHz: 20,
        maxTicks: 600,
        redDifficulty: 'hard',
      },
    });
    const body = response.body as {
      status: EnvironmentStatus;
      observation: { allies: unknown[]; enemies: unknown[] };
    };

    expect(response.status).toBe(200);
    expect(body.observation.allies).toHaveLength(5);
    expect(body.observation.enemies).toHaveLength(2);
    expect(body.status).toMatchObject({
      seed: 9,
      decisionHz: 20,
      ticksPerDecision: 3,
      configuration: {
        blueUnits: 5,
        redUnits: 2,
        arenaWidth: 50,
        arenaHeight: 24,
        maxTicks: 600,
        redDifficulty: 'hard',
      },
    });
  });

  it('rejects invalid or misspelled scenario configuration', () => {
    const service = new SnowGymService();
    expect(
      service.handle('POST', '/reset', {
        scenario: { blueUnits: 11 },
      }),
    ).toMatchObject({ status: 400, body: { error: 'invalid_request' } });
    expect(
      service.handle('POST', '/reset', {
        scenario: { blueUnit: 2 },
      }),
    ).toMatchObject({
      status: 400,
      body: { error: 'invalid_request', message: 'unknown scenario fields: blueUnit' },
    });
  });

  it('accepts a selectable red controller and reports it in status', () => {
    const service = new SnowGymService();
    const response = service.handle('POST', '/reset', {
      seed: 11,
      scenario: { redController: 'random' },
    });
    const body = response.body as { status: EnvironmentStatus };

    expect(response.status).toBe(200);
    expect(body.status.configuration.redController).toBe('random');

    const step = service.handle('POST', '/step-scripted', {});
    expect(step.status).toBe(200);
  });

  it('rejects an unknown red controller', () => {
    const service = new SnowGymService();
    expect(
      service.handle('POST', '/reset', {
        scenario: { redController: 'skynet' },
      }),
    ).toMatchObject({
      status: 400,
      body: { error: 'invalid_request', message: 'redController must be one of: scripted, random' },
    });
  });

  it('runs a full autoplay episode against the random red controller', () => {
    const service = new SnowGymService();
    service.handle('POST', '/reset', { seed: 21, scenario: { redController: 'random' } });
    const response = service.handle('POST', '/autoplay', { maxDecisions: 2_000 });
    const body = response.body as { decisions: number; status: EnvironmentStatus };

    expect(response.status).toBe(200);
    expect(body.decisions).toBeGreaterThan(0);
    expect(body.status.terminated || body.status.truncated).toBe(true);
    expect(body.status.configuration.redController).toBe('random');
  });

  it('resets onto a bundled map with terrain and reports it in status', () => {
    const service = new SnowGymService();
    const response = service.handle('POST', '/reset', {
      seed: 5,
      scenario: { map: 'arena4.json' },
    });
    const body = response.body as {
      status: EnvironmentStatus;
      observation: { obstacles: unknown[]; allies: unknown[]; enemies: unknown[] };
    };

    expect(response.status).toBe(200);
    expect(body.status.configuration.map).toBe('arena4.json');
    expect(body.observation.obstacles.length).toBeGreaterThan(0);
    expect(body.observation.allies).toHaveLength(3);
    expect(body.observation.enemies).toHaveLength(3);
  });

  it('loads the ten-spawn Winter Front map as a 10v10 server scenario', () => {
    const service = new SnowGymService();
    const response = service.handle('POST', '/reset', {
      seed: 42,
      scenario: { map: 'arena6.json', maxTicks: 900 },
    });
    const body = response.body as {
      status: EnvironmentStatus;
      observation: { obstacles: unknown[]; allies: unknown[]; enemies: unknown[] };
    };

    expect(response.status).toBe(200);
    expect(body.status.configuration).toMatchObject({
      map: 'arena6.json',
      blueUnits: 10,
      redUnits: 10,
      maxTicks: 900,
    });
    expect(body.observation.allies).toHaveLength(10);
    expect(body.observation.enemies).toHaveLength(10);
    expect(body.observation.obstacles).toHaveLength(27);
  });

  it('selects an asymmetric roster from a map native spawn pool', () => {
    const service = new SnowGymService();
    const response = service.handle('POST', '/reset', {
      seed: 17,
      scenario: { map: 'arena6.json', blueUnits: 5, redUnits: 2 },
    });
    const body = response.body as { status: EnvironmentStatus };

    expect(response.status).toBe(200);
    expect(body.status.configuration).toMatchObject({
      map: 'arena6.json',
      blueUnits: 5,
      redUnits: 2,
    });
  });

  it('rejects an unknown map and a map with a conflicting roster', () => {
    const service = new SnowGymService();
    expect(service.handle('POST', '/reset', { scenario: { map: 'arena99.json' } })).toMatchObject({
      status: 400,
      body: { error: 'invalid_request' },
    });
    expect(
      service.handle('POST', '/reset', { scenario: { map: 'arena1.json', arenaWidth: 50 } }),
    ).toMatchObject({ status: 400, body: { error: 'invalid_request' } });
  });

  it('publishes machine-readable capabilities for maps, actions, and Gym ids', () => {
    const service = new SnowGymService();
    const response = service.handle('GET', '/capabilities');
    const body = response.body as {
      format: string;
      endpoints: { step: { requires: string[] }; stepScripted: { path: string } };
      actions: {
        types: { hold: { required: string[] } };
        semantics: { hold: string; noop: string };
      };
      scenarios: { maxTeamSize: number; maps: Array<{ id: string; blueCapacity: number }> };
      gymnasium: { environments: Array<{ id: string }> };
    };

    expect(response.status).toBe(200);
    expect(body.format).toBe('snowgym.capabilities.v0');
    expect(body.endpoints.step.requires).toEqual(['action']);
    expect(body.endpoints.stepScripted.path).toBe('/step-scripted');
    expect(body.actions.types.hold.required).toEqual(['type', 'unitId']);
    expect(body.actions.semantics).toMatchObject({
      hold: 'cancels-current-movement-order',
      noop: 'does-not-cancel-current-movement-order',
    });
    expect(body.scenarios.maxTeamSize).toBe(10);
    expect(body.scenarios.maps).toContainEqual(
      expect.objectContaining({ id: 'arena6.json', blueCapacity: 10 }),
    );
    expect(body.gymnasium.environments.map(({ id }) => id)).toEqual([
      'SnowGym/Squad-v0',
      'SnowGym/Squad-v1',
      'SnowGym/Squad-v2',
    ]);
  });

  it('rejects misspelled top-level fields without changing state', () => {
    const service = new SnowGymService();
    const before = service.handle('GET', '/status').body as { status: EnvironmentStatus };

    expect(service.handle('POST', '/reset', { sead: 42 })).toMatchObject({
      status: 400,
      body: { error: 'invalid_request', message: 'unknown request fields: sead' },
    });
    expect(service.handle('POST', '/step', { aciton: { actions: [] } })).toMatchObject({
      status: 400,
      body: { error: 'invalid_request' },
    });
    expect(service.handle('POST', '/autoplay', { maxDecision: 1 })).toMatchObject({
      status: 400,
      body: { error: 'invalid_request' },
    });

    const after = service.handle('GET', '/status').body as { status: EnvironmentStatus };
    expect(after.status.stateHash).toBe(before.status.stateHash);
    expect(after.status.tick).toBe(0);
  });

  it('requires an explicit external action and strictly validates its fields', () => {
    const service = new SnowGymService();

    expect(service.handle('POST', '/step', {})).toMatchObject({
      status: 400,
      body: {
        error: 'invalid_request',
        message: 'action is required; use /step-scripted for the built-in policy',
      },
    });
    expect(
      service.handle('POST', '/step', {
        action: { actions: [{ type: 'noop', unitId: 1, surprise: true }] },
      }),
    ).toMatchObject({
      status: 400,
      body: { error: 'invalid_request', message: 'unknown unit action fields: surprise' },
    });
  });

  it('rejects a stale expected state hash without advancing', () => {
    const service = new SnowGymService();
    const response = service.handle('POST', '/step-scripted', {
      expectedStateHash: 'fnv1a64:0000000000000000',
      idempotencyKey: 'stale-step-1',
    });

    expect(response).toMatchObject({
      status: 409,
      body: { error: 'stale_state', expectedStateHash: 'fnv1a64:0000000000000000' },
    });
    const status = service.handle('GET', '/status').body as { status: EnvironmentStatus };
    expect(status.status.tick).toBe(0);
  });

  it('deduplicates repeated mutations and rejects reuse with different input', () => {
    const service = new SnowGymService();
    const snapshot = service.handle('GET', '/status').body as { status: EnvironmentStatus };
    const request = {
      expectedStateHash: snapshot.status.stateHash,
      idempotencyKey: 'scripted-step-1',
    };

    const first = service.handle('POST', '/step-scripted', request);
    const duplicate = service.handle('POST', '/step-scripted', request);
    expect(duplicate).toEqual(first);
    const after = service.handle('GET', '/status').body as { status: EnvironmentStatus };
    expect(after.status.tick).toBe(6);

    expect(
      service.handle('POST', '/autoplay', {
        maxDecisions: 1,
        idempotencyKey: 'scripted-step-1',
      }),
    ).toMatchObject({ status: 409, body: { error: 'idempotency_conflict' } });
  });
});
