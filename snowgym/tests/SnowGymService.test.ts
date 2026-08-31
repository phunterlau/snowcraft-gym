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

  it('resets by seed and advances with the default blue policy', () => {
    const service = new SnowGymService();
    const reset = service.handle('POST', '/reset', { seed: 42 });
    const step = service.handle('POST', '/step', {});
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

    const step = service.handle('POST', '/step', {});
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

  it('rejects an unknown map and a map with a conflicting roster', () => {
    const service = new SnowGymService();
    expect(service.handle('POST', '/reset', { scenario: { map: 'arena99.json' } })).toMatchObject({
      status: 400,
      body: { error: 'invalid_request' },
    });
    expect(
      service.handle('POST', '/reset', { scenario: { map: 'arena1.json', blueUnits: 5 } }),
    ).toMatchObject({ status: 400, body: { error: 'invalid_request' } });
  });
});
