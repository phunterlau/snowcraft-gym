import { describe, expect, it } from 'vitest';
import { BatchHost, BATCH_REQUEST_FORMAT } from '../batch/BatchHost';
import { commandedTenVsTenPlan } from '../orchestration/examples/CommandedReplayExample';
import { SnowGymService } from '../server/SnowGymService';

function request(operation: string, items: unknown[], requestId = 'test-1'): unknown {
  return { format: BATCH_REQUEST_FORMAT, requestId, operation, items };
}

function resetBody(seed: number): object {
  return {
    seed,
    scenario: {
      blueUnits: 1,
      redUnits: 1,
      arenaWidth: 40,
      arenaHeight: 30,
      maxTicks: 1200,
      decisionHz: 10,
      redDifficulty: 'normal',
      redController: 'random',
    },
  };
}

function oneGroupPlan(): object {
  const plan = commandedTenVsTenPlan();
  return { ...plan, groups: [plan.groups[0]] };
}

describe('BatchHost', () => {
  it('publishes a versioned handshake before mutations', () => {
    const response = new BatchHost().handle(request('handshake', []));
    expect(response.results[0]?.status).toBe(200);
    expect(response.results[0]?.body).toMatchObject({
      protocolVersion: 'snowgym.batch.v0',
      isolation: 'per-world-explicit-result',
    });
  });

  it('matches the SnowGymService reset and step payload exactly', () => {
    const host = new BatchHost();
    const direct = new SnowGymService();
    const reset = resetBody(42);
    const batchReset = host.handle(request('reset', [{ worldId: 'world-0', body: reset }]));
    const directReset = direct.handle('POST', '/reset', reset);
    expect(batchReset.results[0]).toEqual({ worldId: 'world-0', ...directReset });

    const snapshot = directReset.body as {
      status: { stateHash: string };
      observation: { allies: { id: number }[] };
    };
    const body = {
      expectedStateHash: snapshot.status.stateHash,
      idempotencyKey: 'step-1',
      action: { actions: [{ type: 'hold', unitId: snapshot.observation.allies[0]!.id }] },
    };
    const batchStep = host.handle(request('step', [{ worldId: 'world-0', body }], 'test-2'));
    const directStep = direct.handle('POST', '/step', body);
    expect(batchStep.results[0]).toEqual({ worldId: 'world-0', ...directStep });
  });

  it('matches the built-in scripted-policy endpoint exactly', () => {
    const host = new BatchHost();
    const direct = new SnowGymService();
    const reset = resetBody(43);
    const batchReset = host.handle(request('reset', [{ worldId: 'world-0', body: reset }]));
    const directReset = direct.handle('POST', '/reset', reset);
    expect(batchReset.results[0]).toEqual({ worldId: 'world-0', ...directReset });
    const snapshot = directReset.body as { status: { stateHash: string } };
    const body = {
      expectedStateHash: snapshot.status.stateHash,
      idempotencyKey: 'scripted-1',
    };
    const batchStep = host.handle(
      request('stepScripted', [{ worldId: 'world-0', body }], 'scripted-step'),
    );
    const directStep = direct.handle('POST', '/step-scripted', body);
    expect(batchStep.results[0]).toEqual({ worldId: 'world-0', ...directStep });
  });

  it('activates and reads plans independently across persistent worlds', () => {
    const host = new BatchHost();
    const reset = host.handle(
      request('reset', [
        { worldId: 'planned', body: resetBody(51) },
        { worldId: 'unplanned', body: resetBody(52) },
      ]),
    );
    const planned = reset.results[0]!.body as { status: { stateHash: string } };
    const activated = host.handle(
      request('activatePlan', [
        {
          worldId: 'planned',
          body: {
            planId: 'batch-plan',
            plan: oneGroupPlan(),
            expectedStateHash: planned.status.stateHash,
            idempotencyKey: 'batch-plan-activation',
          },
        },
      ]),
    );
    expect(activated.results[0]).toMatchObject({
      worldId: 'planned',
      status: 200,
      body: { planId: 'batch-plan', planGroupMask: [1, 0, 0] },
    });

    const observations = host.handle(
      request('planObservation', [{ worldId: 'planned' }, { worldId: 'unplanned' }]),
    );
    expect(observations.results[0]).toMatchObject({
      worldId: 'planned',
      status: 200,
      body: { planId: 'batch-plan', tick: 0 },
    });
    expect(observations.results[1]).toEqual({
      worldId: 'unplanned',
      status: 409,
      body: { error: 'plan_not_active' },
    });
  });

  it('keeps world state independent and reports failures explicitly', () => {
    const host = new BatchHost();
    host.handle(
      request('reset', [
        { worldId: 'left', body: resetBody(11) },
        { worldId: 'right', body: resetBody(12) },
      ]),
    );
    const before = host.handle(
      request('status', [{ worldId: 'left' }, { worldId: 'right' }], 'status-1'),
    );
    const left = before.results[0]!.body as {
      status: { stateHash: string };
      observation: { allies: { id: number }[] };
    };
    const result = host.handle(
      request(
        'step',
        [
          {
            worldId: 'left',
            body: {
              expectedStateHash: left.status.stateHash,
              action: { actions: [{ type: 'hold', unitId: left.observation.allies[0]!.id }] },
            },
          },
          { worldId: 'missing', body: { action: { actions: [] } } },
        ],
        'step-mixed',
      ),
    );
    expect(result.results.map((item) => item.status)).toEqual([200, 404]);
    const after = host.handle(
      request('status', [{ worldId: 'left' }, { worldId: 'right' }], 'status-2'),
    );
    expect((after.results[0]!.body as { status: { tick: number } }).status.tick).toBe(6);
    expect((after.results[1]!.body as { status: { tick: number } }).status.tick).toBe(0);
  });

  it('rejects duplicate slots before any world can advance', () => {
    const host = new BatchHost();
    expect(() =>
      host.handle(
        request('reset', [
          { worldId: 'same', body: resetBody(1) },
          { worldId: 'same', body: resetBody(2) },
        ]),
      ),
    ).toThrow(/duplicate/);
    expect(host.size).toBe(0);
  });

  it('does not retain a default world after a failed first reset', () => {
    const host = new BatchHost();
    const response = host.handle(
      request('reset', [{ worldId: 'invalid', body: { seed: 1, scenario: { blueUnits: 0 } } }]),
    );
    expect(response.results[0]?.status).toBe(400);
    expect(host.size).toBe(0);
  });

  it.each([8, 32])('preserves exact service parity across %i worlds', (worldCount) => {
    const host = new BatchHost();
    const services = Array.from({ length: worldCount }, () => new SnowGymService());
    const resets = Array.from({ length: worldCount }, (_, index) => ({
      worldId: `world-${index}`,
      body: resetBody(100 + index),
    }));
    const batchReset = host.handle(request('reset', resets, 'eight-reset'));
    const directReset = services.map((service, index) =>
      service.handle('POST', '/reset', resetBody(100 + index)),
    );
    expect(batchReset.results).toEqual(
      directReset.map((response, index) => ({ worldId: `world-${index}`, ...response })),
    );

    const steps = directReset.map((response, index) => {
      const snapshot = response.body as {
        status: { stateHash: string };
        observation: { allies: { id: number }[] };
      };
      return {
        worldId: `world-${index}`,
        body: {
          expectedStateHash: snapshot.status.stateHash,
          action: {
            actions: [{ type: 'hold', unitId: snapshot.observation.allies[0]!.id }],
          },
        },
      };
    });
    const batchStep = host.handle(request('step', steps, 'eight-step'));
    const directStep = services.map((service, index) =>
      service.handle('POST', '/step', steps[index]!.body),
    );
    expect(batchStep.results).toEqual(
      directStep.map((response, index) => ({ worldId: `world-${index}`, ...response })),
    );
  });
});
