import { describe, expect, it } from 'vitest';
import { COMMAND_PLAN_VERSION, type CommandPlan } from '../orchestration/command/CommandPlan';
import type { CommanderRequest } from '../orchestration/commander/CommanderClient';
import { STRATEGIC_SUMMARY_VERSION } from '../orchestration/commander/StrategicSummary';
import {
  OPENAI_COMMANDER_MODEL,
  OpenAICommanderClient,
  OpenAICommanderError,
  openAIRequestBody,
} from '../orchestration/providers/OpenAICommanderClient';

describe('OpenAICommanderClient', () => {
  it('sends a stateless Luna reasoning request with strict structured output', async () => {
    const calls: Array<{ input: RequestInfo | URL; init?: RequestInit }> = [];
    const fetchMock: typeof fetch = async (input, init) => {
      calls.push({ input, init });
      return jsonResponse(successPayload(), 200, { 'x-request-id': 'req-provider-1' });
    };
    const clock = [100, 125];
    const client = new OpenAICommanderClient({
      apiKey: 'test-key',
      reasoningEffort: 'high',
      fetch: fetchMock,
      now: () => clock.shift() ?? 125,
    });

    const response = await client.plan(commanderRequest());

    expect(calls).toHaveLength(1);
    expect(String(calls[0].input)).toBe('https://api.openai.com/v1/responses');
    expect(new Headers(calls[0].init?.headers).get('authorization')).toBe('Bearer test-key');
    const body = JSON.parse(String(calls[0].init?.body));
    expect(body).toMatchObject({
      model: OPENAI_COMMANDER_MODEL,
      store: false,
      reasoning: { effort: 'high' },
      max_output_tokens: 4096,
      text: {
        format: {
          type: 'json_schema',
          name: 'snowgym_command_plan',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            required: ['schemaVersion', 'intentSummary', 'groups'],
          },
        },
      },
    });
    expect(body.text.format.schema).not.toHaveProperty('$schema');
    expect(body.text.format.schema).not.toHaveProperty('$id');
    expect(body.text.format.schema).not.toHaveProperty('title');
    expect(body.text.format.schema.properties.schemaVersion).toMatchObject({
      type: 'string',
      const: COMMAND_PLAN_VERSION,
    });
    expect(body.text.format.schema.$defs.enemyClusterObjective.properties.kind).toMatchObject({
      type: 'string',
      const: 'enemy_cluster',
    });
    expect(body.text.format.schema.$defs.group.properties.role).toMatchObject({
      type: 'string',
      enum: ['main', 'maneuver', 'reserve'],
    });
    expect(JSON.parse(body.input[0].content[0].text)).toMatchObject({
      requestId: 'commander-request-1',
      triggers: ['plan_expired'],
      strategicSummary: { schemaVersion: STRATEGIC_SUMMARY_VERSION, sourceTick: 60 },
    });
    expect(body.input[0].content[0].text).not.toContain('"unitIds"');
    expect(body.input[0].content[0].text).not.toContain('"enemyIds"');
    expect(response).toEqual({
      decision: oneGroupPlan(),
      metadata: {
        model: OPENAI_COMMANDER_MODEL,
        latencyMs: 25,
        tokensIn: 120,
        tokensOut: 80,
        reasoningTokens: 50,
        cachedInputTokens: 20,
        responseId: 'resp-test-1',
        providerRequestId: 'req-provider-1',
      },
    });
  });

  it('builds the exact requested model and configurable reasoning body without credentials', () => {
    const body = openAIRequestBody(commanderRequest(), 'medium', 2_048);
    expect(body).toMatchObject({
      model: 'gpt-5.6-luna',
      reasoning: { effort: 'medium' },
      max_output_tokens: 2048,
      store: false,
    });
    expect(JSON.stringify(body)).not.toContain('test-key');
  });

  it('requires an environment-only credential and rejects unsafe endpoints', () => {
    expect(() => new OpenAICommanderClient({ apiKey: '' })).toThrow('OPENAI_API_KEY is required');
    expect(
      () =>
        new OpenAICommanderClient({
          apiKey: 'test-key',
          endpoint: 'http://api.example.test/v1/responses',
        }),
    ).toThrow('endpoint must use HTTPS');
    expect(
      () =>
        new OpenAICommanderClient({
          apiKey: 'test-key',
          endpoint: 'ftp://localhost/responses',
        }),
    ).toThrow('endpoint must use HTTPS');
  });

  it('reports HTTP errors with provider request IDs but never includes the API key', async () => {
    const client = new OpenAICommanderClient({
      apiKey: 'secret-not-for-errors',
      fetch: async () =>
        jsonResponse({ error: { message: 'model access denied' } }, 403, {
          'x-request-id': 'req-denied',
        }),
    });

    const error = await capturedError(client.plan(commanderRequest()));
    expect(error).toBeInstanceOf(OpenAICommanderError);
    expect(error.message).toContain('HTTP 403 (req-denied): model access denied');
    expect(error.message).not.toContain('secret-not-for-errors');
  });

  it('rejects refusals, incomplete responses, invalid JSON output, and missing output', async () => {
    await expectPlanError(
      responsePayload({ type: 'refusal', refusal: 'cannot comply' }),
      'refused the plan request',
    );
    await expectPlanError(
      {
        ...successPayload(),
        status: 'incomplete',
        incomplete_details: { reason: 'max_output_tokens' },
      },
      'status incomplete: max_output_tokens',
    );
    await expectPlanError(responsePayload({ type: 'output_text', text: '{bad' }), 'not valid JSON');
    await expectPlanError({ ...successPayload(), output: [] }, 'did not contain output_text');
  });

  it('wraps transport failures while preserving abort errors for scheduler handling', async () => {
    const networkClient = new OpenAICommanderClient({
      apiKey: 'test-key',
      fetch: async () => {
        throw new Error('socket closed');
      },
    });
    await expect(networkClient.plan(commanderRequest())).rejects.toThrow(
      'OpenAI request failed: socket closed',
    );

    const controller = new AbortController();
    controller.abort();
    const abort = new Error('aborted');
    abort.name = 'AbortError';
    const abortClient = new OpenAICommanderClient({
      apiKey: 'test-key',
      fetch: async () => {
        throw abort;
      },
    });
    await expect(abortClient.plan(commanderRequest(), controller.signal)).rejects.toBe(abort);
  });
});

function commanderRequest(): CommanderRequest {
  return {
    requestId: 'commander-request-1',
    triggers: ['plan_expired'],
    summary: {
      schemaVersion: STRATEGIC_SUMMARY_VERSION,
      sourceTick: 60,
      sourceStateHash: 'fnv1a64:0000000000000001',
      arena: { width: 40, height: 30, obstacleCount: 2 },
      ownForce: { alive: 10, healthFraction: 0.9, centroid: { x: -5, y: 0 }, spread: 3 },
      enemyForce: { alive: 8, healthFraction: 0.7, centroid: { x: 5, y: 0 }, spread: 4 },
      hostileProjectileCount: 2,
      groups: [
        {
          role: 'main',
          mission: 'engage',
          assigned: 10,
          living: 10,
          objectiveKind: 'enemy_cluster',
        },
      ],
    },
    currentPlan: oneGroupPlan(),
  };
}

function oneGroupPlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: 'Pressure the nearest enemy cluster.',
    groups: [
      {
        role: 'main',
        allocationWeight: 1,
        selection: 'balanced',
        order: {
          mission: 'engage',
          objective: { kind: 'enemy_cluster', select: 'nearest' },
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
  };
}

function successPayload(): Record<string, unknown> {
  return responsePayload({ type: 'output_text', text: JSON.stringify(oneGroupPlan()) });
}

function responsePayload(content: unknown): Record<string, unknown> {
  return {
    id: 'resp-test-1',
    model: OPENAI_COMMANDER_MODEL,
    status: 'completed',
    incomplete_details: null,
    output: [{ type: 'message', content: [content] }],
    usage: {
      input_tokens: 120,
      output_tokens: 80,
      input_tokens_details: { cached_tokens: 20 },
      output_tokens_details: { reasoning_tokens: 50 },
    },
  };
}

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

async function expectPlanError(payload: unknown, message: string): Promise<void> {
  const client = new OpenAICommanderClient({
    apiKey: 'test-key',
    fetch: async () => jsonResponse(payload),
  });
  await expect(client.plan(commanderRequest())).rejects.toThrow(message);
}

async function capturedError(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof Error) return error;
    throw new Error(`expected Error, received ${String(error)}`, { cause: error });
  }
  throw new Error('expected promise to reject');
}
