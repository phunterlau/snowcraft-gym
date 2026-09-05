import commandPlanSchema from '../command/command-plan.schema.json';
import type {
  CommanderClient,
  CommanderRequest,
  CommanderResponse,
  CommanderResponseMetadata,
} from '../commander/CommanderClient';

export const OPENAI_COMMANDER_MODEL = 'gpt-6-astra' as const;
export const OPENAI_COMMANDER_MODELS = ['gpt-5.6-luna', 'gpt-6-astra'] as const;
export type OpenAICommanderModel = (typeof OPENAI_COMMANDER_MODELS)[number];
export type OpenAICommanderReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh' | 'max';

export interface OpenAICommanderClientOptions {
  readonly apiKey?: string;
  readonly model?: OpenAICommanderModel;
  readonly reasoningEffort?: OpenAICommanderReasoningEffort;
  readonly maxOutputTokens?: number;
  readonly endpoint?: string;
  readonly fetch?: typeof fetch;
  readonly now?: () => number;
}

/** Server-only Responses API adapter. Never import this module from browser code. */
export class OpenAICommanderClient implements CommanderClient {
  private readonly apiKey: string;
  private readonly model: OpenAICommanderModel;
  private readonly reasoningEffort: OpenAICommanderReasoningEffort;
  private readonly maxOutputTokens: number;
  private readonly endpoint: string;
  private readonly fetch: typeof fetch;
  private readonly now: () => number;

  constructor(options: OpenAICommanderClientOptions = {}) {
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey || apiKey.trim().length === 0) {
      throw new OpenAICommanderError('OPENAI_API_KEY is required');
    }
    this.apiKey = apiKey.trim();
    this.model = validModel(options.model ?? OPENAI_COMMANDER_MODEL);
    this.reasoningEffort = validReasoningEffort(
      options.reasoningEffort ?? defaultCommanderReasoning(this.model),
    );
    this.maxOutputTokens = positiveInteger(options.maxOutputTokens ?? 4_096, 'maxOutputTokens');
    this.endpoint = validEndpoint(options.endpoint ?? 'https://api.openai.com/v1/responses');
    this.fetch = options.fetch ?? globalThis.fetch;
    this.now = options.now ?? performance.now.bind(performance);
  }

  async plan(request: CommanderRequest, signal?: AbortSignal): Promise<CommanderResponse> {
    const startedAt = this.now();
    let response: Response;
    try {
      response = await this.fetch(this.endpoint, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${this.apiKey}`,
          'content-type': 'application/json',
        },
        body: JSON.stringify(
          openAIRequestBody(request, this.reasoningEffort, this.maxOutputTokens, this.model),
        ),
        signal,
      });
    } catch (error) {
      if (signal?.aborted) throw error;
      throw new OpenAICommanderError(
        `OpenAI request failed: ${errorMessage(error).replaceAll(this.apiKey, '[redacted]')}`,
      );
    }

    const providerRequestId = response.headers.get('x-request-id') ?? undefined;
    let payload: unknown;
    try {
      payload = await readJson(response);
    } catch (error) {
      throw new OpenAICommanderError(errorMessage(error).replaceAll(this.apiKey, '[redacted]'), {
        requestedModel: this.model,
        reasoningEffort: this.reasoningEffort,
        latencyMs: Math.max(0, this.now() - startedAt),
        providerRequestId,
      });
    }
    if (!response.ok) {
      throw new OpenAICommanderError(
        `OpenAI request failed with HTTP ${response.status}${providerRequestId ? ` (${providerRequestId})` : ''}: ${apiErrorMessage(payload).replaceAll(this.apiKey, '[redacted]')}`,
        {
          requestedModel: this.model,
          reasoningEffort: this.reasoningEffort,
          latencyMs: Math.max(0, this.now() - startedAt),
          providerRequestId,
        },
      );
    }
    let parsed: ParsedResponse;
    try {
      parsed = parseResponse(payload);
      if (parsed.model !== this.model && !parsed.model.startsWith(`${this.model}-`)) {
        throw new OpenAICommanderError(
          `OpenAI returned unexpected model ${parsed.model}; requested ${this.model}`,
        );
      }
    } catch (error) {
      const failed = recordOrNull(payload);
      const usage = parseUsage(failed?.usage);
      throw new OpenAICommanderError(errorMessage(error).replaceAll(this.apiKey, '[redacted]'), {
        requestedModel: this.model,
        reasoningEffort: this.reasoningEffort,
        latencyMs: Math.max(0, this.now() - startedAt),
        providerRequestId,
        model: typeof failed?.model === 'string' ? failed.model : undefined,
        responseId: typeof failed?.id === 'string' ? failed.id : undefined,
        tokensIn: usage?.inputTokens,
        tokensOut: usage?.outputTokens,
        reasoningTokens: usage?.reasoningTokens,
        cachedInputTokens: usage?.cachedInputTokens,
      });
    }
    return {
      decision: parsed.decision,
      metadata: {
        requestedModel: this.model,
        reasoningEffort: this.reasoningEffort,
        model: parsed.model,
        latencyMs: Math.max(0, this.now() - startedAt),
        tokensIn: parsed.usage?.inputTokens,
        tokensOut: parsed.usage?.outputTokens,
        reasoningTokens: parsed.usage?.reasoningTokens,
        cachedInputTokens: parsed.usage?.cachedInputTokens,
        responseId: parsed.responseId,
        providerRequestId,
      },
    };
  }
}

export class OpenAICommanderError extends Error {
  constructor(
    message: string,
    readonly metadata?: CommanderResponseMetadata,
  ) {
    super(message);
    this.name = 'OpenAICommanderError';
  }
}

export function openAIRequestBody(
  request: CommanderRequest,
  reasoningEffort?: OpenAICommanderReasoningEffort,
  maxOutputTokens = 4_096,
  model: OpenAICommanderModel = OPENAI_COMMANDER_MODEL,
): Record<string, unknown> {
  return {
    model: validModel(model),
    store: false,
    reasoning: {
      effort: validReasoningEffort(reasoningEffort ?? defaultCommanderReasoning(model)),
    },
    max_output_tokens: positiveInteger(maxOutputTokens, 'maxOutputTokens'),
    instructions: COMMANDER_INSTRUCTIONS,
    input: [
      {
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: JSON.stringify({
              requestId: request.requestId,
              triggers: request.triggers,
              strategicSummary: request.summary,
              currentPlan: request.currentPlan,
              ...(request.recoveryEvidence ? { recoveryEvidence: request.recoveryEvidence } : {}),
              ...(request.trajectory ? { trajectory: request.trajectory } : {}),
              ...(request.previousPlanOutcome
                ? { previousPlanOutcome: request.previousPlanOutcome }
                : {}),
            }),
          },
        ],
      },
    ],
    text: {
      format: {
        type: 'json_schema',
        name: 'snowgym_command_plan',
        strict: true,
        schema: structuredOutputSchema(),
      },
    },
  };
}

const COMMANDER_INSTRUCTIONS = `You are the slow strategic commander for SnowGym's blue team.
Return one strict command plan for the next several seconds. Use only the bounded symbolic roles,
missions, objectives, approaches, and engagement policies in the response schema. Never invent unit
IDs, enemy IDs, coordinates, timings, physical moves, throws, or fields outside the schema. Account
for the stated replan triggers and current strategic summary. The host will assign units, resolve
symbolic objectives against newer state, validate the plan again, and control all physical actions.
When trajectory or previous-plan outcome evidence is present, use its bounded group-level trends to
correct strategy without trying to correct individual physical actions.`;

function structuredOutputSchema(): Record<string, unknown> {
  const { $schema: _schema, $id: _id, title: _title, ...schema } = commandPlanSchema;
  return normalizeStructuredOutputSchema(schema) as Record<string, unknown>;
}

function normalizeStructuredOutputSchema(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(normalizeStructuredOutputSchema);
  const source = recordOrNull(value);
  if (!source) return value;
  const normalized = Object.fromEntries(
    Object.entries(source).map(([key, child]) => [key, normalizeStructuredOutputSchema(child)]),
  );
  if (
    normalized.type === undefined &&
    (typeof normalized.const === 'string' ||
      (Array.isArray(normalized.enum) && normalized.enum.every((item) => typeof item === 'string')))
  ) {
    normalized.type = 'string';
  }
  return normalized;
}

interface ParsedResponse {
  readonly responseId: string;
  readonly model: string;
  readonly decision: unknown;
  readonly usage?: {
    readonly inputTokens?: number;
    readonly outputTokens?: number;
    readonly reasoningTokens?: number;
    readonly cachedInputTokens?: number;
  };
}

function parseResponse(value: unknown): ParsedResponse {
  const response = record(value, 'OpenAI response');
  const responseId = nonEmptyString(response.id, 'OpenAI response id');
  const model = nonEmptyString(response.model, 'OpenAI response model');
  if (response.status !== 'completed') {
    const reason = recordOrNull(response.incomplete_details)?.reason;
    throw new OpenAICommanderError(
      `OpenAI response ended with status ${String(response.status)}${reason ? `: ${String(reason)}` : ''}`,
    );
  }
  if (!Array.isArray(response.output)) {
    throw new OpenAICommanderError('OpenAI response output must be an array');
  }
  for (const itemValue of response.output) {
    const item = record(itemValue, 'OpenAI output item');
    if (item.type !== 'message' || !Array.isArray(item.content)) continue;
    for (const contentValue of item.content) {
      const content = record(contentValue, 'OpenAI output content');
      if (content.type === 'refusal') {
        throw new OpenAICommanderError(
          `OpenAI refused the plan request: ${String(content.refusal)}`,
        );
      }
      if (content.type !== 'output_text') continue;
      const text = nonEmptyString(content.text, 'OpenAI output text');
      let decision: unknown;
      try {
        decision = JSON.parse(text);
      } catch (error) {
        throw new OpenAICommanderError(`OpenAI output was not valid JSON: ${errorMessage(error)}`);
      }
      return { responseId, model, decision, usage: parseUsage(response.usage) };
    }
  }
  throw new OpenAICommanderError('OpenAI response did not contain output_text');
}

function parseUsage(value: unknown): ParsedResponse['usage'] {
  const usage = recordOrNull(value);
  if (!usage) return undefined;
  const inputDetails = recordOrNull(usage.input_tokens_details);
  const outputDetails = recordOrNull(usage.output_tokens_details);
  return {
    inputTokens: optionalNonNegativeInteger(usage.input_tokens),
    outputTokens: optionalNonNegativeInteger(usage.output_tokens),
    cachedInputTokens: optionalNonNegativeInteger(inputDetails?.cached_tokens),
    reasoningTokens: optionalNonNegativeInteger(outputDetails?.reasoning_tokens),
  };
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (text.length === 0) return {};
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new OpenAICommanderError(`OpenAI returned invalid JSON: ${errorMessage(error)}`);
  }
}

function apiErrorMessage(value: unknown): string {
  const outer = recordOrNull(value);
  const error = recordOrNull(outer?.error);
  return typeof error?.message === 'string' && error.message.length > 0
    ? error.message
    : 'unknown error';
}

function record(value: unknown, name: string): Record<string, unknown> {
  const result = recordOrNull(value);
  if (!result) throw new OpenAICommanderError(`${name} must be an object`);
  return result;
}

function recordOrNull(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function nonEmptyString(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new OpenAICommanderError(`${name} must be a non-empty string`);
  }
  return value;
}

function optionalNonNegativeInteger(value: unknown): number | undefined {
  return Number.isSafeInteger(value) && (value as number) >= 0 ? (value as number) : undefined;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
}

function validEndpoint(value: string): string {
  const url = new URL(value);
  const loopback = url.hostname === '127.0.0.1' || url.hostname === 'localhost';
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) {
    throw new RangeError('endpoint must use HTTPS except for loopback tests');
  }
  return url.toString();
}

export function defaultCommanderReasoning(
  model: OpenAICommanderModel,
): OpenAICommanderReasoningEffort {
  return validModel(model) === 'gpt-6-astra' ? 'low' : 'medium';
}

export function validReasoningEffort(value: string): OpenAICommanderReasoningEffort {
  if (
    value === 'low' ||
    value === 'medium' ||
    value === 'high' ||
    value === 'xhigh' ||
    value === 'max'
  ) {
    return value;
  }
  throw new RangeError('reasoningEffort must be low, medium, high, xhigh, or max');
}

function validModel(value: string): OpenAICommanderModel {
  if (value === 'gpt-5.6-luna' || value === 'gpt-6-astra') return value;
  throw new RangeError('commander model must be gpt-5.6-luna or gpt-6-astra');
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
