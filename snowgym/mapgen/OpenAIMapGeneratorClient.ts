import candidateSchema from './map-candidate.schema.json';
import { parseMapCandidate } from './MapValidator';
import type { MapCandidate, MapGenerationRequest, MapValidationFinding } from './types';

export const OPENAI_MAP_GENERATOR_MODEL = 'gpt-5.6-luna' as const;
export type MapGeneratorReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh' | 'max';

export interface OpenAIMapGeneratorClientOptions {
  apiKey?: string;
  reasoningEffort?: MapGeneratorReasoningEffort;
  maxOutputTokens?: number;
  endpoint?: string;
  fetch?: typeof fetch;
  now?: () => number;
}

export interface MapGeneratorResponse {
  candidate: MapCandidate;
  metadata: {
    model: string;
    reasoningEffort: MapGeneratorReasoningEffort;
    latencyMs: number;
    responseId: string;
    providerRequestId?: string;
    inputTokens?: number;
    outputTokens?: number;
    reasoningTokens?: number;
    cachedInputTokens?: number;
  };
}

export interface MapRepairContext {
  rejectedCandidate: unknown;
  errors: MapValidationFinding[];
}

export class OpenAIMapGeneratorClient {
  private readonly apiKey: string;
  private readonly reasoningEffort: MapGeneratorReasoningEffort;
  private readonly maxOutputTokens: number;
  private readonly endpoint: string;
  private readonly fetch: typeof fetch;
  private readonly now: () => number;

  constructor(options: OpenAIMapGeneratorClientOptions = {}) {
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey || apiKey.trim().length === 0)
      throw new OpenAIMapGeneratorError('OPENAI_API_KEY is required');
    this.apiKey = apiKey.trim();
    this.reasoningEffort = reasoning(options.reasoningEffort ?? 'medium');
    this.maxOutputTokens = positiveInteger(options.maxOutputTokens ?? 8_192, 'maxOutputTokens');
    this.endpoint = endpoint(options.endpoint ?? 'https://api.openai.com/v1/responses');
    this.fetch = options.fetch ?? globalThis.fetch;
    this.now = options.now ?? performance.now.bind(performance);
  }

  async generate(
    request: MapGenerationRequest,
    repair?: MapRepairContext,
    signal?: AbortSignal,
  ): Promise<MapGeneratorResponse> {
    const startedAt = this.now();
    let response: Response;
    try {
      response = await this.fetch(this.endpoint, {
        method: 'POST',
        headers: { authorization: `Bearer ${this.apiKey}`, 'content-type': 'application/json' },
        body: JSON.stringify(
          mapGeneratorRequestBody(request, this.reasoningEffort, this.maxOutputTokens, repair),
        ),
        signal,
      });
    } catch (error) {
      throw new OpenAIMapGeneratorError(`OpenAI request failed: ${message(error)}`);
    }
    const providerRequestId = response.headers.get('x-request-id') ?? undefined;
    const payload = await responseJson(response);
    if (!response.ok) {
      throw new OpenAIMapGeneratorError(
        `OpenAI request failed with HTTP ${response.status}${providerRequestId ? ` (${providerRequestId})` : ''}: ${apiError(payload)}`,
      );
    }
    const parsed = parseResponse(payload);
    return {
      candidate: parseMapCandidate(parsed.output),
      metadata: {
        model: parsed.model,
        reasoningEffort: this.reasoningEffort,
        latencyMs: Math.max(0, this.now() - startedAt),
        responseId: parsed.responseId,
        providerRequestId,
        inputTokens: parsed.inputTokens,
        outputTokens: parsed.outputTokens,
        reasoningTokens: parsed.reasoningTokens,
        cachedInputTokens: parsed.cachedInputTokens,
      },
    };
  }
}

export function mapGeneratorRequestBody(
  request: MapGenerationRequest,
  reasoningEffort: MapGeneratorReasoningEffort = 'medium',
  maxOutputTokens = 8_192,
  repair?: MapRepairContext,
): Record<string, unknown> {
  return {
    model: OPENAI_MAP_GENERATOR_MODEL,
    store: false,
    reasoning: { effort: reasoning(reasoningEffort) },
    max_output_tokens: positiveInteger(maxOutputTokens, 'maxOutputTokens'),
    instructions: MAP_GENERATOR_INSTRUCTIONS,
    input: [
      {
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: JSON.stringify({
              request,
              ...(repair
                ? {
                    repair: {
                      rejectedCandidate: repair.rejectedCandidate,
                      validationErrors: repair.errors.map(({ code, path, message: error }) => ({
                        code,
                        path,
                        message: error,
                      })),
                    },
                  }
                : {}),
            }),
          },
        ],
      },
    ],
    text: {
      format: {
        type: 'json_schema',
        name: 'snowgym_map_candidate',
        strict: true,
        schema: structuredOutputSchema(),
      },
    },
  };
}

const MAP_GENERATOR_INSTRUCTIONS = `You design exact SnowGym battlefield geometry for research.
Return one strict map candidate. Use only tree, rock, fort, fence, and prop. Player spawns are blue;
enemy spawns are red. Respect the exact arena size, spawn capacities, object budget, topology,
symmetry, density, and cover request. Coordinates are centered at (0,0). Keep full obstacle
footprints inside the arena, keep spawns separated and clear of blocking objects, and preserve a
traversable route from every spawn to opposing engagement space. Trees and rocks use radius;
forts and fences use width and height; props have no size attribute. Never output rotation,
gameplay flags, code, textures, unit IDs, or fields outside the schema. If repair evidence is
present, correct every listed deterministic validation error without changing the request.`;

function structuredOutputSchema(): Record<string, unknown> {
  const { $schema: _schema, $id: _id, ...schema } = candidateSchema;
  return schema;
}

interface ParsedProviderResponse {
  responseId: string;
  model: string;
  output: unknown;
  inputTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
  cachedInputTokens?: number;
}

function parseResponse(value: unknown): ParsedProviderResponse {
  const response = record(value, 'OpenAI response');
  const responseId = string(response.id, 'OpenAI response.id');
  const model = string(response.model, 'OpenAI response.model');
  if (response.status !== 'completed') {
    const reason = optionalRecord(response.incomplete_details)?.reason;
    throw new OpenAIMapGeneratorError(
      `OpenAI response ended with status ${String(response.status)}${reason ? `: ${String(reason)}` : ''}`,
    );
  }
  if (!Array.isArray(response.output))
    throw new OpenAIMapGeneratorError('OpenAI response.output must be an array');
  for (const itemValue of response.output) {
    const item = record(itemValue, 'OpenAI output item');
    if (item.type !== 'message' || !Array.isArray(item.content)) continue;
    for (const contentValue of item.content) {
      const content = record(contentValue, 'OpenAI output content');
      if (content.type === 'refusal')
        throw new OpenAIMapGeneratorError(
          `OpenAI refused map generation: ${String(content.refusal)}`,
        );
      if (content.type !== 'output_text') continue;
      const raw = string(content.text, 'OpenAI output text');
      let output: unknown;
      try {
        output = JSON.parse(raw);
      } catch (error) {
        throw new OpenAIMapGeneratorError(`OpenAI output was not valid JSON: ${message(error)}`);
      }
      const usage = optionalRecord(response.usage);
      const inputDetails = optionalRecord(usage?.input_tokens_details);
      const outputDetails = optionalRecord(usage?.output_tokens_details);
      return {
        responseId,
        model,
        output,
        inputTokens: optionalInteger(usage?.input_tokens),
        outputTokens: optionalInteger(usage?.output_tokens),
        reasoningTokens: optionalInteger(outputDetails?.reasoning_tokens),
        cachedInputTokens: optionalInteger(inputDetails?.cached_tokens),
      };
    }
  }
  throw new OpenAIMapGeneratorError('OpenAI response did not contain output_text');
}

async function responseJson(response: Response): Promise<unknown> {
  const body = await response.text();
  if (body.length === 0) return {};
  try {
    return JSON.parse(body);
  } catch (error) {
    throw new OpenAIMapGeneratorError(`OpenAI returned invalid JSON: ${message(error)}`);
  }
}

function apiError(value: unknown): string {
  const error = optionalRecord(optionalRecord(value)?.error);
  return typeof error?.message === 'string' ? error.message : 'unknown API error';
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value))
    throw new OpenAIMapGeneratorError(`${name} must be an object`);
  return value as Record<string, unknown>;
}

function optionalRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function string(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.length === 0)
    throw new OpenAIMapGeneratorError(`${name} must be a non-empty string`);
  return value;
}

function optionalInteger(value: unknown): number | undefined {
  return Number.isSafeInteger(value) && (value as number) >= 0 ? (value as number) : undefined;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0)
    throw new RangeError(`${name} must be a positive integer`);
  return value;
}

function reasoning(value: string): MapGeneratorReasoningEffort {
  if (
    value === 'low' ||
    value === 'medium' ||
    value === 'high' ||
    value === 'xhigh' ||
    value === 'max'
  )
    return value;
  throw new RangeError('reasoningEffort must be low, medium, high, xhigh, or max');
}

function endpoint(value: string): string {
  const parsed = new URL(value);
  if (
    parsed.protocol !== 'https:' &&
    parsed.hostname !== '127.0.0.1' &&
    parsed.hostname !== 'localhost'
  ) {
    throw new RangeError('endpoint must use HTTPS except for loopback tests');
  }
  return parsed.toString();
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class OpenAIMapGeneratorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OpenAIMapGeneratorError';
  }
}
