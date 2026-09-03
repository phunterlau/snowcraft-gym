import type { OpenAIMapGeneratorClient } from './OpenAIMapGeneratorClient';
import { digestJson, validateGeneratedMap } from './MapValidator';
import type {
  MapCandidate,
  MapGenerationRequest,
  MapValidationReport,
  ProviderAttemptMetadata,
} from './types';

export interface GeneratedMapResult {
  request: MapGenerationRequest;
  candidate: MapCandidate;
  validationHistory: MapValidationReport[];
  attempts: ProviderAttemptMetadata[];
}

/** One draft plus at most one validator-guided repair. There are no hidden transport retries. */
export async function generateValidatedMap(
  client: OpenAIMapGeneratorClient,
  request: MapGenerationRequest,
  options: { maxRequests?: 1 | 2; signal?: AbortSignal } = {},
): Promise<GeneratedMapResult> {
  const maxRequests = options.maxRequests ?? 2;
  const attempts: ProviderAttemptMetadata[] = [];
  const validationHistory: MapValidationReport[] = [];
  let repair: Parameters<OpenAIMapGeneratorClient['generate']>[1];
  let lastCandidate: MapCandidate | undefined;
  for (let attempt = 1; attempt <= maxRequests; attempt++) {
    try {
      const response = await client.generate(request, repair, options.signal);
      lastCandidate = response.candidate;
      const validation = validateGeneratedMap(response.candidate.map, request);
      validationHistory.push(validation);
      attempts.push({
        attempt,
        ...response.metadata,
        outcome: validation.valid ? 'accepted' : 'invalid',
      });
      if (validation.valid) {
        return {
          request,
          candidate: { ...response.candidate, map: validation.canonicalMap! },
          validationHistory,
          attempts,
        };
      }
      repair = {
        rejectedCandidate: response.candidate,
        errors: validation.findings.filter((finding) => finding.severity === 'error'),
      };
    } catch (error) {
      attempts.push({
        attempt,
        model: 'gpt-5.6-luna',
        reasoningEffort: 'unknown',
        latencyMs: 0,
        outcome: 'provider_error',
        error: error instanceof Error ? error.message : String(error),
      });
      throw new MapGenerationFailedError('provider request failed', attempts, validationHistory);
    }
  }
  throw new MapGenerationFailedError(
    `map remained invalid after ${maxRequests} request${maxRequests === 1 ? '' : 's'}${lastCandidate ? ` (${digestJson(lastCandidate)})` : ''}`,
    attempts,
    validationHistory,
  );
}

export class MapGenerationFailedError extends Error {
  constructor(
    message: string,
    readonly attempts: ProviderAttemptMetadata[],
    readonly validationHistory: MapValidationReport[],
  ) {
    super(message);
    this.name = 'MapGenerationFailedError';
  }
}
