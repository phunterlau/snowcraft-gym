import type { MapData } from '../../src/game/types';

export const MAP_GENERATION_REQUEST_VERSION = 'snowgym.map-generation-request.v0' as const;
export const MAP_CANDIDATE_VERSION = 'snowgym.map-candidate.v0' as const;
export const MAP_ARTIFACT_VERSION = 'snowgym.map-artifact.v0' as const;
export const MAP_VALIDATION_VERSION = 'snowgym.map-validation.v0' as const;
export const MAP_EVALUATION_VERSION = 'snowgym.map-evaluation.v0' as const;

export type MapTopology = 'open' | 'lanes' | 'chokepoint' | 'pockets' | 'mixed';
export type MapSymmetry = 'mirror' | 'rotational' | 'asymmetric';
export type MapDensity = 'sparse' | 'medium' | 'dense';
export type DatasetSplit = 'development' | 'evaluation';

export interface MapGenerationRequest {
  schemaVersion: typeof MAP_GENERATION_REQUEST_VERSION;
  brief: string;
  blueCapacity: number;
  redCapacity: number;
  width: number;
  height: number;
  topology: MapTopology;
  symmetry: MapSymmetry;
  density: MapDensity;
  objectBudget: number;
  desiredCover: 'low' | 'medium' | 'high';
  split: DatasetSplit;
  tags: string[];
}

export interface MapCandidate {
  schemaVersion: typeof MAP_CANDIDATE_VERSION;
  intentSummary: string;
  map: MapData;
}

export type MapValidationSeverity = 'error' | 'warning';

export interface MapValidationFinding {
  severity: MapValidationSeverity;
  code: string;
  path: string;
  message: string;
}

export interface MapMetrics {
  obstacleCount: number;
  blockingObstacleCount: number;
  decorativeObstacleCount: number;
  obstacleDensity: number;
  blockingFootprintFraction: number;
  overlapPairs: number;
  symmetryError: number;
  blueMeanPathLength: number | null;
  redMeanPathLength: number | null;
  pathLengthImbalance: number | null;
  blueMeanCoverDistance: number | null;
  redMeanCoverDistance: number | null;
  coverAccessImbalance: number | null;
}

export interface MapValidationReport {
  schemaVersion: typeof MAP_VALIDATION_VERSION;
  valid: boolean;
  mapDigest: string | null;
  findings: MapValidationFinding[];
  metrics: MapMetrics | null;
  canonicalMap?: MapData;
}

export interface ProviderAttemptMetadata {
  attempt: number;
  model: string;
  reasoningEffort: string;
  latencyMs: number;
  responseId?: string;
  providerRequestId?: string;
  inputTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
  cachedInputTokens?: number;
  outcome: 'accepted' | 'invalid' | 'provider_error';
  error?: string;
}

export interface MapArtifactManifest {
  schemaVersion: typeof MAP_ARTIFACT_VERSION;
  artifactId: string;
  createdAt: string;
  generatorRevision: string;
  model: 'gpt-5.6-luna' | 'deterministic-mirror-v0';
  reasoningEffort: string;
  requestDigest: string;
  mapDigest: string;
  attempts: ProviderAttemptMetadata[];
}

export interface MapEvaluationEpisode {
  seed: number;
  swappedSpawns: boolean;
  decisions: number;
  ticks: number;
  terminated: boolean;
  truncated: boolean;
  winner: 'blue' | 'red' | null;
  blueAlive: number;
  redAlive: number;
  rejectedActions: number;
  finalStateHash: string;
}

export interface MapEvaluationReport {
  schemaVersion: typeof MAP_EVALUATION_VERSION;
  mapDigest: string;
  episodes: MapEvaluationEpisode[];
  summary: {
    episodeCount: number;
    terminationRate: number;
    blueWinRate: number;
    redWinRate: number;
    meanTicks: number;
    sideAdvantage: number;
  };
}
