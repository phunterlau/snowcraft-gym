import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { parseReplayRecording, type ReplayRecording } from '../replay/ReplayRecording';
import { digestJson, parseMapData, validateGeneratedMap } from './MapValidator';
import {
  MAP_ARTIFACT_VERSION,
  MAP_EVALUATION_VERSION,
  MAP_GENERATION_REQUEST_VERSION,
  type MapArtifactManifest,
  type MapEvaluationReport,
  type MapGenerationRequest,
  type MapValidationReport,
  type ProviderAttemptMetadata,
} from './types';

export interface MapArtifact {
  directory: string;
  map: ReturnType<typeof parseMapData>;
  request: MapGenerationRequest;
  manifest: MapArtifactManifest;
  validation: MapValidationReport[];
  evaluation?: MapEvaluationReport;
  replay?: ReplayRecording;
}

export function writeMapArtifact(options: {
  output: string;
  map: ReturnType<typeof parseMapData>;
  request: MapGenerationRequest;
  validation: MapValidationReport[];
  attempts: ProviderAttemptMetadata[];
  reasoningEffort: string;
  model?: MapArtifactManifest['model'];
  force?: boolean;
  now?: () => Date;
  generatorRevision?: string;
  evaluation?: MapEvaluationReport;
  replay?: ReplayRecording;
}): MapArtifact {
  const directory = resolve(options.output);
  if (existsSync(directory) && !options.force) {
    throw new Error(`refusing to overwrite ${directory}; pass --force to replace it`);
  }
  const finalValidation = validateGeneratedMap(options.map, options.request);
  if (!finalValidation.valid || !finalValidation.canonicalMap || !finalValidation.mapDigest) {
    throw new Error('refusing to write an invalid map artifact');
  }
  mkdirSync(directory, { recursive: true });
  const manifest: MapArtifactManifest = {
    schemaVersion: MAP_ARTIFACT_VERSION,
    artifactId: basename(directory),
    createdAt: (options.now ?? (() => new Date()))().toISOString(),
    generatorRevision: options.generatorRevision ?? gitRevision(),
    model: options.model ?? 'gpt-5.6-luna',
    reasoningEffort: options.reasoningEffort,
    requestDigest: digestJson(options.request),
    mapDigest: finalValidation.mapDigest,
    attempts: options.attempts,
  };
  writeJson(`${directory}/map.json`, finalValidation.canonicalMap);
  writeJson(`${directory}/request.json`, options.request);
  writeJson(`${directory}/manifest.json`, manifest);
  writeJson(`${directory}/validation.json`, options.validation);
  if (options.evaluation) writeJson(`${directory}/evaluation.json`, options.evaluation);
  else if (options.force && existsSync(`${directory}/evaluation.json`))
    unlinkSync(`${directory}/evaluation.json`);
  if (options.replay) writeJson(`${directory}/replay.json`, options.replay);
  else if (options.force && existsSync(`${directory}/replay.json`))
    unlinkSync(`${directory}/replay.json`);
  return {
    directory,
    map: finalValidation.canonicalMap,
    request: options.request,
    manifest,
    validation: options.validation,
    evaluation: options.evaluation,
    replay: options.replay,
  };
}

export function readMapArtifact(path: string): MapArtifact {
  const directory = resolve(path);
  const map = parseMapData(readJson(`${directory}/map.json`));
  const request = readJson(`${directory}/request.json`) as MapGenerationRequest;
  const manifest = readJson(`${directory}/manifest.json`) as MapArtifactManifest;
  const validation = readJson(`${directory}/validation.json`) as MapValidationReport[];
  if (manifest.schemaVersion !== MAP_ARTIFACT_VERSION) {
    throw new Error(`unsupported artifact schema ${String(manifest.schemaVersion)}`);
  }
  if (request.schemaVersion !== MAP_GENERATION_REQUEST_VERSION) {
    throw new Error(`unsupported request schema ${String(request.schemaVersion)}`);
  }
  const currentMapDigest = validateGeneratedMap(map, request).mapDigest;
  if (currentMapDigest === null || currentMapDigest !== manifest.mapDigest) {
    throw new Error('artifact map digest does not match its validated request and manifest');
  }
  const evaluationPath = `${directory}/evaluation.json`;
  const replayPath = `${directory}/replay.json`;
  const evaluation = existsSync(evaluationPath)
    ? (readJson(evaluationPath) as MapEvaluationReport)
    : undefined;
  if (
    evaluation &&
    (evaluation.schemaVersion !== MAP_EVALUATION_VERSION ||
      evaluation.mapDigest !== manifest.mapDigest)
  ) {
    throw new Error('artifact evaluation does not match the map manifest');
  }
  return {
    directory,
    map,
    request,
    manifest,
    validation,
    ...(evaluation ? { evaluation } : {}),
    ...(existsSync(replayPath) ? { replay: parseReplayRecording(readJson(replayPath)) } : {}),
  };
}

export function updateArtifactEvaluation(
  directory: string,
  evaluation: MapEvaluationReport,
  replay?: ReplayRecording,
): void {
  writeJson(`${resolve(directory)}/evaluation.json`, evaluation);
  if (replay) writeJson(`${resolve(directory)}/replay.json`, replay);
}

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(path, 'utf8'));
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function gitRevision(): string {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  } catch {
    return 'unknown';
  }
}
