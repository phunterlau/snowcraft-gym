import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs, type ParseArgsOptionsConfig } from 'node:util';
import { readMapArtifact, updateArtifactEvaluation, writeMapArtifact } from './ArtifactStore';
import { createMapGenerationRequest, type MapGenerationRequestOptions } from './GenerationRequest';
import { createMirroredControlMap } from './DeterministicControl';
import { evaluateGeneratedMap } from './MapEvaluator';
import { generateValidatedMap } from './MapGenerationService';
import { validateGeneratedMap } from './MapValidator';
import {
  OpenAIMapGeneratorClient,
  type MapGeneratorReasoningEffort,
} from './OpenAIMapGeneratorClient';
import { promoteGeneratedMap } from './Promotion';
import type { MapGenerationRequest } from './types';

async function main(): Promise<void> {
  const [command, ...args] = process.argv.slice(2);
  try {
    switch (command) {
      case 'generate':
        await generate(args);
        break;
      case 'validate':
        validate(args);
        break;
      case 'control':
        control(args);
        break;
      case 'evaluate':
        evaluate(args);
        break;
      case 'suite':
        await suite(args);
        break;
      case 'promote':
        promote(args);
        break;
      case 'help':
      case '--help':
      case '-h':
      case undefined:
        help();
        break;
      default:
        throw new RangeError(
          `unknown command ${JSON.stringify(command)}; expected generate, control, validate, evaluate, suite, or promote`,
        );
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

async function generate(argv: string[]): Promise<void> {
  const { values } = parseArgs({
    args: argv,
    options: generationOptions(),
    strict: true,
  });
  if (!values.prompt) throw new RangeError('--prompt is required');
  if (!values.output) throw new RangeError('--output is required');
  if (existsSync(resolve(values.output)) && !values.force) {
    throw new Error(`refusing to overwrite ${resolve(values.output)}; pass --force to replace it`);
  }
  const reasoning = parseReasoning(values.reasoning);
  const request = requestFromValues(values);
  const client = new OpenAIMapGeneratorClient({
    reasoningEffort: reasoning,
    maxOutputTokens: integer(values['max-output-tokens'], 'max-output-tokens'),
  });
  const result = await generateValidatedMap(client, request, {
    maxRequests: parseMaxRequests(values['max-requests']),
  });
  const evaluated =
    values.evaluate || values.replay
      ? evaluateGeneratedMap(result.candidate.map, {
          seeds: parseSeeds(values.seeds),
          includeSwappedSpawns: true,
        })
      : undefined;
  const artifact = writeMapArtifact({
    output: values.output,
    map: result.candidate.map,
    request,
    validation: result.validationHistory,
    attempts: result.attempts,
    reasoningEffort: reasoning,
    force: values.force,
    evaluation: evaluated?.report,
    replay: values.replay ? evaluated?.replay : undefined,
  });
  print(
    {
      ok: true,
      artifact: artifact.directory,
      mapDigest: artifact.manifest.mapDigest,
      attempts: artifact.manifest.attempts.length,
      evaluated: Boolean(evaluated),
      replay: Boolean(artifact.replay),
    },
    values.json,
  );
}

function validate(argv: string[]): void {
  const { positionals, values } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: { json: { type: 'boolean', default: false } },
    strict: true,
  });
  if (positionals.length !== 1)
    throw new RangeError('validate requires one map file or artifact directory');
  const map = readMapInput(positionals[0]);
  const report = validateGeneratedMap(map);
  print(report, values.json);
  if (!report.valid) process.exitCode = 1;
}

function control(argv: string[]): void {
  const { values } = parseArgs({ args: argv, options: generationOptions(), strict: true });
  if (!values.output) throw new RangeError('--output is required');
  if (existsSync(resolve(values.output)) && !values.force) {
    throw new Error(`refusing to overwrite ${resolve(values.output)}; pass --force to replace it`);
  }
  const request = requestFromValues({
    ...values,
    prompt: values.prompt ?? 'Deterministic mirrored research control',
  });
  const map = createMirroredControlMap(request);
  const validation = validateGeneratedMap(map, request);
  if (!validation.valid) throw new Error('deterministic control failed validation');
  const evaluated =
    values.evaluate || values.replay
      ? evaluateGeneratedMap(map, { seeds: parseSeeds(values.seeds) })
      : undefined;
  const artifact = writeMapArtifact({
    output: values.output,
    map,
    request,
    validation: [validation],
    attempts: [],
    reasoningEffort: 'none',
    model: 'deterministic-mirror-v0',
    force: values.force,
    evaluation: evaluated?.report,
    replay: values.replay ? evaluated?.replay : undefined,
  });
  print(
    { ok: true, artifact: artifact.directory, mapDigest: artifact.manifest.mapDigest },
    values.json,
  );
}

function evaluate(argv: string[]): void {
  const { positionals, values } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      seeds: { type: 'string', default: '41,42,43' },
      replay: { type: 'boolean', default: false },
      force: { type: 'boolean', default: false },
      json: { type: 'boolean', default: false },
    },
    strict: true,
  });
  if (positionals.length !== 1) throw new RangeError('evaluate requires one artifact directory');
  const artifact = readMapArtifact(positionals[0]);
  if (artifact.evaluation && !values.force) {
    throw new Error('evaluation.json already exists; pass --force to replace it');
  }
  if (values.replay && artifact.replay && !values.force) {
    throw new Error('replay.json already exists; pass --force to replace it');
  }
  const result = evaluateGeneratedMap(artifact.map, {
    seeds: parseSeeds(values.seeds),
    includeSwappedSpawns: true,
  });
  updateArtifactEvaluation(
    artifact.directory,
    result.report,
    values.replay ? result.replay : undefined,
  );
  print({ ok: true, artifact: artifact.directory, ...result.report.summary }, values.json);
}

async function suite(argv: string[]): Promise<void> {
  const { positionals, values } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      output: { type: 'string' },
      reasoning: { type: 'string', default: 'medium' },
      'max-requests': { type: 'string', default: '2' },
      'max-maps': { type: 'string', default: '20' },
      evaluate: { type: 'boolean', default: false },
      replay: { type: 'boolean', default: false },
      force: { type: 'boolean', default: false },
      json: { type: 'boolean', default: false },
    },
    strict: true,
  });
  if (positionals.length !== 1 || !values.output)
    throw new RangeError('suite requires CONFIG and --output DIR');
  const outputDirectory = resolve(values.output);
  const source = JSON.parse(readFileSync(resolve(positionals[0]), 'utf8')) as unknown;
  const entries = suiteEntries(source);
  const maxMaps = integer(values['max-maps'], 'max-maps');
  if (entries.length > maxMaps)
    throw new RangeError(`suite has ${entries.length} maps, exceeding --max-maps ${maxMaps}`);
  const suiteManifestPath = `${outputDirectory}/suite-manifest.json`;
  if (existsSync(suiteManifestPath) && !values.force) {
    throw new Error(`refusing to overwrite ${suiteManifestPath}; pass --force to replace it`);
  }
  const occupiedArtifact = entries
    .map((entry) => `${outputDirectory}/${entry.id}`)
    .find((path) => existsSync(path));
  if (occupiedArtifact && !values.force) {
    throw new Error(`refusing to overwrite ${occupiedArtifact}; pass --force to replace it`);
  }
  const reasoning = parseReasoning(values.reasoning);
  const maxRequests = parseMaxRequests(values['max-requests']);
  const client = new OpenAIMapGeneratorClient({ reasoningEffort: reasoning });
  const summaries: Array<{ id: string; mapDigest: string; attempts: number }> = [];
  for (const entry of entries) {
    const request = createMapGenerationRequest(entry);
    const result = await generateValidatedMap(client, request, {
      maxRequests,
    });
    const evaluated =
      values.evaluate || values.replay ? evaluateGeneratedMap(result.candidate.map) : undefined;
    const artifact = writeMapArtifact({
      output: `${outputDirectory}/${entry.id}`,
      map: result.candidate.map,
      request,
      validation: result.validationHistory,
      attempts: result.attempts,
      reasoningEffort: reasoning,
      force: values.force,
      evaluation: evaluated?.report,
      replay: values.replay ? evaluated?.replay : undefined,
    });
    summaries.push({
      id: entry.id,
      mapDigest: artifact.manifest.mapDigest,
      attempts: result.attempts.length,
    });
  }
  mkdirSync(outputDirectory, { recursive: true });
  writeFileSync(
    suiteManifestPath,
    `${JSON.stringify(
      {
        schemaVersion: 'snowgym.map-suite-result.v0',
        sourceConfig: resolve(positionals[0]),
        model: 'gpt-5.6-luna',
        reasoningEffort: reasoning,
        maxRequestsPerMap: maxRequests,
        maps: entries.map((entry, index) => ({
          ...summaries[index],
          split: entry.split ?? 'development',
          requestDigest: readMapArtifact(`${outputDirectory}/${entry.id}`).manifest.requestDigest,
        })),
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
  print({ ok: true, mapCount: summaries.length, maps: summaries }, values.json);
}

function promote(argv: string[]): void {
  const { positionals, values } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      id: { type: 'string' },
      force: { type: 'boolean', default: false },
      json: { type: 'boolean', default: false },
    },
    strict: true,
  });
  if (positionals.length !== 1 || !values.id)
    throw new RangeError('promote requires ARTIFACT and --id arenaNAME.json');
  const artifact = readMapArtifact(positionals[0]);
  if (artifact.manifest.mapDigest !== validateGeneratedMap(artifact.map).mapDigest) {
    throw new Error('artifact map digest does not match manifest');
  }
  print(
    promoteGeneratedMap({ map: artifact.map, id: values.id, force: values.force }),
    values.json,
  );
}

const GENERATION_OPTIONS = {
  prompt: { type: 'string' },
  output: { type: 'string' },
  'blue-capacity': { type: 'string', default: '10' },
  'red-capacity': { type: 'string', default: '10' },
  width: { type: 'string', default: '64' },
  height: { type: 'string', default: '48' },
  topology: { type: 'string', default: 'mixed' },
  symmetry: { type: 'string', default: 'mirror' },
  density: { type: 'string', default: 'medium' },
  'object-budget': { type: 'string', default: '40' },
  cover: { type: 'string', default: 'medium' },
  split: { type: 'string', default: 'development' },
  tags: { type: 'string', default: '' },
  reasoning: { type: 'string', default: 'medium' },
  'max-requests': { type: 'string', default: '2' },
  'max-output-tokens': { type: 'string', default: '8192' },
  evaluate: { type: 'boolean', default: false },
  replay: { type: 'boolean', default: false },
  seeds: { type: 'string', default: '41,42,43' },
  force: { type: 'boolean', default: false },
  json: { type: 'boolean', default: false },
} as const satisfies ParseArgsOptionsConfig;

function generationOptions(): typeof GENERATION_OPTIONS {
  return GENERATION_OPTIONS;
}

function requestFromValues(
  values: Record<string, string | boolean | undefined>,
): MapGenerationRequest {
  parseMaxRequests(asString(values['max-requests']));
  return createMapGenerationRequest({
    brief: asString(values.prompt),
    blueCapacity: integer(asString(values['blue-capacity']), 'blue-capacity'),
    redCapacity: integer(asString(values['red-capacity']), 'red-capacity'),
    width: number(asString(values.width), 'width'),
    height: number(asString(values.height), 'height'),
    topology: asString(values.topology) as MapGenerationRequestOptions['topology'],
    symmetry: asString(values.symmetry) as MapGenerationRequestOptions['symmetry'],
    density: asString(values.density) as MapGenerationRequestOptions['density'],
    objectBudget: integer(asString(values['object-budget']), 'object-budget'),
    desiredCover: asString(values.cover) as MapGenerationRequestOptions['desiredCover'],
    split: asString(values.split) as MapGenerationRequestOptions['split'],
    tags: asString(values.tags)
      .split(',')
      .map((tag) => tag.trim())
      .filter(Boolean),
  });
}

function parseMaxRequests(value: string | undefined): 1 | 2 {
  const parsed = integer(value, 'max-requests');
  if (parsed !== 1 && parsed !== 2) throw new RangeError('max-requests must be 1 or 2');
  return parsed;
}

function suiteEntries(value: unknown): Array<MapGenerationRequestOptions & { id: string }> {
  if (typeof value !== 'object' || value === null || Array.isArray(value))
    throw new RangeError('suite config must be an object');
  const record = value as Record<string, unknown>;
  if (record.schemaVersion !== 'snowgym.map-suite.v0' || !Array.isArray(record.maps)) {
    throw new RangeError('suite config must use snowgym.map-suite.v0 and contain maps[]');
  }
  const allowed = new Set([
    'id',
    'brief',
    'blueCapacity',
    'redCapacity',
    'width',
    'height',
    'topology',
    'symmetry',
    'density',
    'objectBudget',
    'desiredCover',
    'split',
    'tags',
  ]);
  const entries = record.maps.map((entry, index) => {
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry))
      throw new RangeError(`maps[${index}] must be an object`);
    const item = entry as Record<string, unknown>;
    const unknown = Object.keys(item).filter((key) => !allowed.has(key));
    if (unknown.length > 0)
      throw new RangeError(`maps[${index}] has unknown fields: ${unknown.sort().join(', ')}`);
    if (typeof item.id !== 'string' || !/^[A-Za-z0-9._-]+$/.test(item.id))
      throw new RangeError(`maps[${index}].id is invalid`);
    return { ...(item as unknown as MapGenerationRequestOptions), id: item.id };
  });
  if (new Set(entries.map((entry) => entry.id)).size !== entries.length) {
    throw new RangeError('suite map ids must be unique');
  }
  return entries;
}

function readMapInput(path: string): unknown {
  const absolute = resolve(path);
  const file = statSync(absolute).isDirectory() ? `${absolute}/map.json` : absolute;
  return JSON.parse(readFileSync(file, 'utf8'));
}

function parseSeeds(value: string | undefined): number[] {
  const seeds = asString(value)
    .split(',')
    .map((seed) => integer(seed.trim(), 'seed'));
  if (seeds.length === 0) throw new RangeError('at least one seed is required');
  return seeds;
}

function parseReasoning(value: string | undefined): MapGeneratorReasoningEffort {
  if (
    value === 'low' ||
    value === 'medium' ||
    value === 'high' ||
    value === 'xhigh' ||
    value === 'max'
  )
    return value;
  throw new RangeError('reasoning must be low, medium, high, xhigh, or max');
}

function integer(value: string | undefined, name: string): number {
  if (value === undefined || !/^-?\d+$/.test(value))
    throw new RangeError(`${name} must be an integer`);
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new RangeError(`${name} must be a safe integer`);
  return parsed;
}

function number(value: string | undefined, name: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new RangeError(`${name} must be finite`);
  return parsed;
}

function asString(value: string | boolean | undefined): string {
  if (typeof value !== 'string') throw new RangeError('expected a string option');
  return value;
}

function print(value: unknown, json: boolean | undefined): void {
  if (json) console.log(JSON.stringify(value));
  else console.log(JSON.stringify(value, null, 2));
}

function help(): void {
  console.log(`SnowGym GPT-5.6 Luna map generator

Usage:
  npm run snowgym:mapgen -- generate --prompt TEXT --output DIR [options]
  npm run snowgym:mapgen -- control --output DIR [generation options]
  npm run snowgym:mapgen -- validate MAP_OR_ARTIFACT [--json]
  npm run snowgym:mapgen -- evaluate ARTIFACT [--seeds 41,42,43] [--replay]
  npm run snowgym:mapgen -- suite CONFIG --output DIR [--max-maps 20]
  npm run snowgym:mapgen -- promote ARTIFACT --id arenaNAME.json

Generation is server-only, reads OPENAI_API_KEY, uses gpt-5.6-luna, and makes at
most two requests per map. Generated maps are not registered until promote.`);
}

await main();
