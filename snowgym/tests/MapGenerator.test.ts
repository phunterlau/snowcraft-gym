import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Team, type MapData } from '../../src/game/types';
import { readMapArtifact, writeMapArtifact } from '../mapgen/ArtifactStore';
import { createMapGenerationRequest } from '../mapgen/GenerationRequest';
import { evaluateGeneratedMap } from '../mapgen/MapEvaluator';
import { generateValidatedMap } from '../mapgen/MapGenerationService';
import {
  canonicalizeMap,
  digestJson,
  parseMapCandidate,
  validateGeneratedMap,
} from '../mapgen/MapValidator';
import {
  mapGeneratorRequestBody,
  OpenAIMapGeneratorClient,
  type MapGeneratorResponse,
} from '../mapgen/OpenAIMapGeneratorClient';
import { promoteGeneratedMap } from '../mapgen/Promotion';
import {
  MAP_CANDIDATE_VERSION,
  type MapCandidate,
  type MapGenerationRequest,
} from '../mapgen/types';

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) rmSync(directory, { recursive: true });
});

describe('SnowGym map generation contract', () => {
  it('validates and canonicalizes a playable exact-layout map', () => {
    const request = requestFixture();
    const report = validateGeneratedMap(mapFixture(), request);

    expect(report.valid).toBe(true);
    expect(report.mapDigest).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(report.metrics).toMatchObject({ obstacleCount: 3, decorativeObstacleCount: 1 });
    expect(report.canonicalMap?.spawns?.map((spawn) => spawn.team)).toEqual([
      Team.Enemy,
      Team.Player,
    ]);
  });

  it('produces a stable digest independent of source object order and negative zero', () => {
    const first = mapFixture();
    const second = { ...first, objects: [...first.objects].reverse(), width: 40, height: 30 };
    second.objects[1] = { ...second.objects[1], y: -0 };

    expect(digestJson(canonicalizeMap(second))).toBe(digestJson(canonicalizeMap(first)));
  });

  it.each([
    [{ ...mapFixture(), objects: [{ type: 'tree', x: 0, y: 0, rotation: 1 }] }, 'schema_invalid'],
    [
      { ...mapFixture(), objects: [{ type: 'fort', x: 20, y: 0, width: 4, height: 2 }] },
      'object_out_of_bounds',
    ],
    [{ ...mapFixture(), spawns: [{ team: 'player', x: -12, y: 0 }] }, 'spawn_capacity_invalid'],
    [
      {
        ...mapFixture(),
        objects: [{ type: 'fort', x: 0, y: 0, width: 1, height: 30 }],
      },
      'engagement_space_disconnected',
    ],
  ])('rejects invalid or misleading geometry %#', (map, code) => {
    const report = validateGeneratedMap(map);
    expect(report.valid).toBe(false);
    expect(report.findings.map((finding) => finding.code)).toContain(code);
  });

  it('reports overlapping blockers as quality evidence without rejecting a connected map', () => {
    const map = mapFixture();
    map.objects.push({ type: 'rock', x: 0.2, y: 0, radius: 0.7 });
    const report = validateGeneratedMap(map);
    expect(report.valid).toBe(true);
    expect(report.findings).toContainEqual(
      expect.objectContaining({ code: 'blocking_objects_overlap', severity: 'warning' }),
    );
  });

  it('accepts the native 10v10 map at the fixed Gym capacities', () => {
    const map = JSON.parse(
      readFileSync(new URL('../../public/maps/arena6.json', import.meta.url), 'utf8'),
    );
    const request = createMapGenerationRequest({
      brief: 'Native 10v10 control',
      blueCapacity: 10,
      redCapacity: 10,
      width: 64,
      height: 48,
      objectBudget: 40,
    });
    const report = validateGeneratedMap(map, request);
    expect(report.valid).toBe(true);
    expect(report.metrics).toMatchObject({ obstacleCount: 27 });
  });

  it('strictly parses the candidate envelope and rejects unknown fields', () => {
    expect(parseMapCandidate(candidateFixture())).toMatchObject({
      schemaVersion: MAP_CANDIDATE_VERSION,
    });
    expect(() => parseMapCandidate({ ...candidateFixture(), extra: true })).toThrow(
      'unknown fields',
    );
  });
});

describe('GPT-5.6 Luna map provider', () => {
  it('builds a stateless strict structured-output request', () => {
    const body = mapGeneratorRequestBody(requestFixture(), 'high', 5000);
    expect(body).toMatchObject({
      model: 'gpt-5.6-luna',
      store: false,
      reasoning: { effort: 'high' },
      max_output_tokens: 5000,
      text: { format: { type: 'json_schema', name: 'snowgym_map_candidate', strict: true } },
    });
    expect(JSON.stringify(body)).not.toContain('OPENAI_API_KEY');
  });

  it('parses candidate, request IDs, latency, and token usage', async () => {
    const fetch = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            id: 'resp_1',
            model: 'gpt-5.6-luna',
            status: 'completed',
            output: [
              {
                type: 'message',
                content: [{ type: 'output_text', text: JSON.stringify(candidateFixture()) }],
              },
            ],
            usage: {
              input_tokens: 100,
              output_tokens: 200,
              input_tokens_details: { cached_tokens: 25 },
              output_tokens_details: { reasoning_tokens: 50 },
            },
          }),
          { status: 200, headers: { 'x-request-id': 'req_1' } },
        ),
    );
    const times = [10, 35];
    const client = new OpenAIMapGeneratorClient({
      apiKey: 'test',
      fetch,
      now: () => times.shift()!,
    });

    const result = await client.generate(requestFixture());
    expect(result.candidate.map.name).toBe('Research Crossing');
    expect(result.metadata).toMatchObject({
      responseId: 'resp_1',
      providerRequestId: 'req_1',
      latencyMs: 25,
      inputTokens: 100,
      outputTokens: 200,
      reasoningTokens: 50,
      cachedInputTokens: 25,
    });
  });

  it('surfaces refusals and incomplete responses', async () => {
    const refusal = new OpenAIMapGeneratorClient({
      apiKey: 'test',
      fetch: async () =>
        providerResponse([{ type: 'message', content: [{ type: 'refusal', refusal: 'no' }] }]),
    });
    await expect(refusal.generate(requestFixture())).rejects.toThrow('refused');

    const incomplete = new OpenAIMapGeneratorClient({
      apiKey: 'test',
      fetch: async () => providerResponse([], 'incomplete'),
    });
    await expect(incomplete.generate(requestFixture())).rejects.toThrow('incomplete');
  });

  it('uses exactly one validator-guided repair and never a hidden third request', async () => {
    const invalid = candidateFixture();
    invalid.map.spawns = invalid.map.spawns?.slice(0, 1);
    const generate = vi
      .fn()
      .mockResolvedValueOnce({ candidate: invalid, metadata: metadata() })
      .mockResolvedValueOnce({ candidate: candidateFixture(), metadata: metadata() });
    const client = { generate } as unknown as OpenAIMapGeneratorClient;

    const result = await generateValidatedMap(client, requestFixture(), { maxRequests: 2 });
    expect(result.validationHistory).toHaveLength(2);
    expect(result.attempts.map((attempt) => attempt.outcome)).toEqual(['invalid', 'accepted']);
    expect(generate).toHaveBeenCalledTimes(2);
    expect(generate.mock.calls[1][1].errors).toContainEqual(
      expect.objectContaining({ code: 'spawn_capacity_invalid' }),
    );
  });
});

describe('map artifacts, headless evaluation, and promotion', () => {
  it('writes immutable provenance and round-trips an artifact', () => {
    const directory = temporaryDirectory();
    const output = join(directory, 'artifact');
    const validation = validateGeneratedMap(mapFixture(), requestFixture());
    const artifact = writeMapArtifact({
      output,
      map: mapFixture(),
      request: requestFixture(),
      validation: [validation],
      attempts: [{ attempt: 1, ...metadata(), outcome: 'accepted' }],
      reasoningEffort: 'medium',
      now: () => new Date('2026-09-02T00:00:00.000Z'),
      generatorRevision: 'test-revision',
    });

    expect(readMapArtifact(output).manifest).toEqual(artifact.manifest);
    expect(() =>
      writeMapArtifact({
        output,
        map: mapFixture(),
        request: requestFixture(),
        validation: [validation],
        attempts: [],
        reasoningEffort: 'medium',
      }),
    ).toThrow('refusing to overwrite');

    writeFileSync(join(output, 'map.json'), JSON.stringify({ ...mapFixture(), width: 41 }), 'utf8');
    expect(() => readMapArtifact(output)).toThrow('map digest does not match');
  });

  it('runs deterministic paired side probes and emits a normal visual replay', () => {
    const first = evaluateGeneratedMap(mapFixture(), { seeds: [7], maxTicks: 120 });
    const second = evaluateGeneratedMap(mapFixture(), { seeds: [7], maxTicks: 120 });

    expect(second).toEqual(first);
    expect(first.report.episodes).toHaveLength(2);
    expect(first.report.episodes.map((episode) => episode.swappedSpawns)).toEqual([false, true]);
    expect(first.replay?.format).toBe('snowgym.replay.v0');
    expect(first.replay?.frames[0].obstacles).toHaveLength(3);
  });

  it('promotes only through explicit markers and preserves browser/headless parity', () => {
    const root = temporaryDirectory();
    mkdirSync(join(root, 'public/maps'), { recursive: true });
    mkdirSync(join(root, 'snowgym/scenarios'), { recursive: true });
    mkdirSync(join(root, 'src'), { recursive: true });
    writeFileSync(
      join(root, 'snowgym/scenarios/maps.ts'),
      'const MAP_DATA = {\n  // MAPGEN_PROMOTED_MAPS\n};\n',
    );
    writeFileSync(join(root, 'src/main.ts'), 'const MAPS = [\n  // MAPGEN_PROMOTED_MAPS\n];\n');

    const result = promoteGeneratedMap({
      map: mapFixture(),
      id: 'arenaResearch.json',
      repositoryRoot: root,
    });
    expect(JSON.parse(readFileSync(result.mapPath, 'utf8'))).toEqual(canonicalizeMap(mapFixture()));
    expect(readFileSync(result.registryPath, 'utf8')).toContain('arenaResearch.json');
    expect(readFileSync(result.browserPath, 'utf8')).toContain('Research Crossing');
    expect(() =>
      promoteGeneratedMap({ map: mapFixture(), id: 'arenaResearch.json', repositoryRoot: root }),
    ).toThrow('refusing to overwrite');
  });
});

function requestFixture(): MapGenerationRequest {
  return createMapGenerationRequest({
    brief: 'A small balanced crossing',
    blueCapacity: 1,
    redCapacity: 1,
    width: 40,
    height: 30,
    objectBudget: 6,
  });
}

function mapFixture(): MapData {
  return {
    name: 'Research Crossing',
    width: 40,
    height: 30,
    objects: [
      { type: 'tree', x: 0, y: 6, radius: 0.5 },
      { type: 'fort', x: 0, y: 0, width: 4, height: 1 },
      { type: 'prop', x: 0, y: -6 },
    ],
    spawns: [
      { team: Team.Player, x: -15, y: 0 },
      { team: Team.Enemy, x: 15, y: 0 },
    ],
  };
}

function candidateFixture(): MapCandidate {
  return {
    schemaVersion: MAP_CANDIDATE_VERSION,
    intentSummary: 'A central crossing with sparse cover.',
    map: mapFixture(),
  };
}

function metadata(): MapGeneratorResponse['metadata'] {
  return {
    model: 'gpt-5.6-luna',
    reasoningEffort: 'medium' as const,
    latencyMs: 10,
    responseId: 'resp_test',
  };
}

function providerResponse(output: unknown[], status = 'completed'): Response {
  return new Response(JSON.stringify({ id: 'resp_test', model: 'gpt-5.6-luna', status, output }));
}

function temporaryDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), 'snowgym-mapgen-'));
  temporaryDirectories.push(directory);
  return directory;
}
