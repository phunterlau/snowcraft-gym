import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { afterEach, describe, expect, it } from 'vitest';
import {
  BACKEND_PILOT_CASES,
  backendBenchmarkArms,
  backendCallSchedule,
  benchmarkBackendCall,
  benchmarkDigest,
  continueBackendPlan,
  prepareBackendFixture,
  restoreBackendFixture,
  summarizeBackendRows,
} from '../orchestration/benchmark/CommanderBackendBenchmark';
import {
  sealBenchmarkArtifacts,
  verifyBenchmarkArtifacts,
  writeBenchmarkJson,
} from '../orchestration/benchmark/BackendBenchmarkArtifacts';
import { OpenAICommanderError } from '../orchestration/providers/OpenAICommanderClient';

const temporary: string[] = [];
afterEach(() => {
  for (const path of temporary.splice(0)) rmSync(path, { recursive: true, force: true });
});
function temp(): string {
  const path = mkdtempSync(join(tmpdir(), 'snowgym-backend-test-'));
  temporary.push(path);
  return path;
}
const cli = 'snowgym/orchestration/examples/commander-backend-benchmark.ts';

describe('matched commander backend benchmark', () => {
  it('reconstructs exact physical state, plan, and real bounded history across all four cases', () => {
    for (const configuration of BACKEND_PILOT_CASES) {
      const fixture = prepareBackendFixture(configuration);
      expect(prepareBackendFixture(configuration)).toEqual(fixture);
      expect(restoreBackendFixture(fixture).environment.status().stateHash).toBe(
        fixture.request.summary.sourceStateHash,
      );
      expect(fixture.request.triggers).toEqual([]);
      expect(fixture.request.previousPlanOutcome !== undefined).toBe(
        configuration.switchAt !== undefined,
      );
      const before = benchmarkDigest(fixture);
      const first = continueBackendPlan(fixture, fixture.request.currentPlan, 10);
      expect(continueBackendPlan(fixture, fixture.request.currentPlan, 10)).toEqual(first);
      expect(first.metrics.rejectedActions).toBe(0);
      expect(first.actions).toHaveLength(10);
      expect(first.stateHashes).toHaveLength(11);
      expect(benchmarkDigest(fixture)).toBe(before);
    }
  });

  it('rejects tampering before calling a provider, including a re-signed wrong physical prefix', async () => {
    const fixture = prepareBackendFixture(BACKEND_PILOT_CASES[1]);
    const tampered = structuredClone(fixture);
    tampered.observation.allies[0].health--;
    expect(() => restoreBackendFixture(tampered)).toThrow('digest mismatch');
    const { fixtureDigest: _old, ...body } = tampered;
    expect(() => restoreBackendFixture({ ...body, fixtureDigest: benchmarkDigest(body) })).toThrow(
      'state mismatch',
    );
    let calls = 0;
    await expect(
      benchmarkBackendCall(
        {
          plan: async () => {
            calls++;
            throw new Error('should not run');
          },
        },
        backendBenchmarkArms()[0],
        tampered,
      ),
    ).rejects.toThrow();
    expect(calls).toBe(0);
  });

  it('uses every arm once per snapshot with counterbalanced request positions', () => {
    const arms = backendBenchmarkArms();
    const schedule = backendCallSchedule(4, arms);
    expect(schedule).toHaveLength(16);
    for (let position = 0; position < 4; position++) {
      expect(
        new Set(
          schedule
            .filter((_, index) => index % 4 === position)
            .map(({ arm }) => `${arm.backend}-${arm.reasoningEffort}`),
        ).size,
      ).toBe(4);
    }
    expect(() => backendBenchmarkArms(['astra'], ['light', 'low'])).toThrow('duplicate');
    expect(() => backendBenchmarkArms(['astra'], ['high'])).toThrow('pilot reasoning');
    expect(() => backendBenchmarkArms([])).toThrow();
  });

  it('keeps likelihood-free scripted continuation identical for identical plans across backends', async () => {
    const fixture = prepareBackendFixture(BACKEND_PILOT_CASES[0]);
    const rows: Awaited<ReturnType<typeof benchmarkBackendCall>>[] = [];
    for (const arm of backendBenchmarkArms()) {
      rows.push(
        await benchmarkBackendCall(
          {
            plan: async (request) => {
              expect(request).toEqual(fixture.request);
              return {
                decision: request.currentPlan,
                metadata: { model: arm.model, tokensIn: 100, tokensOut: 80, reasoningTokens: 40 },
              };
            },
          },
          arm,
          fixture,
          { horizon: 15 },
        ),
      );
    }
    rows.forEach((row) => {
      expect(row.error).toBeNull();
      expect(row.schemaValid).toBe(true);
      expect(row.continuation).toEqual(rows[0].continuation);
    });
    expect(new Set(rows.map((row) => row.requestDigest)).size).toBe(4);
    const summaries = summarizeBackendRows(rows);
    expect(summaries).toHaveLength(4);
    expect(
      summaries.every(
        (row) => row.accepted === 1 && row.fallback === 0 && row.meanReasoningTokens === 40,
      ),
    ).toBe(true);
  });

  it('separates schema errors, model mismatch, unknown usage, and fallback outcomes', async () => {
    const fixture = prepareBackendFixture(BACKEND_PILOT_CASES[0]);
    const arm = backendBenchmarkArms()[0];
    const invalid = await benchmarkBackendCall(
      { plan: async () => ({ decision: { invalid: true }, metadata: { model: arm.model } }) },
      arm,
      fixture,
      { horizon: 1 },
    );
    expect(invalid.schemaValid).toBe(false);
    expect(invalid.continuation.fallbackUsed).toBe(true);
    const mismatch = await benchmarkBackendCall(
      {
        plan: async () => ({
          decision: fixture.request.currentPlan,
          metadata: { model: 'different-model' },
        }),
      },
      arm,
      fixture,
      { horizon: 1 },
    );
    expect(mismatch.modelMatches).toBe(false);
    expect(mismatch.continuation.fallbackUsed).toBe(true);
    const failure = await benchmarkBackendCall(
      {
        plan: async () => {
          throw new OpenAICommanderError('incomplete', { tokensIn: 100, tokensOut: 4096 });
        },
      },
      arm,
      fixture,
      { horizon: 1 },
    );
    expect(failure.metadata?.tokensOut).toBe(4096);
    const summary = summarizeBackendRows([invalid, mismatch, failure])[0];
    expect(summary.meanAcceptedNetDamage).toBeNull();
    expect(summary.acceptedBlueWins).toBe(0);
    expect(summary.meanInputTokens).toBe(100);
    expect(summary.usageKnownRequests).toBe(1);
    expect(summary.meanReasoningTokens).toBeNull();
  });

  it('enforces a wall-clock timeout with one attempt even if a client ignores abort', async () => {
    let calls = 0;
    const row = await benchmarkBackendCall(
      {
        plan: async () => {
          calls++;
          return new Promise(() => {});
        },
      },
      backendBenchmarkArms()[0],
      prepareBackendFixture(BACKEND_PILOT_CASES[0]),
      { timeoutMs: 5, horizon: 1 },
    );
    expect(row.timedOut).toBe(true);
    expect(row.continuation.fallbackUsed).toBe(true);
    expect(calls).toBe(1);
  });

  it('uses the same fallback when a response is missing or semantically invalid', () => {
    const fixture = prepareBackendFixture(BACKEND_PILOT_CASES[0]);
    const fallback = continueBackendPlan(fixture, undefined, 5);
    const invalid = continueBackendPlan(fixture, { invalid: true }, 5);
    expect(invalid.activationStatus).toBe('rejected');
    expect(invalid.metrics).toEqual(fallback.metrics);
    expect(invalid.stateHashes).toEqual(fallback.stateHashes);
  });

  it('seals artifacts, prevents overwrites, and detects changed files', () => {
    const directory = temp();
    writeBenchmarkJson(directory, 'report.json', { evidence: 1 });
    expect(() => writeBenchmarkJson(directory, 'report.json', {})).toThrow();
    sealBenchmarkArtifacts(directory);
    expect(() => verifyBenchmarkArtifacts(directory)).not.toThrow();
    writeFileSync(join(directory, 'report.json'), '{}');
    expect(() => verifyBenchmarkArtifacts(directory)).toThrow('artifact digest mismatch');
  });

  it('runs an entirely credential-free CLI dry run and fails closed on duplicate output and request budget', () => {
    const output = join(temp(), 'run');
    const env = { ...process.env, OPENAI_API_KEY: '' };
    execFileSync(
      process.execPath,
      ['--import', 'tsx', cli, '--dry-run', '--output', output, '--horizon', '1'],
      { env },
    );
    expect(JSON.parse(readFileSync(join(output, 'report.json'), 'utf8')).attempts).toBe(0);
    verifyBenchmarkArtifacts(output);
    const duplicate = spawnSync(process.execPath, ['--import', 'tsx', cli, '--output', output], {
      env,
      encoding: 'utf8',
    });
    expect(duplicate.status).not.toBe(0);
    expect(duplicate.stderr).toContain('refusing to overwrite');
    const budget = spawnSync(
      process.execPath,
      ['--import', 'tsx', cli, '--max-requests', '1', '--output', join(temp(), 'run')],
      { env, encoding: 'utf8' },
    );
    expect(budget.status).not.toBe(0);
    expect(budget.stderr).toContain('no calls made');
  }, 20000);
});
