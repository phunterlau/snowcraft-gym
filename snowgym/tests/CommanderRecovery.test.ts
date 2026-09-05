import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BACKEND_PILOT_CASES,
  benchmarkDigest,
  prepareBackendFixture,
} from '../orchestration/benchmark/CommanderBackendBenchmark';
import { verifyBenchmarkArtifacts } from '../orchestration/benchmark/BackendBenchmarkArtifacts';
import { openAIRequestBody } from '../orchestration/providers/OpenAICommanderClient';
import { summarizeRecovery, segmentObstructed } from '../orchestration/recovery/RecoveryEvidence';
import {
  collectRecoveryFixtures,
  continueRecoveryPlan,
  recoveryRequest,
  restoreRecoveryFixture,
  RECOVERY_SCAN_CASES,
} from '../orchestration/recovery/RecoveryBenchmark';

const initial = () => prepareBackendFixture(BACKEND_PILOT_CASES[0]);
const scan = () => collectRecoveryFixtures([RECOVERY_SCAN_CASES[1]], 300);

describe('commander recovery evidence', () => {
  it('is detached, deterministic, bounded and ID-free with explicit proxy semantics', () => {
    const fixture = initial();
    const before = benchmarkDigest(fixture);
    const result = summarizeRecovery(fixture.observation, fixture.observation, fixture.snapshot);
    expect(summarizeRecovery(fixture.observation, fixture.observation, fixture.snapshot)).toEqual(
      result,
    );
    expect(benchmarkDigest(fixture)).toBe(before);
    expect(result.groups).toHaveLength(1);
    expect(result.groups[0].healthFractionOfActivation).toBe(1);
    expect(result.groups[0].recent).toBeNull();
    expect(result.detectedFamilies).toEqual([]);
    expect(result.semantics.capability).toContain('uncalibrated');
    expect(JSON.stringify(result)).not.toMatch(/"(?:unitId|enemyIds|unitIds|x|y|projectiles)"/);
    result.groups[0].living = 0;
    expect(fixture.observation.match.blueAlive).toBe(5);
  });

  it('retains the activation health denominator through casualties and separates readiness from range', () => {
    const fixture = initial();
    const observation = structuredClone(fixture.observation);
    observation.allies[0].alive = false;
    observation.allies[0].health = 0;
    observation.allies[1].state = 'hit';
    const result = summarizeRecovery(observation, fixture.observation, fixture.snapshot).groups[0];
    expect(result.healthFractionOfActivation).toBe(0.8);
    expect(result.livingFraction).toBe(0.8);
    expect(result.canThrowFraction).toBe(0.75);
    expect(result.inExecutorRangeFraction).toBe(0);
    observation.allies.forEach((unit) => {
      unit.alive = false;
      unit.health = 0;
    });
    const empty = summarizeRecovery(observation, fixture.observation, fixture.snapshot).groups[0];
    expect(empty.objectiveDistance).toBeNull();
    expect(empty.canThrowFraction).toBeNull();
    expect(empty.frozenTargetHealthFraction).toBe(1);
    expect(JSON.stringify(empty)).not.toMatch(/NaN|Infinity/);
  });

  it('scores the frozen target independently of tactical replacement', () => {
    const fixture = initial();
    const snapshot = structuredClone(fixture.snapshot);
    const target = snapshot.plan.groups[0].objective;
    if (target.kind !== 'enemy_cluster') throw new Error('test requires enemy objective');
    Object.assign(target, { enemyIds: [fixture.observation.enemies[0].id] });
    const observation = structuredClone(fixture.observation);
    observation.enemies[0].alive = false;
    observation.enemies[0].health = 0;
    const result = summarizeRecovery(observation, fixture.observation, snapshot);
    expect(result.groups[0].frozenTargetEliminated).toBe(true);
    expect(result.groups[0].frozenTargetHealthFraction).toBe(0);
    expect(result.groups[0].meanRangeExcess).not.toBeNull();
    expect(result.detectedFamilies).toContain('target_eliminated');
  });

  it('uses obstruction flags and handles axis-parallel, tangential and zero-length segments', () => {
    const observation = initial().observation;
    observation.obstacles = [
      {
        id: 1,
        type: 'wall' as never,
        x: 0,
        y: 0,
        halfWidth: 1,
        halfHeight: 1,
        blocksMovement: true,
        blocksProjectiles: false,
        blocksSight: true,
      },
    ];
    expect(segmentObstructed({ x: -2, y: 0 }, { x: 2, y: 0 }, observation, 'blocksMovement')).toBe(
      true,
    );
    expect(segmentObstructed({ x: -2, y: 1 }, { x: 2, y: 1 }, observation, 'blocksMovement')).toBe(
      true,
    );
    expect(segmentObstructed({ x: -2, y: 2 }, { x: 2, y: 2 }, observation, 'blocksMovement')).toBe(
      false,
    );
    expect(segmentObstructed({ x: 0, y: 0 }, { x: 0, y: 0 }, observation, 'blocksMovement')).toBe(
      true,
    );
    expect(segmentObstructed({ x: 2, y: 2 }, { x: 2, y: 2 }, observation, 'blocksMovement')).toBe(
      false,
    );
    expect(
      segmentObstructed({ x: -2, y: 0 }, { x: 2, y: 0 }, observation, 'blocksProjectiles'),
    ).toBe(false);
  });

  it('requires measured windows and separates accepted throws from rejected attempts', () => {
    const fixture = prepareBackendFixture({ ...BACKEND_PILOT_CASES[1], prefixDecisions: 20 });
    const activation = prepareBackendFixture({
      ...fixture.configuration,
      prefixDecisions: 0,
    }).observation;
    const digest = structuredClone(fixture.request.trajectory!);
    Object.assign(digest.groups[0], {
      livingStart: 5,
      livingEnd: 4,
      enemyHealthDelta: 0,
      issuedActions: { noop: 0, move: 0, hold: 0, throw: 5 },
      rejectedActions: { noop: 0, move: 0, hold: 0, throw: 5 },
    });
    expect(
      summarizeRecovery(fixture.observation, activation, fixture.snapshot, digest).detectedFamilies,
    ).toEqual(['recent_casualties']);
    Object.assign(digest.groups[0].rejectedActions, { throw: 0 });
    expect(
      summarizeRecovery(fixture.observation, activation, fixture.snapshot, digest).detectedFamilies,
    ).toContain('throws_without_damage');
    Object.assign(digest, { decisions: 19 });
    expect(
      summarizeRecovery(fixture.observation, activation, fixture.snapshot, digest).detectedFamilies,
    ).toEqual([]);
    Object.assign(digest, { endTick: digest.endTick - 6 });
    expect(() =>
      summarizeRecovery(fixture.observation, activation, fixture.snapshot, digest),
    ).toThrow('interval');
    expect(() =>
      summarizeRecovery(fixture.observation, fixture.observation, fixture.snapshot),
    ).toThrow('activation');
  });

  it('adds evidence only to input and preserves instructions, model config and output schema', () => {
    const fixture = initial();
    const original = openAIRequestBody(fixture.request);
    const enriched = openAIRequestBody({
      ...fixture.request,
      recoveryEvidence: summarizeRecovery(
        fixture.observation,
        fixture.observation,
        fixture.snapshot,
      ),
    });
    const { input: oldInput, ...oldRest } = original;
    const { input: newInput, ...newRest } = enriched;
    expect(newRest).toEqual(oldRest);
    const parse = (input: unknown) =>
      JSON.parse((input as { content: { text: string }[] }[])[0].content[0].text);
    const { recoveryEvidence, ...unchanged } = parse(newInput);
    expect(recoveryEvidence.schemaVersion).toBe('snowgym.recovery-evidence.v0');
    expect(unchanged).toEqual(parse(oldInput));
  });

  it('preserves all archived pilot request bodies exactly', () => {
    const path = 'snowgym/orchestration/benchmark/examples/luna-astra-20260905-v0/preflight';
    verifyBenchmarkArtifacts(path);
    const fixtures = JSON.parse(readFileSync(`${path}/fixtures.json`, 'utf8'));
    const requests = JSON.parse(readFileSync(`${path}/requests.json`, 'utf8'));
    for (const row of requests) {
      const fixture = fixtures.find(
        (item: { configuration: { id: string } }) => item.configuration.id === row.caseId,
      );
      expect(openAIRequestBody(fixture.request, row.reasoningEffort, 4096, row.model)).toEqual(
        row.body,
      );
    }
  });
});

describe('commander recovery reconstruction and delayed control', () => {
  it('collects observed opportunities deterministically and reports missing families honestly', () => {
    const collection = scan();
    expect(collection).toEqual(scan());
    expect(collection.fixtures.length).toBeGreaterThan(0);
    expect(collection.missingFamilies).toContain('blocked_advance');
    for (const fixture of collection.fixtures) {
      const before = benchmarkDigest(fixture);
      expect(restoreRecoveryFixture(fixture).observation).toEqual(fixture.base.observation);
      expect(recoveryRequest(fixture, false)).toEqual(fixture.base.request);
      expect(recoveryRequest(fixture, true).recoveryEvidence).toEqual(fixture.evidence);
      expect(benchmarkDigest(fixture)).toBe(before);
    }
  });

  it('rejects tampered evidence even when the outer digest is recomputed', () => {
    const fixture = scan().fixtures[0];
    fixture.evidence.groups[0].healthFractionOfActivation = 99;
    expect(() => restoreRecoveryFixture(fixture)).toThrow('digest mismatch');
    const { digest: _digest, ...body } = fixture;
    expect(() => restoreRecoveryFixture({ ...body, digest: benchmarkDigest(body) })).toThrow(
      'evidence mismatch',
    );
  });

  it('shares exact prefixes during delay, uses a fixed total horizon, and keeps no-change anchors', () => {
    const fixture = scan().fixtures[0];
    const baseline = continueRecoveryPlan(fixture, null, 0, 100);
    const delayed = continueRecoveryPlan(fixture, fixture.base.request.currentPlan, 20, 100);
    expect(delayed.stateHashes.slice(0, 21)).toEqual(baseline.stateHashes.slice(0, 21));
    expect(delayed.actions.slice(0, 20)).toEqual(baseline.actions.slice(0, 20));
    expect(delayed.sourceAgeAtActivationSeconds).toBe(2);
    expect(delayed.planReactivated).toBe(true);
    expect(baseline.planReactivated).toBe(false);
    expect(baseline.activationStatus).toBe('kept');
    expect(delayed.actions.length).toBeLessThanOrEqual(100);
    expect(delayed).toEqual(
      continueRecoveryPlan(fixture, fixture.base.request.currentPlan, 20, 100),
    );
    expect(delayed.metrics.rejectedActions).toBe(0);
    expect(delayed.stateHashes.length).toBe(delayed.actions.length + 1);
    const invalid = continueRecoveryPlan(fixture, { invalid: true }, 0, 10);
    expect(invalid.activationStatus).toBe('rejected');
    expect(invalid.activationError).toBeTruthy();
    expect(() => continueRecoveryPlan(fixture, null, 80, 80)).toThrow('shorter');
    expect(() => continueRecoveryPlan(fixture, null, -1)).toThrow('integer');
  });

  it('runs credential-free, seals outputs, refuses overwrites and rejects invalid budgets', () => {
    const directory = mkdtempSync(join(tmpdir(), 'snowgym-recovery-test-'));
    const output = join(directory, 'run');
    const cli = 'snowgym/orchestration/examples/commander-recovery-benchmark.ts';
    const env = { ...process.env, OPENAI_API_KEY: '' };
    try {
      execFileSync(
        process.execPath,
        ['--import', 'tsx', cli, '--output', output, '--scan-horizon', '20', '--horizon', '81'],
        { env },
      );
      verifyBenchmarkArtifacts(output);
      const report = JSON.parse(readFileSync(join(output, 'report.json'), 'utf8'));
      expect(report.providerAttempts).toBe(0);
      expect(report.providerReady).toBe(false);
      const duplicate = spawnSync(process.execPath, ['--import', 'tsx', cli, '--output', output], {
        env,
        encoding: 'utf8',
      });
      expect(duplicate.status).not.toBe(0);
      expect(duplicate.stderr).toContain('refusing to overwrite');
      const invalid = spawnSync(
        process.execPath,
        ['--import', 'tsx', cli, '--output', join(directory, 'invalid'), '--horizon', 'NaN'],
        { env, encoding: 'utf8' },
      );
      expect(invalid.status).not.toBe(0);
      expect(invalid.stderr).toContain('invalid horizon');
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }, 20000);
});
