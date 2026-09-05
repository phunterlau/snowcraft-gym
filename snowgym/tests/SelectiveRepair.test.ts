import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { benchmarkDigest } from '../orchestration/benchmark/CommanderBackendBenchmark';
import { ReactiveUnitPolicy } from '../orchestration/execution/ReactiveUnitPolicy';
import type { UnitPolicyContext } from '../orchestration/execution/UnitPolicy';
import { TargetResolver } from '../orchestration/grounding/TargetResolver';
import { BindingRefreshResolver, LocalFirePolicy } from '../orchestration/recovery/SelectiveRepair';
import {
  restoreRecoveryFixture,
  type RecoveryFixture,
} from '../orchestration/recovery/RecoveryBenchmark';
import {
  runMechanismBranch,
  MECHANISM_ARMS,
  MECHANISM_CASES,
  MECHANISM_DELAYS,
  MECHANISM_VERSION,
  firstChangedAction,
} from '../orchestration/recovery/MechanismAudit';
import {
  auditSeedDocuments,
  parseSeedJson,
  pairedMechanismReport,
  writeMechanismArtifact,
  verifyMechanismArtifacts,
  sealMechanismArtifacts,
  readMechanismArtifact,
  verifyRegisteredMechanism,
  type MechanismRow,
} from '../orchestration/recovery/MechanismArtifacts';

const historical = 'snowgym/orchestration/recovery/examples/preflight-20260905-v0';
const fixtures = JSON.parse(
  readFileSync(`${historical}/fixtures.json`, 'utf8'),
) as RecoveryFixture[];
function context(): UnitPolicyContext {
  const f = fixtures[0];
  const observation = structuredClone(f.base.observation);
  const self = {
    ...observation.allies[0],
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    state: 'idle' as const,
    throwCooldown: 0,
  };
  const far = { ...observation.enemies[0], id: 100, x: 12, y: 0, vx: 0, vy: 0 };
  const near = { ...far, id: 101, x: 5 };
  observation.allies = [self];
  observation.enemies = [far, near];
  observation.projectiles = [];
  const group = f.base.snapshot.plan.groups[0];
  return {
    self,
    observation,
    group: {
      role: 'main',
      command: group.command,
      memberIds: [self.id],
      livingMemberIds: [self.id],
      centroid: { x: 0, y: 0 },
      objective: group.objective,
      candidateEnemyIds: [100],
      focusTargetId: 100,
    },
  };
}

describe('isolated local firing', () => {
  it('uses the production throw geometry while preserving the input', () => {
    const c = context();
    const before = benchmarkDigest(c);
    const policy = new LocalFirePolicy();
    const action = policy.act(c);
    expect(action.type).toBe('throw');
    expect(action).toEqual(
      new ReactiveUnitPolicy().act({
        ...c,
        group: { ...c.group, candidateEnemyIds: [101], focusTargetId: 101 },
      }),
    );
    expect(benchmarkDigest(c)).toBe(before);
    expect(policy.shots.map((row) => row.targetId)).toEqual([101]);
    expect(new ReactiveUnitPolicy().act(c).type).toBe('move');
  });

  it('preserves readiness, living-unit checks and dodge priority', () => {
    for (const state of ['hit', 'throwing', 'recovering', 'frozen'] as const) {
      const c = context();
      Object.assign(c.self, { state });
      const p = new LocalFirePolicy();
      expect(p.act(c)).toEqual(new ReactiveUnitPolicy().act(c));
      expect(p.shots).toEqual([]);
    }
    for (const override of [{ throwCooldown: 0.01 }, { alive: false }]) {
      const c = context();
      Object.assign(c.self, override);
      const p = new LocalFirePolicy();
      expect(p.act(c)).toEqual(new ReactiveUnitPolicy().act(c));
      expect(p.shots).toEqual([]);
    }
    const c = context();
    c.observation.projectiles = [
      {
        id: 500,
        ownerId: 101,
        team: 'red',
        x: 2,
        y: 0,
        vx: -1,
        vy: 0,
        height: 1,
        heightVelocity: 0,
      },
    ];
    const p = new LocalFirePolicy();
    expect(p.act(c)).toEqual(new ReactiveUnitPolicy().act(c));
    expect(p.shots).toEqual([]);
  });

  it('uses inclusive range, skips dead enemies and resolves equal distance by ID', () => {
    const c = context();
    c.observation.enemies[1].x = 9;
    expect(new LocalFirePolicy().act(c).type).toBe('throw');
    c.observation.enemies[1].x = 9 + 1e-8;
    expect(new LocalFirePolicy().act(c)).toEqual(new ReactiveUnitPolicy().act(c));
    c.observation.enemies[1].x = 5;
    c.observation.enemies.push({ ...c.observation.enemies[1], id: 99, x: -5 });
    const p = new LocalFirePolicy();
    p.act(c);
    expect(p.shots[0].targetId).toBe(99);
    c.observation.enemies[1].alive = false;
    c.observation.enemies[2].alive = false;
    expect(new LocalFirePolicy().act(c)).toEqual(new ReactiveUnitPolicy().act(c));
  });

  it('does not bypass normal target selection when any assigned candidate is in range', () => {
    const c = context();
    c.observation.enemies.push({ ...c.observation.enemies[1], id: 102, x: 6 });
    Object.assign(c.group, { candidateEnemyIds: [100, 102] });
    const p = new LocalFirePolicy();
    expect(p.act(c)).toEqual(new ReactiveUnitPolicy().act(c));
    expect(p.shots).toEqual([]);
  });
});

describe('selective-repair mechanism isolation', () => {
  it('refreshes bindings without changing plan metadata and tracks surviving replacements', () => {
    const world = restoreRecoveryFixture(fixtures[0]);
    const snapshot = world.store.current();
    const before = benchmarkDigest(snapshot);
    const resolver = new BindingRefreshResolver(snapshot, world.observation);
    const objective = snapshot.plan.groups[0].objective;
    const resolved = resolver.refresh(objective, world.observation);
    expect(resolved.kind === 'enemy_cluster' && resolved.enemyIds).toEqual([6, 7]);
    const after = structuredClone(world.observation);
    after.enemies.find((unit) => unit.id === 6)!.alive = false;
    after.enemies.find((unit) => unit.id === 7)!.x = 42;
    expect(resolver.refresh(objective, after).anchor.x).toBe(42);
    after.enemies.find((unit) => unit.id === 7)!.alive = false;
    const fallback = resolver.refresh(objective, after);
    expect(
      fallback.kind === 'enemy_cluster' && fallback.enemyIds.some((id) => id === 6 || id === 7),
    ).toBe(false);
    const region = {
      kind: 'region' as const,
      region: 'center_lane' as const,
      anchor: { x: 4, y: 5 },
    };
    expect(resolver.refresh(region, after)).toBe(region);
    const current = { kind: 'current_position' as const, anchor: { x: 4, y: 5 } };
    expect(resolver.refresh(current, after)).toBe(current);
    const support = {
      kind: 'ally_group' as const,
      role: 'main' as const,
      anchor: { x: 4, y: 5 },
      unitIds: snapshot.plan.groups[0].assignment.unitIds,
    };
    const assignments = snapshot.plan.groups.map((group) => group.assignment);
    expect(resolver.refresh(support, after, assignments)).toEqual(
      new TargetResolver().refresh(support, after, assignments),
    );
    expect(benchmarkDigest(snapshot)).toBe(before);
    expect(resolver.revision).toBe(1);
  });

  it('preserves every historical keep/reactivation action and hash across all delays', () => {
    for (const [index, fixture] of fixtures.entries())
      for (const delay of MECHANISM_DELAYS)
        for (const arm of ['keep', 'reactivate'] as const) {
          const result = runMechanismBranch(fixture, arm, delay);
          const old = JSON.parse(
            readFileSync(
              `${historical}/continuation-${index}-${arm === 'keep' ? 'keep' : 'reactivate_current'}-${delay}.json`,
              'utf8',
            ),
          );
          expect(result.actions).toEqual(old.actions);
          expect(result.stateHashes).toEqual(old.stateHashes);
          expect(result.metrics.winner).toBe(old.metrics.winner);
          expect(result.metrics.damageDealt).toBe(old.metrics.damageDealt);
          expect(result.metrics.damageReceived).toBe(old.metrics.damageReceived);
        }
  }, 20000);

  it('isolates metadata and scoring, shares delay prefixes, and reruns every new arm exactly', () => {
    const fixture = fixtures[0];
    const digest = benchmarkDigest(fixture);
    for (const delay of [0, 20, 80]) {
      const baseline = runMechanismBranch(fixture, 'keep', delay);
      for (const arm of MECHANISM_ARMS) {
        const result = runMechanismBranch(fixture, arm, delay);
        expect(runMechanismBranch(fixture, arm, delay)).toEqual(result);
        expect(result.actions.slice(0, delay)).toEqual(baseline.actions.slice(0, delay));
        expect(result.stateHashes.slice(0, delay + 1)).toEqual(
          baseline.stateHashes.slice(0, delay + 1),
        );
        expect(result.initialMeasure.frozenTargets).toEqual(baseline.initialMeasure.frozenTargets);
        expect(result.metrics.rejectedActions).toBe(0);
        if (delay === 80) {
          expect(result.interventionStatus).toBe('not_reached');
          expect(result.interventionTick).toBeNull();
          expect(result.actions).toEqual(baseline.actions);
        } else if (arm !== 'reactivate')
          expect(result.postInterventionSnapshot).toEqual(result.sourceSnapshot);
        if (arm === 'local_fire' && delay === 0)
          expect(result.metrics.opportunisticShots).toBeGreaterThan(0);
        if (arm === 'keep') expect(firstChangedAction(result, baseline)).toBeNull();
      }
    }
    expect(benchmarkDigest(fixture)).toBe(digest);
    const tampered = structuredClone(fixture);
    tampered.evidence.groups[0].living = 99;
    expect(() => runMechanismBranch(tampered, 'keep', 0)).toThrow('digest mismatch');
    const { digest: _old, ...body } = tampered;
    expect(() => runMechanismBranch({ ...body, digest: benchmarkDigest(body) }, 'keep', 0)).toThrow(
      'evidence mismatch',
    );
    expect(() => runMechanismBranch(fixture, 'keep', 1)).toThrow('invalid');
  }, 20000);
});

describe('mechanism audit lineage and analysis', () => {
  it('detects scalar seeds, nested partition ranges and enclosing schedules without matching hashes', () => {
    const documents = [
      {
        path: 'manifest',
        value: {
          seed: 630000,
          hash: 'sha256:630001',
          seedSchedule: { minimum: 600000, maximum: 700000, nextSeed: 600001 },
          partitions: { qualification: [630050, 630060] },
        },
      },
    ];
    expect(auditSeedDocuments(documents).collisions).toHaveLength(3);
    expect(
      auditSeedDocuments([{ path: 'clear', value: { seeds: [100000, 107999], hash: '630001' } }])
        .collisions,
    ).toEqual([]);
    const parsed = parseSeedJson(
      '{"distance": Infinity, "text":"Infinity", "x":-Infinity,"y":NaN,"seed":7}',
    );
    expect(parsed.nonFiniteValues).toBe(3);
    expect((parsed.value as { text: string }).text).toBe('Infinity');
    expect(auditSeedDocuments([{ path: 'python', value: parsed.value }]).collisions).toEqual([]);
    expect(() =>
      auditSeedDocuments([{ path: 'bad', value: parseSeedJson('{"seed":NaN}').value }]),
    ).toThrow('non-finite seed');
  });

  it('bootstraps paired seeds reproducibly, excludes historical data and rejects duplicate pairs', () => {
    const rows: MechanismRow[] = [1, 2, 3].flatMap((seed) =>
      MECHANISM_ARMS.map((arm) => ({
        cohort: 'fresh',
        seed,
        roster: '5v5',
        arm,
        delayDecisions: 0,
        filename: 'test.json',
        firstChangedAction: null,
        interventionStatus: 'applied',
        actualBindingChanges: 0,
        metrics: {
          winner: arm === 'keep' ? 'red' : 'blue',
          damageDealt: 10,
          damageReceived: 0,
          blueAlive: 1,
          rejectedActions: 0,
          censored: false,
          decisions: 10,
        },
      })),
    );
    const result = pairedMechanismReport(rows, 1000);
    expect(result).toEqual(pairedMechanismReport([...rows].reverse(), 1000));
    expect(result.comparisons[0].metrics.blueWinRateDifference).toEqual({
      mean: 1,
      lower95: 1,
      upper95: 1,
    });
    expect(result.comparisons[0].underCovered).toBe(true);
    expect(pairedMechanismReport([...rows, { ...rows[0], cohort: 'historical' }], 1000)).toEqual(
      result,
    );
    expect(() => pairedMechanismReport([...rows, rows[0]], 100)).toThrow('duplicate');
  });

  it('seals compressed traces, refuses overwrite and detects tampering', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mechanism-artifact-test-'));
    try {
      writeMechanismArtifact(dir, 'trace.json.gz', { actions: [1, 2] });
      expect(readMechanismArtifact(dir, 'trace.json.gz')).toEqual({ actions: [1, 2] });
      expect(() => writeMechanismArtifact(dir, 'trace.json.gz', {})).toThrow();
      sealMechanismArtifacts(dir);
      verifyMechanismArtifacts(dir);
      writeFileSync(join(dir, 'trace.json.gz'), '{}');
      expect(() => verifyMechanismArtifacts(dir)).toThrow('digest mismatch');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('recognizes only sealed runs with the exact registered experiment declaration', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mechanism-registered-test-'));
    try {
      writeMechanismArtifact(dir, 'configuration.json', {
        format: MECHANISM_VERSION,
        cases: MECHANISM_CASES,
        arms: MECHANISM_ARMS,
        delays: MECHANISM_DELAYS,
        horizon: 300,
        scanHorizon: 300,
        mode: 'full',
      });
      expect(() => verifyRegisteredMechanism(dir)).toThrow();
      sealMechanismArtifacts(dir);
      expect(verifyRegisteredMechanism(dir)).toMatch(/^sha256:[a-f0-9]{64}$/);
      writeFileSync(
        join(dir, 'configuration.json'),
        JSON.stringify({ format: MECHANISM_VERSION, cases: [] }),
      );
      expect(() => verifyRegisteredMechanism(dir)).toThrow('digest mismatch');
      rmSync(join(dir, 'manifest.json'));
      sealMechanismArtifacts(dir);
      expect(() => verifyRegisteredMechanism(dir)).toThrow('declaration mismatch');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('runs the CLI preflight without credentials or new episodes and verifies it', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mechanism-cli-test-'));
    const path = join(dir, 'run');
    const cli = 'snowgym/orchestration/examples/selective-repair-audit.ts';
    const env = { ...process.env, OPENAI_API_KEY: '' };
    try {
      execFileSync(process.execPath, ['--import', 'tsx', cli, '--preflight', '--output', path], {
        env,
      });
      expect(readMechanismArtifact(path, 'fixtures.json.gz')).toEqual([]);
      expect(
        readMechanismArtifact<{ providerCalls: number }>(path, 'report.json').providerCalls,
      ).toBe(0);
      execFileSync(process.execPath, ['--import', 'tsx', cli, '--verify', path], { env });
      const duplicate = spawnSync(process.execPath, ['--import', 'tsx', cli, '--output', path], {
        env,
        encoding: 'utf8',
      });
      expect(duplicate.stderr).toContain('refusing to overwrite');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 20000);
});
