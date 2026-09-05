import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { fileDigest, verifyBenchmarkArtifacts } from '../benchmark/BackendBenchmarkArtifacts';
import { benchmarkDigest } from '../benchmark/CommanderBackendBenchmark';
import {
  collectRecoveryFixtures,
  restoreRecoveryFixture,
  type RecoveryFixture,
} from '../recovery/RecoveryBenchmark';
import {
  MECHANISM_ARMS,
  MECHANISM_CASES,
  MECHANISM_DELAYS,
  MECHANISM_VERSION,
  firstChangedAction,
  runMechanismBranch,
  type MechanismResult,
} from '../recovery/MechanismAudit';
import {
  auditSeedDocuments,
  parseSeedJson,
  pairedMechanismReport,
  readMechanismArtifact,
  sealMechanismArtifacts,
  verifyMechanismArtifacts,
  verifyRegisteredMechanism,
  writeMechanismArtifact,
  type MechanismRow,
} from '../recovery/MechanismArtifacts';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    verify: { type: 'string' },
    preflight: { type: 'boolean' },
    'historical-only': { type: 'boolean' },
    help: { type: 'boolean', short: 'h' },
  },
  strict: true,
});
if (values.help) {
  console.log(`Selective repair mechanism audit; no providers, credentials, server or browser.
node --import tsx snowgym/orchestration/examples/selective-repair-audit.ts --output NEW_DIR
--preflight          Audit seed metadata and freeze configuration; collect no new episodes
--historical-only    Run the three known fixtures only (not fresh-cohort evidence)
--verify DIRECTORY   Verify immutable inventory, digests and reconstruct fixture semantics
Default: 120 declared seeds, first casualty within 300 decisions, four arms,
0/1/2/4/8-second delays, 300-decision shared horizon, two executions per branch.
Output must not exist. Collision/contract failure stops without resampling.`);
  process.exit(0);
}
if (values.verify) {
  const path = resolve(values.verify);
  verifyMechanismArtifacts(path);
  const fixtures = readMechanismArtifact(path, 'fixtures.json.gz') as RecoveryFixture[];
  fixtures.forEach(restoreRecoveryFixture);
  console.log('Mechanism artifact digests and fixture reconstruction verified.');
  process.exit(0);
}
if (!values.output) throw new Error('--output NEW_DIR is required');
if (values.preflight && values['historical-only']) throw new Error('select one audit mode');
const output = resolve(values.output);
if (existsSync(output)) throw new Error(`refusing to overwrite ${output}`);
const repository = fileURLToPath(new URL('../../../', import.meta.url));
const historicalPath = resolve(
  repository,
  'snowgym/orchestration/recovery/examples/preflight-20260905-v0',
);
verifyBenchmarkArtifacts(historicalPath);
const historical: RecoveryFixture[] = JSON.parse(
  readFileSync(resolve(historicalPath, 'fixtures.json'), 'utf8'),
);
historical.forEach(restoreRecoveryFixture);
if (historical.length !== 3 || historical.some((f) => f.family !== 'recent_casualties'))
  throw new Error('historical fixture declaration mismatch');
// Include ignored local run metadata too; rg follows the repository's normal inventory rules.
const paths = execFileSync(
  'rg',
  [
    '--files',
    '--hidden',
    '--no-ignore',
    'snowgym',
    'refs',
    '-g',
    '*.json',
    '-g',
    '!**/.venv/**',
    '-g',
    '!**/node_modules/**',
    '-g',
    '!**/__pycache__/**',
  ],
  { cwd: repository, encoding: 'utf8', maxBuffer: 20_000_000 },
)
  .trim()
  .split('\n')
  .filter(Boolean)
  .sort();
const seedSources = Object.fromEntries(
  paths.map((path) => [path, fileDigest(resolve(repository, path))]),
);
const parsedSeedDocuments = paths.map((path) => ({
  path,
  ...parseSeedJson(readFileSync(resolve(repository, path), 'utf8')),
}));
// Recognize only sealed runs of this exact declaration. Keep their hashes in the
// inventory and explicitly report reuse; other experiments still fail on collision.
const registeredRuns = parsedSeedDocuments
  .filter(
    (row) =>
      row.path.startsWith('snowgym/orchestration/recovery/examples/') &&
      row.path.endsWith('/configuration.json') &&
      (row.value as { format?: string } | null)?.format === MECHANISM_VERSION,
  )
  .map((row) => ({
    directory: dirname(row.path),
    manifestDigest: verifyRegisteredMechanism(resolve(repository, dirname(row.path))),
  }));
const registeredPaths = new Set(registeredRuns.map((row) => row.directory));
const audit = auditSeedDocuments(
  parsedSeedDocuments.filter((row) => !registeredPaths.has(dirname(row.path))),
);
if (audit.collisions.length)
  throw new Error(`seed preflight collision; no collection: ${JSON.stringify(audit.collisions)}`);
const sourcePaths = execFileSync(
  'git',
  [
    'ls-files',
    '--cached',
    '--others',
    '--exclude-standard',
    '-z',
    '--',
    'src',
    'snowgym',
    'public/maps',
  ],
  { cwd: repository, encoding: 'utf8' },
)
  .split('\0')
  .filter(
    (path) =>
      path.endsWith('.ts') ||
      path.endsWith('.schema.json') ||
      (path.startsWith('public/maps/') && path.endsWith('.json')),
  );
sourcePaths.push(
  'package.json',
  'package-lock.json',
  'tsconfig.json',
  'snowgym/PLAN.md',
  'refs/snowgym_selective_repair_review_and_plan.md',
);
const sources = Object.fromEntries(
  [...new Set(sourcePaths)].sort().map((path) => [path, fileDigest(resolve(repository, path))]),
);
mkdirSync(output, { recursive: true });
writeMechanismArtifact(output, 'configuration.json', {
  format: MECHANISM_VERSION,
  mode: values.preflight ? 'preflight' : values['historical-only'] ? 'historical-only' : 'full',
  sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: repository,
    encoding: 'utf8',
  }).trim(),
  sources,
  sourceDigest: benchmarkDigest(sources),
  historicalManifestDigest: fileDigest(resolve(historicalPath, 'manifest.json')),
  cases: MECHANISM_CASES,
  arms: MECHANISM_ARMS,
  delays: MECHANISM_DELAYS,
  horizon: 300,
  scanHorizon: 300,
  analysisRng: 730001,
  bootstrapResamples: 10000,
  autonomousQualificationEligible: false,
  providerCalls: 0,
});
writeMechanismArtifact(output, 'seed-audit.json', {
  ...audit,
  seedSources,
  registeredSameExperimentRuns: registeredRuns,
  nonFiniteMetadata: parsedSeedDocuments
    .filter((row) => row.nonFiniteValues)
    .map((row) => ({ path: row.path, count: row.nonFiniteValues })),
  limitation:
    'Checks recorded JSON seed fields, allocations and schedules, including local run metadata. Sealed runs of this exact declaration are reported as same-experiment reuse; unrecorded external runs and standalone compressed metadata cannot be audited. Two-number seed lists are conservatively inclusive ranges.',
});
const fixtures = values.preflight ? [] : [...historical];
const scans: {
  configuration: (typeof MECHANISM_CASES)[number];
  scan: ReturnType<typeof collectRecoveryFixtures>['scans'][number];
  qualified: boolean;
}[] = [];
if (!values.preflight && !values['historical-only'])
  for (const [i, configuration] of MECHANISM_CASES.entries()) {
    const collection = collectRecoveryFixtures([configuration], 300);
    const fixture = collection.fixtures.find((f) => f.family === 'recent_casualties');
    if (fixture) fixtures.push(fixture);
    scans.push({ configuration, scan: collection.scans[0], qualified: !!fixture });
    console.log(
      `Scan ${i + 1}/120: ${configuration.id} ${fixture ? 'qualified' : 'no casualty opportunity'}`,
    );
  }
writeMechanismArtifact(output, 'scan.json', scans);
writeMechanismArtifact(output, 'fixtures.json.gz', fixtures);
const rows: MechanismRow[] = [];
for (const [index, fixture] of fixtures.entries()) {
  console.log(`Branch ${index + 1}/${fixtures.length}: ${fixture.base.configuration.id}`);
  let zeroKeep: MechanismResult | undefined;
  for (const delay of MECHANISM_DELAYS) {
    let baseline: MechanismResult | undefined;
    for (const arm of MECHANISM_ARMS) {
      const result = runMechanismBranch(fixture, arm, delay);
      const duplicate = runMechanismBranch(fixture, arm, delay);
      if (benchmarkDigest(result) !== benchmarkDigest(duplicate))
        throw new Error('independent continuation mismatch');
      if (arm === 'keep') {
        baseline = result;
        if (!zeroKeep) zeroKeep = result;
        if (
          benchmarkDigest(result.actions) !== benchmarkDigest(zeroKeep.actions) ||
          benchmarkDigest(result.stateHashes) !== benchmarkDigest(zeroKeep.stateHashes)
        )
          throw new Error('keep trajectory changed with delay');
      }
      if (!baseline) throw new Error('baseline missing');
      if (
        benchmarkDigest(result.actions.slice(0, delay)) !==
          benchmarkDigest(baseline.actions.slice(0, delay)) ||
        benchmarkDigest(result.stateHashes.slice(0, delay + 1)) !==
          benchmarkDigest(baseline.stateHashes.slice(0, delay + 1))
      )
        throw new Error('delay prefix mismatch');
      if (index < historical.length && (arm === 'keep' || arm === 'reactivate')) {
        const old = JSON.parse(
          readFileSync(
            resolve(
              historicalPath,
              `continuation-${index}-${arm === 'keep' ? 'keep' : 'reactivate_current'}-${delay}.json`,
            ),
            'utf8',
          ),
        );
        if (
          benchmarkDigest(result.actions) !== benchmarkDigest(old.actions) ||
          benchmarkDigest(result.stateHashes) !== benchmarkDigest(old.stateHashes)
        )
          throw new Error('historical parity failed');
      }
      const filename = `branch-${index}-${arm}-${delay}.json.gz`;
      const changed = firstChangedAction(result, baseline);
      writeMechanismArtifact(output, filename, { ...result, firstChangedAction: changed });
      rows.push({
        cohort: index < historical.length ? 'historical' : 'fresh',
        seed: result.seed,
        roster: result.roster,
        arm,
        delayDecisions: delay,
        filename,
        firstChangedAction: changed,
        interventionStatus: result.interventionStatus,
        actualBindingChanges: result.actualBindingChanges,
        metrics: result.metrics,
      });
    }
  }
}
const report = {
  format: MECHANISM_VERSION,
  providerCalls: 0,
  verifiedContinuations: rows.length,
  executions: rows.length * 2,
  deterministicRerunsPassed: true,
  coverage: ['5v5', '10v10', '6v10'].map((roster) => {
    const group = scans.filter(
      (row) => `${row.configuration.blueUnits}v${row.configuration.redUnits}` === roster,
    );
    const qualified = group.filter((row) => row.qualified).length;
    return { roster, scanned: group.length, qualified, underCovered: qualified < 20 };
  }),
  rows,
  analysis: pairedMechanismReport(rows),
  limitations: [
    'Diagnostic casualty cohort only; no learned-fighter or provider qualification.',
    'No combined refresh/local-fire arm; interactions are unmeasured.',
    'Delay consumes shared horizon; terminal-before-intervention cases remain in paired outcomes.',
    'Censored outcomes remain separate from losses; blue win rate counts only completed blue wins.',
    'Historical cases excluded from bootstrap; no multiplicity adjustment or automatic promotion.',
    'Red actions may diverge after interventions; seed equality does not fix opponent actions.',
  ],
};
writeMechanismArtifact(output, 'report.json', report);
for (const [path, digest] of Object.entries({ ...seedSources, ...sources }))
  if (fileDigest(resolve(repository, path)) !== digest)
    throw new Error(`source changed during run: ${path}`);
sealMechanismArtifacts(output);
verifyMechanismArtifacts(output);
console.log(
  JSON.stringify(
    { output, coverage: report.coverage, branches: rows.length, providerCalls: 0 },
    null,
    2,
  ),
);
