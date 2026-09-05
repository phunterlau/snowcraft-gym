import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  fileDigest,
  sealBenchmarkArtifacts,
  verifyBenchmarkArtifacts,
  writeBenchmarkJson,
} from '../benchmark/BackendBenchmarkArtifacts';
import { benchmarkDigest } from '../benchmark/CommanderBackendBenchmark';
import { commanderBackend } from '../providers/CommanderBackend';
import { openAIRequestBody } from '../providers/OpenAICommanderClient';
import {
  collectRecoveryFixtures,
  continueRecoveryPlan,
  recoveryRequest,
  RECOVERY_BENCHMARK_VERSION,
} from '../recovery/RecoveryBenchmark';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    verify: { type: 'string' },
    horizon: { type: 'string', default: '300' },
    'scan-horizon': { type: 'string', default: '300' },
    help: { type: 'boolean', short: 'h' },
  },
  strict: true,
});
if (values.help) {
  console.log(`Headless commander recovery preflight; no credentials or provider calls.
node --import tsx snowgym/orchestration/examples/commander-recovery-benchmark.ts --output NEW_DIR
--scan-horizon 300     Maximum decisions per declared scan world (20..1000)
--horizon 300          Shared continuation horizon including delay (81..1000)
--verify DIRECTORY    Verify sealed artifact inventory and digests
Four recovery predicates; missing families block the planned provider comparison.
Freezes Luna-low, Astra-low and Astra-medium requests for old/enriched input.
Runs no-change and same-symbol reactivation controls at 0/1/2/4/8 simulated seconds.
Output must not exist. This is a scripted-executor diagnostic, not RL qualification.`);
  process.exit(0);
}
if (values.verify) {
  verifyBenchmarkArtifacts(resolve(values.verify));
  console.log('Recovery artifact inventory and digests verified.');
  process.exit(0);
}
if (!values.output) throw new Error('--output NEW_DIR is required');
const output = resolve(values.output);
if (existsSync(output)) throw new Error(`refusing to overwrite ${output}`);
const horizon = Number(values.horizon);
const scanHorizon = Number(values['scan-horizon']);
if (
  !Number.isSafeInteger(horizon) ||
  horizon < 81 ||
  horizon > 1000 ||
  !Number.isSafeInteger(scanHorizon) ||
  scanHorizon < 20 ||
  scanHorizon > 1000
)
  throw new Error('invalid horizon; see --help');
const repository = fileURLToPath(new URL('../../../', import.meta.url));
const paths = [
  ...new Set(
    execFileSync(
      'git',
      ['ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', 'src', 'snowgym'],
      { cwd: repository, encoding: 'utf8' },
    )
      .split('\0')
      .filter((path) => path.endsWith('.ts') || path.endsWith('.schema.json')),
  ),
  'package.json',
  'package-lock.json',
  'tsconfig.json',
  'snowgym/PLAN.md',
];
const sources = Object.fromEntries(
  paths.sort().map((path) => [path, fileDigest(resolve(repository, path))]),
);
const collection = collectRecoveryFixtures(undefined, scanHorizon);
mkdirSync(output, { recursive: true });
const arms = [
  commanderBackend('luna', 'low'),
  commanderBackend('astra', 'low'),
  commanderBackend('astra', 'medium'),
];
const delays = [0, 10, 20, 40, 80];
writeBenchmarkJson(output, 'configuration.json', {
  format: RECOVERY_BENCHMARK_VERSION,
  sources,
  sourceDigest: benchmarkDigest(sources),
  horizon,
  scanHorizon,
  delays,
  arms,
  executor: 'PlanAwareTeamController/ReactiveUnitPolicy',
  autonomousQualificationEligible: false,
  providerAttempts: 0,
  redDifficulty: 'normal',
  decisionHz: 10,
});
writeBenchmarkJson(output, 'fixtures.json', collection.fixtures);
writeBenchmarkJson(output, 'scan.json', collection.scans);
writeBenchmarkJson(
  output,
  'requests.json',
  collection.fixtures.flatMap((fixture) =>
    [false, true].flatMap((enriched) =>
      arms.map((arm) => ({
        fixtureDigest: fixture.digest,
        enriched,
        ...arm,
        body: openAIRequestBody(
          recoveryRequest(fixture, enriched),
          arm.reasoningEffort,
          4096,
          arm.model,
        ),
      })),
    ),
  ),
);
const rows = [];
for (const [index, fixture] of collection.fixtures.entries()) {
  console.log(
    `[${index + 1}/${collection.fixtures.length}] ${fixture.base.configuration.id}/${fixture.family}`,
  );
  for (const control of ['keep', 'reactivate_current'] as const)
    for (const delay of delays) {
      const decision = control === 'keep' ? null : fixture.base.request.currentPlan;
      const result = continueRecoveryPlan(fixture, decision, delay, horizon);
      const duplicate = continueRecoveryPlan(fixture, decision, delay, horizon);
      if (benchmarkDigest(result) !== benchmarkDigest(duplicate))
        throw new Error('continuation reproducibility failed');
      const filename = `continuation-${index}-${control}-${delay}.json`;
      writeBenchmarkJson(output, filename, result);
      rows.push({
        fixtureDigest: fixture.digest,
        family: fixture.family,
        control,
        delaySeconds: delay / 10,
        filename,
        activationStatus: result.activationStatus,
        metrics: result.metrics,
      });
    }
}
const report = {
  format: RECOVERY_BENCHMARK_VERSION,
  providerAttempts: 0,
  fixtures: collection.fixtures.length,
  missingFamilies: collection.missingFamilies,
  scenarioCoveragePassed: collection.missingFamilies.length === 0,
  providerReady: false,
  nextGate:
    'Review coverage and predicate validity; predeclare paired provider budget before live comparison.',
  deterministicRerunsPassed: true,
  rows,
  limitations: [
    'First qualifying event per family/world; multiple families from one world are correlated.',
    'The observation is provider input only in aggregate; exact prefixes and physical snapshots remain local.',
    'Straight-line AABB obstruction is a conservative proxy, not route reachability or ballistic cover.',
    'Enemy health loss is team-level; throws without damage can reflect flight time, cover, range or aim.',
    'Stage 4 scores the original frozen target and may already be true at request time.',
    'No online multi-request recovery, calibrated affordance model or causal model superiority tested.',
    'Keeping a plan preserves activation anchors; reactivating identical symbols can change their grounding.',
    'Red may choose different actions after candidate trajectories diverge.',
  ],
};
writeBenchmarkJson(output, 'report.json', report);
for (const [path, digest] of Object.entries(sources))
  if (fileDigest(resolve(repository, path)) !== digest)
    throw new Error(`source changed during run: ${path}`);
sealBenchmarkArtifacts(output);
verifyBenchmarkArtifacts(output);
console.log(
  JSON.stringify(
    {
      output,
      fixtures: report.fixtures,
      missingFamilies: report.missingFamilies,
      continuations: rows.length,
      deterministicRerunsPassed: true,
      providerAttempts: 0,
    },
    null,
    2,
  ),
);
