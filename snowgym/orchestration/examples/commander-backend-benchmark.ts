import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  BACKEND_BENCHMARK_VERSION,
  BACKEND_PILOT_CASES,
  backendBenchmarkArms,
  backendCallSchedule,
  benchmarkBackendCall,
  benchmarkDigest,
  continueBackendPlan,
  prepareBackendFixture,
  restoreBackendFixture,
  summarizeBackendRows,
  type BackendBenchmarkRow,
  type BackendFixture,
} from '../benchmark/CommanderBackendBenchmark';
import {
  fileDigest,
  sealBenchmarkArtifacts,
  verifyBenchmarkArtifacts,
  writeBenchmarkJson,
} from '../benchmark/BackendBenchmarkArtifacts';
import { OpenAICommanderClient, openAIRequestBody } from '../providers/OpenAICommanderClient';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    fixtures: { type: 'string' },
    verify: { type: 'string' },
    backends: { type: 'string', default: 'luna,astra' },
    reasoning: { type: 'string', default: 'low,medium' },
    'max-requests': { type: 'string', default: '16' },
    'max-output-tokens': { type: 'string', default: '4096' },
    'timeout-ms': { type: 'string', default: '60000' },
    horizon: { type: 'string', default: '300' },
    'dry-run': { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});
if (values.help) {
  console.log(`Matched headless Luna/Astra commander pilot (scripted executor; no RL qualification).

node --import tsx snowgym/orchestration/examples/commander-backend-benchmark.ts --output NEW_DIR [options]

--dry-run                         Freeze fixtures and baseline continuations without API calls
--fixtures FILE                   Reuse fixtures.json from a verified dry-run directory
--backends luna,astra              Exact model choices; no automatic model fallback
--reasoning low,medium             Same levels for each backend; light is an alias for low
--max-requests 16                  Hard attempt cap; refuses an oversized matrix before calls
--max-output-tokens 4096           Shared total output budget (including reasoning)
--timeout-ms 60000                 Wall-clock deadline per call; sequential; no retries
--horizon 300                      Static-plan continuation decisions at 10 Hz
--verify DIRECTORY                Verify an existing sealed artifact directory; no API calls

Live mode uses OPENAI_API_KEY from the environment. Output must not exist.
The default is 4 cases x 2 models x 2 reasoning levels = 16 requests.
API latency is measured, not injected into the zero-delay continuation.
Only aggregate commander evidence, shared instructions and the existing schema leave the process.`);
  process.exit(0);
}
if (values.verify) {
  verifyBenchmarkArtifacts(resolve(values.verify));
  console.log('Benchmark artifact digests verified.');
  process.exit(0);
}
if (!values.output) throw new Error('--output NEW_DIR is required');
const directory = resolve(values.output);
if (existsSync(directory)) throw new Error(`refusing to overwrite ${directory}`);
const integer = (value: string, name: string): number => {
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < 1)
    throw new Error(`${name} must be a positive safe integer`);
  return Number(value);
};
const arms = backendBenchmarkArms(values.backends.split(','), values.reasoning.split(','));
const horizon = integer(values.horizon, 'horizon');
const maxRequests = integer(values['max-requests'], 'max-requests');
const maxOutputTokens = integer(values['max-output-tokens'], 'max-output-tokens');
const timeoutMs = integer(values['timeout-ms'], 'timeout-ms');
const repository = fileURLToPath(new URL('../../../', import.meta.url));
let fixtures: BackendFixture[];
if (values.fixtures) {
  const path = resolve(values.fixtures);
  verifyBenchmarkArtifacts(dirname(path));
  fixtures = JSON.parse(readFileSync(path, 'utf8'));
  if (
    benchmarkDigest(fixtures.map((fixture) => fixture.configuration)) !==
    benchmarkDigest(BACKEND_PILOT_CASES)
  )
    throw new Error('pilot case declaration mismatch');
} else fixtures = BACKEND_PILOT_CASES.map(prepareBackendFixture);
fixtures.forEach(restoreBackendFixture);
const schedule = backendCallSchedule(fixtures.length, arms);
if (schedule.length > maxRequests)
  throw new Error(
    `matrix requires ${schedule.length} requests, cap is ${maxRequests}; no calls made`,
  );
// Construct clients before creating output, but never during --dry-run.
const clients = values['dry-run']
  ? []
  : arms.map((arm) => new OpenAICommanderClient({ ...arm, maxOutputTokens }));
const paths = [
  ...new Set([
    ...execFileSync(
      'git',
      ['ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', 'src', 'snowgym'],
      { cwd: repository, encoding: 'utf8' },
    )
      .split('\0')
      .filter((path) => path.endsWith('.ts') || path.endsWith('.schema.json')),
    'package.json',
    'package-lock.json',
    'tsconfig.json',
    'snowgym/PLAN.md',
  ]),
].sort();
const sources = Object.fromEntries(
  paths.map((path) => [path, fileDigest(resolve(repository, path))]),
);
const configuration = {
  format: BACKEND_BENCHMARK_VERSION,
  mode: values['dry-run'] ? 'dry-run' : 'live',
  arms,
  schedule,
  horizon,
  maxOutputTokens,
  timeoutMs,
  maxRequests,
  attemptsPlanned: values['dry-run'] ? 0 : schedule.length,
  decisionHz: 10,
  redDifficulty: 'normal',
  executor: 'PlanAwareTeamController/ReactiveUnitPolicy',
  activation: 'one static plan; no lifecycle replacement; zero simulated API delay',
  autonomousQualificationEligible: false,
  retries: 0,
  payloadBoundary:
    'strategic summary, current plan, optional aggregate trajectory and previous-plan outcome; shared prompt and CommandPlan schema',
  nodeVersion: process.version,
  createdAt: new Date().toISOString(),
  gitHead: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repository, encoding: 'utf8' }).trim(),
  sources,
  sourceDigest: benchmarkDigest(sources),
};
const baselines = fixtures.map((fixture) => ({
  caseId: fixture.configuration.id,
  continuation: continueBackendPlan(fixture, undefined, horizon),
}));
mkdirSync(dirname(directory), { recursive: true });
mkdirSync(directory); // Exclusive directory ownership: never overwrite an existing run.
writeBenchmarkJson(directory, 'configuration.json', configuration);
writeBenchmarkJson(directory, 'fixtures.json', fixtures);
writeBenchmarkJson(directory, 'baselines.json', baselines);
writeBenchmarkJson(
  directory,
  'requests.json',
  schedule.map(({ caseIndex, arm }) => ({
    caseId: fixtures[caseIndex].configuration.id,
    ...arm,
    body: openAIRequestBody(
      fixtures[caseIndex].request,
      arm.reasoningEffort,
      maxOutputTokens,
      arm.model,
    ),
  })),
);
const rows: BackendBenchmarkRow[] = [];
if (!values['dry-run']) {
  for (const [index, { caseIndex, arm }] of schedule.entries()) {
    console.log(
      `[${index + 1}/${schedule.length}] ${fixtures[caseIndex].configuration.id}: ${arm.backend}/${arm.reasoningEffort}`,
    );
    const row = await benchmarkBackendCall(clients[arms.indexOf(arm)], arm, fixtures[caseIndex], {
      horizon,
      timeoutMs,
      maxOutputTokens,
    });
    rows.push(row);
    writeBenchmarkJson(directory, `response-${String(index + 1).padStart(2, '0')}.json`, row);
    console.log(
      JSON.stringify({
        arm: row.arm,
        latencyMs: Math.round(row.latencyMs),
        status: row.continuation.activationStatus,
        error: row.error,
        metrics: row.continuation.metrics,
      }),
    );
  }
}
const paired = fixtures.map((fixture) => ({
  caseId: fixture.configuration.id,
  baseline: baselines.find((row) => row.caseId === fixture.configuration.id)!.continuation.metrics,
  arms: rows
    .filter((row) => row.caseId === fixture.configuration.id)
    .map((row) => ({
      arm: row.arm,
      eligiblePlan: !row.continuation.fallbackUsed,
      ...row.continuation.metrics,
    })),
}));
const report = {
  format: BACKEND_BENCHMARK_VERSION,
  attempts: rows.length,
  summaries: summarizeBackendRows(rows),
  paired,
  limitations: [
    'Four illustrative cases, one provider sample per arm and case; no superiority or generalization claim.',
    'Same-state zero-delay static-plan comparison; online replanning and latency-injected play are separate experiments.',
    'Red is scripted and deterministic under exact action replay, but may take different actions after trajectories diverge.',
    'Combat horizon censoring is reported; accepted-plan summaries exclude fallback continuations.',
    'Token means use known usage only; unknown usage is not zero. Cached tokens and reasoning tokens are reported separately.',
    'Latency p95 is interpolated over four observations and is descriptive, not a stable tail estimate.',
  ],
};
writeBenchmarkJson(directory, 'report.json', report);
for (const [path, digest] of Object.entries(sources))
  if (fileDigest(resolve(repository, path)) !== digest)
    throw new Error(`source changed during benchmark: ${path}; partial artifacts preserved`);
sealBenchmarkArtifacts(directory);
verifyBenchmarkArtifacts(directory);
console.log(
  JSON.stringify(
    { output: directory, attempts: rows.length, summaries: report.summaries },
    null,
    2,
  ),
);
