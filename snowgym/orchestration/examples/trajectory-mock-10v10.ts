import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { runTrajectoryMockCommanderTenVsTen } from './TrajectoryMockCommanderExample';
import type { CommanderSchedulerEvent } from '../scheduler/CommanderScheduler';

const { values } = parseArgs({
  options: {
    seed: { type: 'string', default: '42' },
    'latency-ticks': { type: 'string', default: '30' },
    'max-decisions': { type: 'string', default: '400' },
    output: { type: 'string' },
    'trace-output': { type: 'string' },
    force: { type: 'boolean', default: false },
    json: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`Run the deterministic C5 trajectory-aware mock commander 10v10 demo.

Usage:
  npx tsx snowgym/orchestration/examples/trajectory-mock-10v10.ts [options]

Options:
  --seed INTEGER
  --latency-ticks INTEGER   Simulated commander latency at 60 Hz (default: 30)
  --max-decisions INTEGER
  --output PATH             Write the visual replay JSON
  --trace-output PATH       Write the bound commander-trace sidecar JSON
  --force                   Replace existing output paths
  --json                    Emit one machine-readable summary`);
  process.exit(0);
}

const result = await runTrajectoryMockCommanderTenVsTen({
  seed: integer(values.seed, 'seed'),
  latencyTicks: nonNegativeInteger(values['latency-ticks'], 'latency-ticks'),
  maxDecisions: positiveInteger(values['max-decisions'], 'max-decisions'),
});
const output = values.output ? resolve(values.output) : null;
const traceOutput = values['trace-output'] ? resolve(values['trace-output']) : null;
if (output && traceOutput && output === traceOutput) {
  throw new Error('replay and commander trace outputs must use different paths');
}
for (const path of [output, traceOutput]) {
  if (path && existsSync(path) && !values.force) {
    throw new Error(`refusing to overwrite ${path}; pass --force to replace it`);
  }
}
if (output) writeJson(output, result.replay);
if (traceOutput) writeJson(traceOutput, result.commanderTrace);
const signals = result.schedulerEvents.filter(({ type }) => type === 'trajectory_signal');
const responses = result.schedulerEvents.filter(isProcessedResponse);
const summary = {
  ok: result.commanderRequests >= 2 && responses.some(({ status }) => status !== 'rejected'),
  commander: 'trajectory-mock-c5',
  map: 'arena6.json',
  seed: result.seed,
  latencyTicks: result.latencyTicks,
  commanderRequests: result.commanderRequests,
  signals,
  activations: responses.map(({ tick, status, sourceAgeTicks }) => ({
    tick,
    status,
    sourceAgeTicks,
  })),
  rejectedActions: result.rejectedActions,
  decisions: result.decisions,
  ticks: result.finalTick,
  blueAlive: result.blueAlive,
  redAlive: result.redAlive,
  winner: result.winner,
  output,
  traceOutput,
};

if (values.json) console.log(JSON.stringify(summary));
else {
  console.log('SnowGym trajectory-aware mock commander 10v10 demo');
  console.log(`  map:        ${summary.map}`);
  console.log(`  seed:       ${summary.seed}`);
  console.log(`  latency:    ${summary.latencyTicks} simulation ticks`);
  console.log(`  signals:    ${summary.signals.length}`);
  console.log(`  requests:   ${summary.commanderRequests}`);
  console.log(`  activations:${JSON.stringify(summary.activations)}`);
  console.log(`  decisions:  ${summary.decisions}`);
  console.log(`  survivors:  blue=${summary.blueAlive} red=${summary.redAlive}`);
  console.log(`  winner:     ${summary.winner}`);
  if (output) console.log(`  replay:     ${output}`);
  if (traceOutput) console.log(`  trace:      ${traceOutput}`);
}

function writeJson(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function integer(value: string | undefined, name: string): number {
  if (value === undefined || !/^-?\d+$/.test(value)) {
    throw new RangeError(`${name} must be an integer`);
  }
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new RangeError(`${name} must be a safe integer`);
  return result;
}

function nonNegativeInteger(value: string | undefined, name: string): number {
  const result = integer(value, name);
  if (result < 0) throw new RangeError(`${name} must be non-negative`);
  return result;
}

function positiveInteger(value: string | undefined, name: string): number {
  const result = integer(value, name);
  if (result <= 0) throw new RangeError(`${name} must be positive`);
  return result;
}

function isProcessedResponse(
  event: CommanderSchedulerEvent,
): event is Extract<CommanderSchedulerEvent, { type: 'response_processed' }> {
  return event.type === 'response_processed';
}
