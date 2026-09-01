import { parseArgs } from 'node:util';
import { runDelayedMockCommanderTenVsTen } from './DelayedMockCommanderExample';

const { values } = parseArgs({
  options: {
    seed: { type: 'string', default: '42' },
    'latency-ticks': { type: 'string', default: '90' },
    'max-decisions': { type: 'string' },
    json: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`Run the deterministic headless C3 delayed-commander 10v10 demo.

Usage:
  npx tsx snowgym/orchestration/examples/delayed-mock-10v10.ts [options]

Options:
  --seed INTEGER
  --latency-ticks INTEGER   Simulated commander latency at 60 Hz (default: 90)
  --max-decisions INTEGER
  --json                    Emit one machine-readable summary`);
  process.exit(0);
}

const result = await runDelayedMockCommanderTenVsTen({
  seed: integer(values.seed, 'seed'),
  latencyTicks: integer(values['latency-ticks'], 'latency-ticks'),
  maxDecisions: optionalInteger(values['max-decisions'], 'max-decisions'),
});
const processed = result.schedulerEvents.find(({ type }) => type === 'response_processed');
const summary = {
  ok: true,
  commander: 'delayed-mock-c3',
  model: null,
  map: 'arena6.json',
  seed: result.seed,
  latencyTicks: result.latencyTicks,
  activationTick: processed?.tick ?? null,
  activationStatus: processed?.type === 'response_processed' ? processed.status : null,
  assignments: result.finalAssignments,
  rejectedActions: result.rejectedActions,
  decisions: result.decisions,
  ticks: result.finalTick,
  blueAlive: result.blueAlive,
  redAlive: result.redAlive,
  winner: result.winner,
};

if (values.json) console.log(JSON.stringify(summary));
else {
  console.log('SnowGym delayed mock-commander 10v10 headless demo');
  console.log(`  map:        ${summary.map}`);
  console.log(`  seed:       ${summary.seed}`);
  console.log(`  latency:    ${summary.latencyTicks} simulation ticks`);
  console.log(`  activation: tick=${summary.activationTick} status=${summary.activationStatus}`);
  console.log(`  assignments:${JSON.stringify(summary.assignments)}`);
  console.log(`  decisions:  ${summary.decisions}`);
  console.log(`  survivors:  blue=${summary.blueAlive} red=${summary.redAlive}`);
  console.log(`  winner:     ${summary.winner}`);
}

function integer(value: string | undefined, name: string): number {
  if (value === undefined || !/^-?\d+$/.test(value)) {
    throw new RangeError(`${name} must be an integer`);
  }
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new RangeError(`${name} must be a safe integer`);
  return result;
}

function optionalInteger(value: string | undefined, name: string): number | undefined {
  return value === undefined ? undefined : integer(value, name);
}
