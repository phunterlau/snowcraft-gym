import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { buildCommandedTenVsTenReplay } from './CommandedReplayExample';

const { values } = parseArgs({
  options: {
    seed: { type: 'string', default: '42' },
    'max-decisions': { type: 'string' },
    output: { type: 'string' },
    force: { type: 'boolean', default: false },
    json: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`Run the deterministic headless C1 commanded 10v10 demo.

Usage:
  npx tsx snowgym/orchestration/examples/commanded-10v10.ts [options]

Options:
  --seed INTEGER
  --max-decisions INTEGER
  --output PATH          Optionally write a visual-replay JSON artifact
  --force                Replace an existing output path
  --json                 Emit one machine-readable summary`);
  process.exit(0);
}

const seed = integer(values.seed, 'seed');
const result = buildCommandedTenVsTenReplay({
  seed,
  maxDecisions: optionalInteger(values['max-decisions'], 'max-decisions'),
});
let output: string | null = null;
if (values.output) {
  output = resolve(values.output);
  if (existsSync(output) && !values.force) {
    throw new Error(`refusing to overwrite ${output}; pass --force to replace it`);
  }
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, `${JSON.stringify(result.replay, null, 2)}\n`, 'utf8');
}

const summary = {
  ok: true,
  model: null,
  commander: 'hard-coded-c1',
  map: result.replay.configuration?.map,
  seed,
  assignments: Object.fromEntries(
    result.plan.groups.map(({ role, assignment }) => [role, assignment.unitIds]),
  ),
  missions: Object.fromEntries(
    result.plan.groups.map(({ role, command }) => [role, command.order.mission]),
  ),
  actionsByRole: result.actionsByRole,
  rejectedActions: result.rejectedActions,
  decisions: result.replay.outcome.decisions,
  ticks: result.replay.outcome.finalTick,
  blueAlive: result.replay.outcome.blueAlive,
  redAlive: result.replay.outcome.redAlive,
  winner: result.replay.outcome.winner,
  output,
};

if (values.json) console.log(JSON.stringify(summary));
else {
  console.log('SnowGym commanded 10v10 headless demo');
  console.log(`  map:         ${summary.map}`);
  console.log(`  seed:        ${summary.seed}`);
  console.log(`  assignments: ${JSON.stringify(summary.assignments)}`);
  console.log(`  missions:    ${JSON.stringify(summary.missions)}`);
  console.log(`  decisions:   ${summary.decisions}`);
  console.log(`  survivors:   blue=${summary.blueAlive} red=${summary.redAlive}`);
  console.log(`  winner:      ${summary.winner}`);
  if (output) console.log(`  output:      ${output}`);
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
