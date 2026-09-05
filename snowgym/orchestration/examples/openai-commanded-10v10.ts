import { parseArgs } from 'node:util';
import { OpenAICommanderClient } from '../providers/OpenAICommanderClient';
import { commanderBackend, DEFAULT_COMMANDER_BACKEND } from '../providers/CommanderBackend';
import { runSingleRequestCommanderBattle } from './OpenAICommandedBattle';

const { values } = parseArgs({
  options: {
    seed: { type: 'string', default: '42' },
    backend: { type: 'string', default: DEFAULT_COMMANDER_BACKEND },
    reasoning: { type: 'string' },
    'pace-ms': { type: 'string', default: '100' },
    'max-decisions': { type: 'string', default: '10000' },
    json: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`Run a paced, renderer-free 10v10 with exactly one OpenAI commander request.

Usage:
  OPENAI_API_KEY=... npx tsx snowgym/orchestration/examples/openai-commanded-10v10.ts [options]

Options:
  --seed INTEGER
  --backend luna|astra     Default: astra/low; explicit luna defaults to medium
  --reasoning light|low|medium|high|xhigh|max
  --pace-ms INTEGER       Wall-clock delay per 10 Hz decision (default: 100)
  --max-decisions INTEGER
  --json`);
  process.exit(0);
}

const backend = commanderBackend(values.backend, values.reasoning);
const result = await runSingleRequestCommanderBattle(new OpenAICommanderClient(backend), {
  seed: integer(values.seed, 'seed'),
  paceMs: nonNegativeInteger(values['pace-ms'], 'pace-ms'),
  maxDecisions: positiveInteger(values['max-decisions'], 'max-decisions'),
});
const response = result.schedulerEvents.find(({ type }) => type === 'response_processed');
const summary = {
  backend: backend.backend,
  reasoningEffort: backend.reasoningEffort,
  ok: response?.type === 'response_processed' && response.status !== 'rejected',
  commanderRequests: result.schedulerEvents.filter(({ type }) => type === 'request_started').length,
  model: response?.type === 'response_processed' ? response.metadata?.model : backend.model,
  map: 'arena6.json',
  seed: result.seed,
  paceMs: result.paceMs,
  activationTick: response?.type === 'response_processed' ? response.tick : null,
  activationStatus: response?.type === 'response_processed' ? response.status : null,
  sourceAgeTicks: response?.type === 'response_processed' ? response.sourceAgeTicks : null,
  apiLatencyMs: response?.type === 'response_processed' ? response.metadata?.latencyMs : null,
  reasoningTokens:
    response?.type === 'response_processed' ? response.metadata?.reasoningTokens : null,
  activePlan: result.activePlan,
  assignments: result.assignments,
  rejectedActions: result.rejectedActions,
  decisions: result.decisions,
  ticks: result.finalTick,
  wallTimeMs: result.wallTimeMs,
  blueAlive: result.blueAlive,
  redAlive: result.redAlive,
  winner: result.winner,
};

if (values.json) console.log(JSON.stringify(summary));
else {
  console.log('SnowGym single-request OpenAI-commanded 10v10 demo');
  console.log(`  model:       ${summary.model}`);
  console.log(`  requests:    ${summary.commanderRequests}`);
  console.log(`  activation:  tick=${summary.activationTick} status=${summary.activationStatus}`);
  console.log(`  API latency: ${Math.round(summary.apiLatencyMs ?? 0)} ms`);
  console.log(`  assignments: ${JSON.stringify(summary.assignments)}`);
  console.log(`  decisions:   ${summary.decisions}`);
  console.log(`  survivors:   blue=${summary.blueAlive} red=${summary.redAlive}`);
  console.log(`  winner:      ${summary.winner}`);
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
