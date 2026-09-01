import { parseArgs } from 'node:util';
import {
  OPENAI_COMMANDER_MODEL,
  OpenAICommanderClient,
  type OpenAICommanderReasoningEffort,
} from '../providers/OpenAICommanderClient';
import { runTrajectoryCommanderBattle } from './TrajectoryCommanderBattle';

const { values } = parseArgs({
  options: {
    seed: { type: 'string', default: '42' },
    reasoning: { type: 'string', default: 'medium' },
    'pace-ms': { type: 'string', default: '100' },
    'max-decisions': { type: 'string', default: '10000' },
    'max-requests': { type: 'string', default: '3' },
    json: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`Run the paced, renderer-free C5 trajectory-aware Luna 10v10 demo.

Usage:
  OPENAI_API_KEY=... npx tsx snowgym/orchestration/examples/openai-trajectory-10v10.ts [options]

Options:
  --seed INTEGER
  --reasoning low|medium|high|xhigh|max
  --pace-ms INTEGER       Wall-clock delay per 10 Hz decision (default: 100)
  --max-decisions INTEGER
  --max-requests INTEGER  Hard provider-attempt cap (default: 3)
  --json`);
  process.exit(0);
}

const result = await runTrajectoryCommanderBattle(
  new OpenAICommanderClient({ reasoningEffort: reasoning(values.reasoning) }),
  {
    seed: integer(values.seed, 'seed'),
    paceMs: nonNegativeInteger(values['pace-ms'], 'pace-ms'),
    maxDecisions: positiveInteger(values['max-decisions'], 'max-decisions'),
    maximumRequests: positiveInteger(values['max-requests'], 'max-requests'),
  },
);
const signals = result.schedulerEvents.filter(({ type }) => type === 'trajectory_signal');
const responses = result.schedulerEvents.filter(isProcessedResponse);
const failures = result.schedulerEvents.filter(
  ({ type }) => type === 'request_failed' || type === 'request_timed_out',
);
const summary = {
  ok:
    signals.length > 0 &&
    responses.some(({ status }) => status !== 'rejected') &&
    result.rejectedActions === 0,
  model: responses[0]?.metadata?.model ?? OPENAI_COMMANDER_MODEL,
  map: 'arena6.json',
  seed: result.seed,
  paceMs: result.paceMs,
  maximumRequests: result.maximumRequests,
  commanderRequests: result.schedulerEvents.filter(({ type }) => type === 'request_started').length,
  trajectorySignals: signals,
  responses: responses.map(({ tick, status, sourceAgeTicks, metadata }) => ({
    tick,
    status,
    sourceAgeTicks,
    latencyMs: metadata?.latencyMs ?? null,
    tokensIn: metadata?.tokensIn ?? null,
    tokensOut: metadata?.tokensOut ?? null,
    reasoningTokens: metadata?.reasoningTokens ?? null,
  })),
  failures,
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
  console.log('SnowGym trajectory-aware OpenAI-commanded 10v10 demo');
  console.log(`  model:       ${summary.model}`);
  console.log(`  signals:     ${summary.trajectorySignals.length}`);
  console.log(`  requests:    ${summary.commanderRequests}/${summary.maximumRequests}`);
  console.log(`  responses:   ${JSON.stringify(summary.responses)}`);
  console.log(`  decisions:   ${summary.decisions}`);
  console.log(`  survivors:   blue=${summary.blueAlive} red=${summary.redAlive}`);
  console.log(`  winner:      ${summary.winner}`);
}

function isProcessedResponse(
  event: (typeof result.schedulerEvents)[number],
): event is Extract<(typeof result.schedulerEvents)[number], { type: 'response_processed' }> {
  return event.type === 'response_processed';
}

function reasoning(value: string | undefined): OpenAICommanderReasoningEffort {
  if (
    value === 'low' ||
    value === 'medium' ||
    value === 'high' ||
    value === 'xhigh' ||
    value === 'max'
  ) {
    return value;
  }
  throw new RangeError('reasoning must be low, medium, high, xhigh, or max');
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
