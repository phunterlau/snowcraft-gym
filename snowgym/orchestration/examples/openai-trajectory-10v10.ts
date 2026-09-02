import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { AiDifficulty } from '../../../src/systems/AISystem';
import { MAP_IDS } from '../../scenarios/maps';
import {
  OPENAI_COMMANDER_MODEL,
  OpenAICommanderClient,
  type OpenAICommanderReasoningEffort,
} from '../providers/OpenAICommanderClient';
import { runTrajectoryCommanderBattle } from './TrajectoryCommanderBattle';

const { values } = parseArgs({
  options: {
    seed: { type: 'string', default: '42' },
    'blue-units': { type: 'string', default: '10' },
    'red-units': { type: 'string', default: '10' },
    map: { type: 'string', default: 'arena6.json' },
    'red-difficulty': { type: 'string', default: 'easy' },
    reasoning: { type: 'string', default: 'medium' },
    'pace-ms': { type: 'string', default: '100' },
    'max-decisions': { type: 'string', default: '10000' },
    'max-requests': { type: 'string', default: '3' },
    output: { type: 'string' },
    'trace-output': { type: 'string' },
    force: { type: 'boolean', default: false },
    json: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`Run a paced, renderer-free C5 trajectory-aware Luna M-v-N demo.

Usage:
  OPENAI_API_KEY=... npx tsx snowgym/orchestration/examples/openai-trajectory-10v10.ts [options]

Options:
  --seed INTEGER
  --blue-units INTEGER
  --red-units INTEGER
  --map MAP_ID            Bundled map (${MAP_IDS.join(', ')})
  --red-difficulty LEVEL  easy, normal, or hard
  --reasoning low|medium|high|xhigh|max
  --pace-ms INTEGER       Wall-clock delay per 10 Hz decision (default: 100)
  --max-decisions INTEGER
  --max-requests INTEGER  Hard provider-attempt cap (default: 3)
  --output PATH           Write the visual replay JSON
  --trace-output PATH     Write the bound commander-trace sidecar JSON
  --force                 Replace existing output paths
  --json`);
  process.exit(0);
}

const result = await runTrajectoryCommanderBattle(
  new OpenAICommanderClient({ reasoningEffort: reasoning(values.reasoning) }),
  {
    seed: integer(values.seed, 'seed'),
    blueUnits: positiveInteger(values['blue-units'], 'blue-units'),
    redUnits: positiveInteger(values['red-units'], 'red-units'),
    map: values.map,
    redDifficulty: difficulty(values['red-difficulty']),
    paceMs: nonNegativeInteger(values['pace-ms'], 'pace-ms'),
    maxDecisions: positiveInteger(values['max-decisions'], 'max-decisions'),
    maximumRequests: positiveInteger(values['max-requests'], 'max-requests'),
  },
);
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
const failures = result.schedulerEvents.filter(
  ({ type }) => type === 'request_failed' || type === 'request_timed_out',
);
const summary = {
  ok:
    signals.length > 0 &&
    responses.some(({ status }) => status !== 'rejected') &&
    result.rejectedActions === 0,
  model: responses[0]?.metadata?.model ?? OPENAI_COMMANDER_MODEL,
  blueUnits: result.replay.configuration?.blueUnits,
  redUnits: result.replay.configuration?.redUnits,
  map: result.replay.configuration?.map,
  redDifficulty: result.replay.configuration?.redDifficulty,
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
  output,
  traceOutput,
};

if (values.json) console.log(JSON.stringify(summary));
else {
  console.log('SnowGym trajectory-aware OpenAI-commanded M-v-N demo');
  console.log(`  model:       ${summary.model}`);
  console.log(`  matchup:     ${summary.blueUnits} blue vs ${summary.redUnits} red`);
  console.log(`  signals:     ${summary.trajectorySignals.length}`);
  console.log(`  requests:    ${summary.commanderRequests}/${summary.maximumRequests}`);
  console.log(`  responses:   ${JSON.stringify(summary.responses)}`);
  console.log(`  decisions:   ${summary.decisions}`);
  console.log(`  survivors:   blue=${summary.blueAlive} red=${summary.redAlive}`);
  console.log(`  winner:      ${summary.winner}`);
  if (output) console.log(`  replay:      ${output}`);
  if (traceOutput) console.log(`  trace:       ${traceOutput}`);
}

function writeJson(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
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

function difficulty(value: string | undefined): AiDifficulty {
  if (value === 'easy' || value === 'normal' || value === 'hard') return value;
  throw new RangeError('red-difficulty must be easy, normal, or hard');
}
