import { parseArgs } from 'node:util';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import { createMapScenario } from '../../scenarios/Scenario';
import { parseCommandPlan } from '../command/PlanValidator';
import type { CommanderRequest } from '../commander/CommanderClient';
import { summarizeStrategy } from '../commander/StrategicSummary';
import { PlanGrounder } from '../grounding/PlanGrounder';
import { createFallbackEnvelope } from '../lifecycle/FallbackPlan';
import {
  OPENAI_COMMANDER_MODEL,
  OpenAICommanderClient,
  type OpenAICommanderReasoningEffort,
} from '../providers/OpenAICommanderClient';
import { PlanStore } from '../runtime/PlanStore';

const { values } = parseArgs({
  options: {
    seed: { type: 'string', default: '42' },
    reasoning: { type: 'string', default: 'medium' },
    json: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`Request and validate one live SnowGym command plan.

Usage:
  OPENAI_API_KEY=... npx tsx snowgym/orchestration/examples/openai-commander-smoke.ts [options]

Options:
  --seed INTEGER
  --reasoning low|medium|high|xhigh|max
  --json`);
  process.exit(0);
}

const seed = integer(values.seed, 'seed');
const scenario = createMapScenario('arena6.json', { seed, blueUnits: 10, redUnits: 10 });
const environment = new SnowEnvironment({ scenario, decisionHz: 10, redDifficulty: 'easy' });
const observation = environment.reset(seed);
const initial = new PlanGrounder().ground(
  createFallbackEnvelope(observation, 'commander_pending', 0),
  observation,
);
const store = new PlanStore(initial, observation.tick);
const request: CommanderRequest = {
  requestId: 'openai-commander-smoke-1',
  triggers: ['plan_expired'],
  summary: summarizeStrategy(observation, store.current()),
  currentPlan: store.current().plan.envelope.decision,
};
const response = await new OpenAICommanderClient({
  reasoningEffort: reasoning(values.reasoning),
}).plan(request);
const decision = parseCommandPlan(response.decision);
const summary = {
  ok: true,
  model: response.metadata?.model ?? OPENAI_COMMANDER_MODEL,
  latencyMs: response.metadata?.latencyMs,
  tokensIn: response.metadata?.tokensIn,
  tokensOut: response.metadata?.tokensOut,
  reasoningTokens: response.metadata?.reasoningTokens,
  responseId: response.metadata?.responseId,
  plan: decision,
};

if (values.json) console.log(JSON.stringify(summary));
else {
  console.log('SnowGym live OpenAI commander smoke');
  console.log(`  model:      ${summary.model}`);
  console.log(`  latency:    ${Math.round(summary.latencyMs ?? 0)} ms`);
  console.log(`  tokens:     in=${summary.tokensIn ?? '?'} out=${summary.tokensOut ?? '?'}`);
  console.log(`  reasoning:  ${summary.reasoningTokens ?? '?'} tokens`);
  console.log(`  plan:       ${JSON.stringify(summary.plan)}`);
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
