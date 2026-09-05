import {
  validReasoningEffort,
  defaultCommanderReasoning,
  type OpenAICommanderModel,
  type OpenAICommanderReasoningEffort,
} from './OpenAICommanderClient';

export type CommanderBackend = 'luna' | 'astra';
export const DEFAULT_COMMANDER_BACKEND = 'astra' as const;

/** One CLI resolver for all demos; no model substitution or provider fallback. */
export function commanderBackend(
  backend: string = DEFAULT_COMMANDER_BACKEND,
  reasoning?: string,
): {
  backend: CommanderBackend;
  model: OpenAICommanderModel;
  reasoningEffort: OpenAICommanderReasoningEffort;
} {
  if (backend !== 'luna' && backend !== 'astra')
    throw new RangeError('backend must be luna or astra');
  const model = backend === 'luna' ? 'gpt-5.6-luna' : 'gpt-6-astra';
  return {
    backend,
    model,
    reasoningEffort: validReasoningEffort(
      reasoning === 'light' ? 'low' : (reasoning ?? defaultCommanderReasoning(model)),
    ),
  };
}
