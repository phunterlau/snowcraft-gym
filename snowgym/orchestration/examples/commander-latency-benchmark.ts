import { existsSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { runCommanderLatencyBenchmark } from '../benchmark/CommanderLatencyBenchmark';

const { values } = parseArgs({ options: {
  seeds: { type: 'string', default: '11,12,13,14,15' },
  'latency-ticks': { type: 'string', default: '0,6,15,30,60,120,240,480' },
  'max-decisions': { type: 'string', default: '400' },
  'blue-units': { type: 'string', default: '10' },
  'red-units': { type: 'string', default: '10' },
  map: { type: 'string', default: 'arena6.json' },
  output: { type: 'string' }, force: { type: 'boolean', default: false },
  json: { type: 'boolean', default: false },
} });

const list = (value: string, name: string) => value.split(',').map((item) => {
  const parsed = Number(item);
  if (!Number.isSafeInteger(parsed)) throw new RangeError(`${name} must be comma-separated integers`);
  return parsed;
});
const positive = (value: string, name: string) => {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new RangeError(`${name} must be positive`);
  return parsed;
};
const result = await runCommanderLatencyBenchmark({
  seeds: list(values.seeds!, 'seeds'),
  latencyTicks: list(values['latency-ticks']!, 'latency-ticks'),
  maxDecisions: positive(values['max-decisions']!, 'max-decisions'),
  blueUnits: positive(values['blue-units']!, 'blue-units'),
  redUnits: positive(values['red-units']!, 'red-units'), map: values.map,
});
if (values.output) {
  const output = resolve(values.output);
  if (existsSync(output) && !values.force) throw new Error(`refusing to overwrite ${output}; pass --force`);
  writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
}
console.log(values.json ? JSON.stringify(result) : result.summaries);
