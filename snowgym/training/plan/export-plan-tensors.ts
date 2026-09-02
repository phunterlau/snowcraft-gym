import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { createMapScenario, createOpenScenario } from '../../scenarios/Scenario';
import { buildPlanTensorDataset } from './PlanTensorDataset';

const { values } = parseArgs({
  options: {
    map: { type: 'string', default: 'arena6.json' },
    'blue-units': { type: 'string', default: '10' },
    'red-units': { type: 'string', default: '10' },
    'environment-seed': { type: 'string', default: '42' },
    'plan-seed': { type: 'string', default: '1000' },
    samples: { type: 'string', default: '60' },
    output: { type: 'string', required: true },
    force: { type: 'boolean', default: false },
    json: { type: 'boolean', default: false },
  },
  strict: true,
});

const map = values.map;
const blueUnits = integer(values['blue-units'], 'blue-units');
const redUnits = integer(values['red-units'], 'red-units');
const environmentSeed = integer(values['environment-seed'], 'environment-seed');
const basePlanSeed = integer(values['plan-seed'], 'plan-seed');
const sampleCount = positiveInteger(values.samples, 'samples');
if (values.output === undefined) throw new RangeError('output is required');
const output = resolve(values.output);
if (existsSync(output) && !values.force) {
  throw new Error(`refusing to overwrite ${output}; pass --force to replace it`);
}
const scenario =
  map === 'open'
    ? createOpenScenario({ seed: environmentSeed, blueUnits, redUnits })
    : createMapScenario(map, {
        seed: environmentSeed,
        blueUnits,
        redUnits,
      });
const dataset = buildPlanTensorDataset({
  scenario,
  environmentSeed,
  basePlanSeed,
  sampleCount,
});
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(dataset, null, 2)}\n`, 'utf8');

const summary = {
  ok: true,
  format: dataset.format,
  scenario: dataset.scenario,
  environmentSeed,
  basePlanSeed,
  samples: sampleCount,
  datasetDigest: dataset.datasetDigest,
  output,
};
console.log(values.json ? JSON.stringify(summary) : summary);

function integer(value: string | undefined, name: string): number {
  if (value === undefined || !/^-?\d+$/.test(value)) {
    throw new RangeError(`${name} must be an integer`);
  }
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new RangeError(`${name} must be a safe integer`);
  return result;
}

function positiveInteger(value: string | undefined, name: string): number {
  const result = integer(value, name);
  if (result <= 0) throw new RangeError(`${name} must be positive`);
  return result;
}
