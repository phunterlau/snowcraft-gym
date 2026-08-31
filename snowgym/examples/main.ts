import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { AiDifficulty } from '../../src/systems/AISystem';
import { RED_CONTROLLER_TYPES, type RedControllerType } from '../agents/opponents';
import { MAP_IDS, mapSpawns } from '../scenarios/maps';
import { Team } from '../../src/game/types';
import { buildReplayExample } from './ExampleBuilder';

const { values } = parseArgs({
  options: {
    'blue-units': { type: 'string', default: '3' },
    'red-units': { type: 'string', default: '3' },
    map: { type: 'string', default: 'open' },
    seed: { type: 'string', default: '42' },
    'arena-width': { type: 'string' },
    'arena-height': { type: 'string' },
    'max-ticks': { type: 'string' },
    'max-decisions': { type: 'string' },
    'decision-hz': { type: 'string', default: '10' },
    'red-difficulty': { type: 'string', default: 'normal' },
    'red-controller': { type: 'string', default: 'scripted' },
    output: { type: 'string' },
    force: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
  strict: true,
});

if (values.help) {
  printHelp();
  process.exit(0);
}

const blueUnits = integer(values['blue-units'], 'blue-units');
const redUnits = integer(values['red-units'], 'red-units');
const map = values.map;
const seed = integer(values.seed, 'seed');
const redDifficulty = difficulty(values['red-difficulty']);
const redController = controller(values['red-controller']);
const output = resolve(
  values.output ??
    `public/replays/example-${map === 'open' ? 'open' : map.replace(/\.json$/, '')}-${blueUnits}v${redUnits}-seed-${seed}.json`,
);
if (existsSync(output) && !values.force) {
  throw new Error(`refusing to overwrite ${output}; pass --force to replace it`);
}

const replay = buildReplayExample({
  blueUnits,
  redUnits,
  map,
  seed,
  arenaWidth: optionalNumber(values['arena-width'], 'arena-width'),
  arenaHeight: optionalNumber(values['arena-height'], 'arena-height'),
  maxTicks: optionalInteger(values['max-ticks'], 'max-ticks'),
  maxDecisions: optionalInteger(values['max-decisions'], 'max-decisions'),
  decisionHz: integer(values['decision-hz'], 'decision-hz'),
  redDifficulty,
  redController,
});
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(replay, null, 2)}\n`, 'utf8');

console.log('SnowGym example built');
console.log(
  `  matchup:   ${replay.configuration?.blueUnits} blue vs ${replay.configuration?.redUnits} red`,
);
console.log(`  map:       ${replay.configuration?.map ?? 'open'}`);
console.log(`  seed:      ${replay.seed}`);
console.log(`  decisions: ${replay.outcome.decisions}`);
console.log(`  ticks:     ${replay.outcome.finalTick}`);
console.log(`  survivors: blue=${replay.outcome.blueAlive} red=${replay.outcome.redAlive}`);
console.log(`  winner:    ${replay.outcome.winner}`);
console.log(`  output:    ${output}`);

function printHelp(): void {
  console.log(`Build a deterministic SnowGym visual replay without starting a server.

Usage:
  npm run snowgym:example -- --blue-units M --red-units N [options]

Options:
  --map open|MAP_ID          Terrain type (default: open)
  --seed INTEGER            Episode seed (default: 42)
  --arena-width NUMBER      Open-arena width only
  --arena-height NUMBER     Open-arena height only
  --max-ticks INTEGER       Simulation truncation limit
  --max-decisions INTEGER   Builder decision safety limit
  --decision-hz INTEGER     Policy decisions per second (divisor of 60)
  --red-difficulty LEVEL    easy, normal, or hard
  --red-controller TYPE     ${RED_CONTROLLER_TYPES.join(', ')}
  --output PATH             Replay JSON destination
  --force                   Replace an existing destination

Bundled maps and native capacities:
${MAP_IDS.map((id) => `  ${id.padEnd(12)} ${mapSpawns(id, Team.Player).length}v${mapSpawns(id, Team.Enemy).length} maximum`).join('\n')}

Map rosters select evenly distributed native spawn points and cannot exceed
the listed capacity. Use --map open for arbitrary 1v1 through 10v10.`);
}

function integer(value: string | undefined, name: string): number {
  if (value === undefined || !/^-?\d+$/.test(value))
    throw new RangeError(`${name} must be an integer`);
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new RangeError(`${name} must be a safe integer`);
  return result;
}

function optionalInteger(value: string | undefined, name: string): number | undefined {
  return value === undefined ? undefined : integer(value, name);
}

function optionalNumber(value: string | undefined, name: string): number | undefined {
  if (value === undefined) return undefined;
  const result = Number(value);
  if (!Number.isFinite(result)) throw new RangeError(`${name} must be finite`);
  return result;
}

function difficulty(value: string | undefined): AiDifficulty {
  if (value === 'easy' || value === 'normal' || value === 'hard') return value;
  throw new RangeError('red-difficulty must be easy, normal, or hard');
}

function controller(value: string | undefined): RedControllerType {
  if (value === 'scripted' || value === 'random') return value;
  throw new RangeError(`red-controller must be one of: ${RED_CONTROLLER_TYPES.join(', ')}`);
}
