import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import type { MapData } from '../../src/game/types';
import { validateGeneratedMap } from './MapValidator';

const ID_PATTERN = /^arena[A-Za-z0-9_-]+\.json$/;
const REGISTRY_MARKER = '  // MAPGEN_PROMOTED_MAPS';
const BROWSER_MARKER = '  // MAPGEN_PROMOTED_MAPS';

export interface PromotionResult {
  id: string;
  mapPath: string;
  registryPath: string;
  browserPath: string;
}

/** Explicitly promotes one validated artifact into both static map catalogs. */
export function promoteGeneratedMap(options: {
  map: MapData;
  id: string;
  repositoryRoot?: string;
  force?: boolean;
}): PromotionResult {
  if (!ID_PATTERN.test(options.id)) {
    throw new RangeError('promotion id must match arena[A-Za-z0-9_-]+.json');
  }
  const validation = validateGeneratedMap(options.map);
  if (!validation.valid || !validation.canonicalMap)
    throw new Error('refusing to promote an invalid map');
  const root = resolve(options.repositoryRoot ?? process.cwd());
  const mapPath = `${root}/public/maps/${options.id}`;
  const registryPath = `${root}/snowgym/scenarios/maps.ts`;
  const browserPath = `${root}/src/main.ts`;
  if (existsSync(mapPath) && !options.force) {
    throw new Error(`refusing to overwrite ${mapPath}; pass --force to replace it`);
  }
  const registry = readFileSync(registryPath, 'utf8');
  const browser = readFileSync(browserPath, 'utf8');
  const registryBlock = block(
    'registry',
    options.id,
    registryEntry(options.id, validation.canonicalMap),
  );
  const browserBlock = block(
    'browser',
    options.id,
    browserEntry(options.id, validation.canonicalMap),
  );
  const nextRegistry = insertOrReplace(
    registry,
    REGISTRY_MARKER,
    registryBlock,
    'registry',
    options.id,
    options.force ?? false,
  );
  const nextBrowser = insertOrReplace(
    browser,
    BROWSER_MARKER,
    browserBlock,
    'browser',
    options.id,
    options.force ?? false,
  );
  writeFileSync(mapPath, `${JSON.stringify(validation.canonicalMap, null, 2)}\n`, 'utf8');
  writeFileSync(registryPath, nextRegistry, 'utf8');
  writeFileSync(browserPath, nextBrowser, 'utf8');
  return { id: options.id, mapPath, registryPath, browserPath };
}

function registryEntry(id: string, map: MapData): string {
  const json = JSON.stringify(map, null, 2)
    .split('\n')
    .map((line, index) => (index === 0 ? line : `  ${line}`))
    .join('\n');
  return `  ${JSON.stringify(id)}: ${json} as unknown as MapData,`;
}

function browserEntry(id: string, map: MapData): string {
  return `  { label: ${JSON.stringify(map.name ?? basename(id, '.json'))}, value: ${JSON.stringify(id)} },`;
}

function block(kind: string, id: string, entry: string): string {
  return `  // MAPGEN_${kind.toUpperCase()} ${id} BEGIN\n${entry}\n  // MAPGEN_${kind.toUpperCase()} ${id} END`;
}

function insertOrReplace(
  source: string,
  marker: string,
  nextBlock: string,
  kind: string,
  id: string,
  force: boolean,
): string {
  if (!source.includes(marker)) throw new Error(`missing promotion marker in ${kind} source`);
  const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(
    `  // MAPGEN_${kind.toUpperCase()} ${escaped} BEGIN\\n[\\s\\S]*?\\n  // MAPGEN_${kind.toUpperCase()} ${escaped} END\\n?`,
  );
  const exists = pattern.test(source);
  if (exists && !force) throw new Error(`${id} is already present in the ${kind} catalog`);
  const withoutExisting = source.replace(pattern, '');
  return withoutExisting.replace(marker, `${nextBlock}\n${marker}`);
}
