import { MapLoader } from '../../src/game/MapLoader';
import { Team, type Arena, type MapData } from '../../src/game/types';
import { IdAllocator } from '../../src/ecs/Entity';
import { getMapData, mapSpawns } from './maps';

export const MAX_TEAM_SIZE = 10;
export const DEFAULT_MAX_TICKS = 60 * 180;
const DEFAULT_SEED = 0x5a17c0de;
const MIN_ARENA_SIZE = 12;
const MAX_ARENA_SIZE = 120;
const SPAWN_MARGIN = 0.5;
const MIN_SPAWN_DISTANCE = 1;

export interface SpawnPosition {
  x: number;
  y: number;
}

export interface Scenario {
  name: string;
  seed: number;
  arena: { width: number; height: number };
  blueSpawns: ReadonlyArray<SpawnPosition>;
  redSpawns: ReadonlyArray<SpawnPosition>;
  respawn: boolean;
  buffs: boolean;
  maxTicks: number;
  /** Bundled map id when the fight runs on real terrain; undefined = open arena. */
  map?: string;
  /** Obstacle-bearing map data; present iff {@link map} is set. */
  mapData?: MapData;
}

export interface OpenScenarioOptions {
  name?: string;
  seed?: number;
  blueUnits?: number;
  redUnits?: number;
  arenaWidth?: number;
  arenaHeight?: number;
  blueSpawns?: ReadonlyArray<SpawnPosition>;
  redSpawns?: ReadonlyArray<SpawnPosition>;
  maxTicks?: number;
}

/** Builds a deterministic, obstacle-free configurable squad scenario. */
export function createOpenScenario(options: OpenScenarioOptions = {}): Scenario {
  const blueUnits = teamSize(options.blueUnits ?? 3, 'blueUnits');
  const redUnits = teamSize(options.redUnits ?? 3, 'redUnits');
  const width = arenaSize(options.arenaWidth ?? 40, 'arenaWidth');
  const height = arenaSize(options.arenaHeight ?? 30, 'arenaHeight');
  const maxTicks = positiveInteger(options.maxTicks ?? DEFAULT_MAX_TICKS, 'maxTicks');
  const seed = safeInteger(options.seed ?? DEFAULT_SEED, 'seed');

  const blueSpawns = options.blueSpawns
    ? validateSpawns(options.blueSpawns, blueUnits, width, height, 'blueSpawns')
    : generatedSpawns(blueUnits, -width * 0.3, height);
  const redSpawns = options.redSpawns
    ? validateSpawns(options.redSpawns, redUnits, width, height, 'redSpawns')
    : generatedSpawns(redUnits, width * 0.3, height);
  validateSpacing([...blueSpawns, ...redSpawns], 'team spawns');

  return {
    name: options.name ?? `${blueUnits}-vs-${redUnits}-open`,
    seed,
    arena: { width, height },
    blueSpawns,
    redSpawns,
    respawn: false,
    buffs: false,
    maxTicks,
  };
}

export const THREE_VS_THREE_OPEN: Scenario = createOpenScenario({
  name: 'three-vs-three-open',
});

/** Options for a fight on a bundled map (real terrain + map spawn points). */
export interface MapScenarioOptions {
  name?: string;
  seed?: number;
  maxTicks?: number;
}

/**
 * Builds a scenario on a bundled map. Team sizes come from the map's spawn
 * lists; obstacles are loaded through the same MapLoader as the browser game.
 */
export function createMapScenario(mapId: string, options: MapScenarioOptions = {}): Scenario {
  const data = getMapData(mapId);
  const seed = safeInteger(options.seed ?? DEFAULT_SEED, 'seed');
  const maxTicks = positiveInteger(options.maxTicks ?? DEFAULT_MAX_TICKS, 'maxTicks');
  const blue = mapSpawns(mapId, Team.Player).map((s) => ({ x: s.x, y: s.y }));
  const red = mapSpawns(mapId, Team.Enemy).map((s) => ({ x: s.x, y: s.y }));
  if (blue.length === 0 || red.length === 0) {
    throw new RangeError(`map "${mapId}" must define both player and enemy spawns`);
  }
  return {
    name: options.name ?? `${data.name ?? mapId}`,
    seed,
    arena: { width: data.width, height: data.height },
    blueSpawns: blue,
    redSpawns: red,
    respawn: false,
    buffs: false,
    maxTicks,
    map: mapId,
    mapData: data,
  };
}

/** Builds the collision/LoS arena for a scenario (empty for open scenarios). */
export function buildArena(scenario: Scenario, ids: IdAllocator): Arena {
  if (!scenario.mapData) {
    return {
      width: scenario.arena.width,
      height: scenario.arena.height,
      obstacles: [],
      spawns: [],
    };
  }
  return new MapLoader(ids).build(scenario.mapData);
}

function generatedSpawns(count: number, x: number, height: number): SpawnPosition[] {
  if (count === 1) return [{ x, y: 0 }];
  const desiredSpan = (count - 1) * 5;
  const span = Math.min(desiredSpan, height - SPAWN_MARGIN * 4);
  return Array.from({ length: count }, (_, index) => ({
    x,
    y: -span / 2 + (span * index) / (count - 1),
  }));
}

function validateSpawns(
  spawns: ReadonlyArray<SpawnPosition>,
  count: number,
  width: number,
  height: number,
  name: string,
): SpawnPosition[] {
  if (spawns.length !== count) throw new RangeError(`${name} must contain ${count} positions`);
  const result = spawns.map((spawn, index) => {
    if (typeof spawn !== 'object' || spawn === null) {
      throw new RangeError(`${name}[${index}] must be an object`);
    }
    const x = finite(spawn.x, `${name}[${index}].x`);
    const y = finite(spawn.y, `${name}[${index}].y`);
    if (
      x < -width / 2 + SPAWN_MARGIN ||
      x > width / 2 - SPAWN_MARGIN ||
      y < -height / 2 + SPAWN_MARGIN ||
      y > height / 2 - SPAWN_MARGIN
    ) {
      throw new RangeError(`${name}[${index}] must be inside the arena`);
    }
    return { x, y };
  });
  validateSpacing(result, name);
  return result;
}

function validateSpacing(spawns: ReadonlyArray<SpawnPosition>, name: string): void {
  for (let i = 0; i < spawns.length; i++) {
    for (let j = i + 1; j < spawns.length; j++) {
      if (Math.hypot(spawns[i].x - spawns[j].x, spawns[i].y - spawns[j].y) < MIN_SPAWN_DISTANCE) {
        throw new RangeError(`${name} positions must not overlap`);
      }
    }
  }
}

function teamSize(value: number, name: string): number {
  const count = positiveInteger(value, name);
  if (count > MAX_TEAM_SIZE) throw new RangeError(`${name} must be at most ${MAX_TEAM_SIZE}`);
  return count;
}

function arenaSize(value: number, name: string): number {
  const size = finite(value, name);
  if (size < MIN_ARENA_SIZE || size > MAX_ARENA_SIZE) {
    throw new RangeError(`${name} must be in [${MIN_ARENA_SIZE}, ${MAX_ARENA_SIZE}]`);
  }
  return size;
}

function positiveInteger(value: number, name: string): number {
  const result = safeInteger(value, name);
  if (result <= 0) throw new RangeError(`${name} must be positive`);
  return result;
}

function safeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value)) throw new RangeError(`${name} must be a safe integer`);
  return value;
}

function finite(value: number, name: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
}
