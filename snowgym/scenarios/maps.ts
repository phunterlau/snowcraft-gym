import { Team, type MapData, type SpawnPoint } from '../../src/game/types';

/** Runtime registry of bundled SnowCraft maps, keyed by their public/maps id. */
export interface MapDefinition {
  /** Map id matching the file name under `public/maps/` (e.g. "arena1.json"). */
  id: string;
  data: MapData;
}

/* Generated from public/maps/*.json so the headless server and Vitest stay
   free of fetch; the Vite browser game loads the same JSON over HTTP.
   Regenerate if the map files change. */
const MAP_DATA: Record<string, MapData> = {
  'arena1.json': {
    name: 'Snowy Clearing',
    width: 40,
    height: 30,
    objects: [
      { type: 'tree', x: -8, y: -6 },
      { type: 'tree', x: -6, y: 7 },
      { type: 'tree', x: 9, y: -8 },
      { type: 'tree', x: 7, y: 6 },
      { type: 'tree', x: 0, y: 11 },
      { type: 'tree', x: 1, y: -11 },
      { type: 'rock', x: -3, y: -2 },
      { type: 'rock', x: 4, y: 2 },
      { type: 'rock', x: -12, y: 3 },
      { type: 'rock', x: 12, y: -3 },
      { type: 'fort', x: -14, y: 0, width: 1.2, height: 4 },
      { type: 'fort', x: 14, y: 0, width: 1.2, height: 4 },
      { type: 'fort', x: 0, y: 0, width: 5, height: 1.2 },
      { type: 'fence', x: -6, y: -10, width: 5, height: 0.3 },
      { type: 'fence', x: 6, y: 10, width: 5, height: 0.3 },
      { type: 'prop', x: -10, y: -12 },
      { type: 'prop', x: 11, y: 12 },
      { type: 'prop', x: -16, y: 10 },
    ],
    spawns: [
      { team: Team.Player, x: -16, y: -6 },
      { team: Team.Player, x: -17, y: 0 },
      { team: Team.Player, x: -16, y: 6 },
      { team: Team.Enemy, x: 16, y: -6 },
      { team: Team.Enemy, x: 17, y: 0 },
      { team: Team.Enemy, x: 16, y: 6 },
    ],
  },
  'arena2.json': {
    name: 'Frozen Pond',
    width: 44,
    height: 30,
    objects: [
      { type: 'rock', x: 0, y: 0, radius: 1.2 },
      { type: 'rock', x: -6, y: -5 },
      { type: 'rock', x: 6, y: 5 },
      { type: 'rock', x: -6, y: 5 },
      { type: 'rock', x: 6, y: -5 },
      { type: 'tree', x: -14, y: -9 },
      { type: 'tree', x: 14, y: 9 },
      { type: 'tree', x: -14, y: 9 },
      { type: 'tree', x: 14, y: -9 },
      { type: 'tree', x: 0, y: 12 },
      { type: 'tree', x: 0, y: -12 },
      { type: 'fort', x: -18, y: 0, width: 1.2, height: 6 },
      { type: 'fort', x: 18, y: 0, width: 1.2, height: 6 },
      { type: 'fence', x: -9, y: 0, width: 0.3, height: 6 },
      { type: 'fence', x: 9, y: 0, width: 0.3, height: 6 },
      { type: 'prop', x: -20, y: -12 },
      { type: 'prop', x: 20, y: 12 },
      { type: 'prop', x: 0, y: 6 },
    ],
    spawns: [
      { team: Team.Player, x: -20, y: -7 },
      { team: Team.Player, x: -21, y: 0 },
      { team: Team.Player, x: -20, y: 7 },
      { team: Team.Enemy, x: 20, y: -7 },
      { team: Team.Enemy, x: 21, y: 0 },
      { team: Team.Enemy, x: 20, y: 7 },
    ],
  },
  'arena3.json': {
    name: 'Village Skirmish',
    width: 46,
    height: 32,
    objects: [
      { type: 'fort', x: -8, y: -8, width: 4, height: 4 },
      { type: 'fort', x: 8, y: 8, width: 4, height: 4 },
      { type: 'fort', x: 8, y: -8, width: 3, height: 3 },
      { type: 'fort', x: -8, y: 8, width: 3, height: 3 },
      { type: 'fence', x: 0, y: -4, width: 8, height: 0.3 },
      { type: 'fence', x: 0, y: 4, width: 8, height: 0.3 },
      { type: 'tree', x: -16, y: 0 },
      { type: 'tree', x: 16, y: 0 },
      { type: 'tree', x: 0, y: 13 },
      { type: 'tree', x: 0, y: -13 },
      { type: 'rock', x: -3, y: 0 },
      { type: 'rock', x: 3, y: 0 },
      { type: 'prop', x: -20, y: -13 },
      { type: 'prop', x: 20, y: 13 },
      { type: 'prop', x: -20, y: 13 },
      { type: 'prop', x: 20, y: -13 },
    ],
    spawns: [
      { team: Team.Player, x: -21, y: -8 },
      { team: Team.Player, x: -22, y: 0 },
      { team: Team.Player, x: -21, y: 8 },
      { team: Team.Enemy, x: 21, y: -8 },
      { team: Team.Enemy, x: 22, y: 0 },
      { team: Team.Enemy, x: 21, y: 8 },
    ],
  },
  'arena4.json': {
    name: 'Pine Forest',
    width: 48,
    height: 34,
    objects: [
      { type: 'tree', x: 0, y: 0, radius: 0.8 },
      { type: 'tree', x: -16, y: -12 },
      { type: 'tree', x: 16, y: 12 },
      { type: 'tree', x: -16, y: 12 },
      { type: 'tree', x: 16, y: -12 },
      { type: 'tree', x: -11, y: -14 },
      { type: 'tree', x: 11, y: 14 },
      { type: 'tree', x: -11, y: 14 },
      { type: 'tree', x: 11, y: -14 },
      { type: 'tree', x: -13, y: -7, radius: 0.5 },
      { type: 'tree', x: 13, y: 7, radius: 0.5 },
      { type: 'tree', x: -13, y: 7, radius: 0.5 },
      { type: 'tree', x: 13, y: -7, radius: 0.5 },
      { type: 'tree', x: -7, y: -10 },
      { type: 'tree', x: 7, y: 10 },
      { type: 'tree', x: -7, y: 10 },
      { type: 'tree', x: 7, y: -10 },
      { type: 'tree', x: -8, y: -3 },
      { type: 'tree', x: 8, y: 3 },
      { type: 'tree', x: -8, y: 3 },
      { type: 'tree', x: 8, y: -3 },
      { type: 'tree', x: -3, y: -13, radius: 0.5 },
      { type: 'tree', x: 3, y: 13, radius: 0.5 },
      { type: 'tree', x: -3, y: 13, radius: 0.5 },
      { type: 'tree', x: 3, y: -13, radius: 0.5 },
      { type: 'tree', x: 0, y: -7 },
      { type: 'tree', x: 0, y: 7 },
      { type: 'tree', x: -14, y: 0 },
      { type: 'tree', x: 14, y: 0 },
      { type: 'tree', x: -3, y: -4, radius: 0.5 },
      { type: 'tree', x: 3, y: 4, radius: 0.5 },
      { type: 'tree', x: -3, y: 4, radius: 0.5 },
      { type: 'tree', x: 3, y: -4, radius: 0.5 },
      { type: 'rock', x: -6, y: 0 },
      { type: 'rock', x: 6, y: 0 },
      { type: 'rock', x: -18, y: 5 },
      { type: 'rock', x: 18, y: -5 },
      { type: 'prop', x: -20, y: 13 },
      { type: 'prop', x: 20, y: -13 },
      { type: 'prop', x: -20, y: -13 },
      { type: 'prop', x: 20, y: 13 },
    ],
    spawns: [
      { team: Team.Player, x: -21, y: -8 },
      { team: Team.Player, x: -22, y: 0 },
      { team: Team.Player, x: -21, y: 8 },
      { team: Team.Enemy, x: 21, y: -8 },
      { team: Team.Enemy, x: 22, y: 0 },
      { team: Team.Enemy, x: 21, y: 8 },
    ],
  },
  'arena5.json': {
    name: 'Schoolyard Scramble',
    width: 46,
    height: 32,
    objects: [
      { type: 'fort', x: -7, y: 12.5, width: 8, height: 4 },
      { type: 'fort', x: 7, y: 12.5, width: 8, height: 4 },
      { type: 'fence', x: 0, y: 6, width: 18, height: 0.3 },
      { type: 'fence', x: 0, y: -6, width: 18, height: 0.3 },
      { type: 'fort', x: -10, y: 2.5, width: 0.6, height: 1 },
      { type: 'fort', x: -10, y: -2.5, width: 0.6, height: 1 },
      { type: 'fort', x: 10, y: 2.5, width: 0.6, height: 1 },
      { type: 'fort', x: 10, y: -2.5, width: 0.6, height: 1 },
      { type: 'rock', x: -5, y: 0, radius: 0.7 },
      { type: 'rock', x: 5, y: 0, radius: 0.7 },
      { type: 'tree', x: -18, y: 12 },
      { type: 'tree', x: 18, y: 12 },
      { type: 'prop', x: 0, y: -13 },
      { type: 'prop', x: -19, y: -12 },
      { type: 'prop', x: 19, y: -12 },
      { type: 'prop', x: -20, y: 4 },
      { type: 'prop', x: 20, y: 4 },
      { type: 'prop', x: -8, y: -11 },
      { type: 'prop', x: 8, y: -11 },
    ],
    spawns: [
      { team: Team.Player, x: -20, y: -8 },
      { team: Team.Player, x: -21, y: 0 },
      { team: Team.Player, x: -20, y: 8 },
      { team: Team.Enemy, x: 20, y: -8 },
      { team: Team.Enemy, x: 21, y: 0 },
      { team: Team.Enemy, x: 20, y: 8 },
    ],
  },
};

export const MAP_IDS: readonly string[] = Object.keys(MAP_DATA);

/** Largest obstacle count across registered maps; fixes the observation tensor. */
export const MAX_MAP_OBSTACLES = Math.max(
  ...Object.values(MAP_DATA).map((map) => map.objects.length),
);

export function isMapId(value: unknown): value is string {
  return typeof value === 'string' && value in MAP_DATA;
}

export function getMapData(id: string): MapData {
  const data = MAP_DATA[id];
  if (!data) throw new RangeError(`unknown map "${id}"; expected one of: ${MAP_IDS.join(', ')}`);
  return data;
}

/** Spawn points for one team, in map order. */
export function mapSpawns(id: string, team: Team): SpawnPoint[] {
  return getMapData(id).spawns?.filter((spawn) => spawn.team === team) ?? [];
}
