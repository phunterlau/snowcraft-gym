import { Team, type MapData } from '../../src/game/types';
import { DEFAULT_MAX_TICKS, type Scenario } from '../scenarios/Scenario';

/** Builds a headless scenario from validated map data without registering it as a bundled map. */
export function createGeneratedMapScenario(
  map: MapData,
  options: { seed?: number; maxTicks?: number; swappedSpawns?: boolean; name?: string } = {},
): Scenario {
  if (!map.spawns) throw new RangeError('generated map must have explicit spawns');
  const swapped = options.swappedSpawns ?? false;
  const blueSpawns = map.spawns
    .filter((spawn) => spawn.team === (swapped ? Team.Enemy : Team.Player))
    .map(({ x, y }) => ({ x, y }));
  const redSpawns = map.spawns
    .filter((spawn) => spawn.team === (swapped ? Team.Player : Team.Enemy))
    .map(({ x, y }) => ({ x, y }));
  return {
    name: options.name ?? `${map.name ?? 'generated-map'}${swapped ? '-swapped' : ''}`,
    seed: options.seed ?? 42,
    arena: { width: map.width, height: map.height },
    blueSpawns,
    redSpawns,
    respawn: false,
    buffs: false,
    maxTicks: options.maxTicks ?? DEFAULT_MAX_TICKS,
    mapData: map,
  };
}
