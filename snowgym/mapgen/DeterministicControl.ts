import { Team, type MapData, type MapSpawnData } from '../../src/game/types';
import type { MapGenerationRequest } from './types';

/** Deterministic mirrored placement baseline for separating LLM value from map count alone. */
export function createMirroredControlMap(request: MapGenerationRequest): MapData {
  const target = Math.max(
    2,
    Math.min(
      request.objectBudget,
      Math.round(
        request.objectBudget *
          (request.density === 'sparse' ? 0.35 : request.density === 'medium' ? 0.6 : 0.85),
      ),
    ),
  );
  const objects: MapData['objects'] = [];
  const pairs = Math.floor(target / 2);
  for (let index = 0; index < pairs; index++) {
    const column = index % 3;
    const row = Math.floor(index / 3);
    const x = request.width * (0.08 + column * 0.09);
    const usableHeight = request.height - 8;
    const y = -usableHeight / 2 + ((row + 1) * usableHeight) / (Math.ceil(pairs / 3) + 1);
    const type = index % 3 === 0 ? 'rock' : 'tree';
    objects.push({ type, x: -x, y, radius: type === 'rock' ? 0.7 : 0.45 });
    objects.push({ type, x, y, radius: type === 'rock' ? 0.7 : 0.45 });
  }
  if (target % 2 === 1) objects.push({ type: 'fort', x: 0, y: 0, width: 4, height: 1 });
  return {
    name: `Deterministic ${request.topology} control`,
    width: request.width,
    height: request.height,
    objects,
    spawns: [
      ...controlSpawns(Team.Player, request.blueCapacity, -request.width / 2 + 3, request.height),
      ...controlSpawns(Team.Enemy, request.redCapacity, request.width / 2 - 3, request.height),
    ],
  };
}

function controlSpawns(team: Team, count: number, x: number, height: number): MapSpawnData[] {
  if (count === 1) return [{ team, x, y: 0 }];
  const span = Math.min(height - 4, (count - 1) * 3.5);
  return Array.from({ length: count }, (_, index) => ({
    team,
    x,
    y: -span / 2 + (span * index) / (count - 1),
  }));
}
