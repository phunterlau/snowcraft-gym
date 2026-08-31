import type { Observation, UnitObservation } from '../observations/Observation';

export function observationWith(
  overrides: Partial<Pick<Observation, 'allies' | 'enemies' | 'arena'>> = {},
): Observation {
  const allies = overrides.allies ?? [unit(1, 'blue', -10, 0)];
  const enemies = overrides.enemies ?? [unit(100, 'red', 10, 0)];
  return {
    tick: 0,
    selfTeam: 'blue',
    simulationHz: 60,
    arena: overrides.arena ?? { width: 40, height: 30 },
    allies,
    enemies,
    projectiles: [],
    obstacles: [],
    match: {
      blueAlive: allies.filter(({ alive }) => alive).length,
      redAlive: enemies.filter(({ alive }) => alive).length,
    },
  };
}

function unit(id: number, team: UnitObservation['team'], x: number, y: number): UnitObservation {
  return {
    id,
    team,
    x,
    y,
    vx: 0,
    vy: 0,
    health: 100,
    maxHealth: 100,
    alive: true,
    state: 'idle',
    throwCooldown: 0,
    charge: 0,
  };
}
