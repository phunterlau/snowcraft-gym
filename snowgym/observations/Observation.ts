import { SIM } from '../../src/game/config';
import { Team } from '../../src/game/types';
import type { World } from '../../src/game/World';

export type SnowTeam = 'blue' | 'red';
export type UnitState =
  | 'idle'
  | 'moving'
  | 'preparingThrow'
  | 'throwing'
  | 'recovering'
  | 'hit'
  | 'frozen'
  | 'defeated';

export interface UnitObservation {
  id: number;
  team: SnowTeam;
  x: number;
  y: number;
  vx: number;
  vy: number;
  health: number;
  maxHealth: number;
  alive: boolean;
  state: UnitState;
  throwCooldown: number;
  charge: number;
}

export interface ProjectileObservation {
  id: number;
  ownerId: number;
  team: SnowTeam;
  x: number;
  y: number;
  vx: number;
  vy: number;
  height: number;
  heightVelocity: number;
}

export interface Observation {
  tick: number;
  selfTeam: SnowTeam;
  simulationHz: number;
  arena: { width: number; height: number };
  allies: UnitObservation[];
  enemies: UnitObservation[];
  projectiles: ProjectileObservation[];
  match: {
    blueAlive: number;
    redAlive: number;
  };
}

/** Returns a detached, deterministic snapshot of simulation state. */
export function observeWorld(world: World, selfTeam: Team, tick: number): Observation {
  const units = world.players
    .map(
      (player): UnitObservation => ({
        id: player.id,
        team: snowTeam(player.team),
        x: player.position.x,
        y: player.position.y,
        vx: player.velocity.x,
        vy: player.velocity.y,
        health: player.health,
        maxHealth: player.maxHealth,
        alive: player.alive,
        state: player.state,
        throwCooldown: player.throwCooldown,
        charge: player.throwCharge,
      }),
    )
    .sort((a, b) => a.id - b.id);

  const projectiles = world.snowballs
    .filter((snowball) => snowball.alive)
    .map(
      (snowball): ProjectileObservation => ({
        id: snowball.id,
        ownerId: snowball.ownerId,
        team: snowTeam(snowball.team),
        x: snowball.position.x,
        y: snowball.position.y,
        vx: snowball.velocity.x,
        vy: snowball.velocity.y,
        height: snowball.height,
        heightVelocity: snowball.heightVelocity,
      }),
    )
    .sort((a, b) => a.id - b.id);

  return {
    tick,
    selfTeam: snowTeam(selfTeam),
    simulationHz: SIM.hz,
    arena: { width: world.arena.width, height: world.arena.height },
    allies: units.filter((unit) => unit.team === snowTeam(selfTeam)),
    enemies: units.filter((unit) => unit.team !== snowTeam(selfTeam)),
    projectiles,
    match: {
      blueAlive: world.countLiving(Team.Player),
      redAlive: world.countLiving(Team.Enemy),
    },
  };
}

function snowTeam(team: Team): SnowTeam {
  return team === Team.Player ? 'blue' : 'red';
}
