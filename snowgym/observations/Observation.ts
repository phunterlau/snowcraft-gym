import { SIM } from '../../src/game/config';
import { Team, type ObstacleType } from '../../src/game/types';
import type { World } from '../../src/game/World';
import type { Shape } from '../../src/physics/shapes';

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

/** Static terrain obstacle, normalized to a center + half-extents footprint. */
export interface ObstacleObservation {
  id: number;
  type: ObstacleType;
  x: number;
  y: number;
  /** Half-width (circle radius, or half the rect width). */
  halfWidth: number;
  /** Half-height (equals halfWidth for circles). */
  halfHeight: number;
  blocksSight: boolean;
  blocksProjectiles: boolean;
  blocksMovement: boolean;
}

export interface Observation {
  tick: number;
  selfTeam: SnowTeam;
  simulationHz: number;
  arena: { width: number; height: number };
  allies: UnitObservation[];
  enemies: UnitObservation[];
  projectiles: ProjectileObservation[];
  /** Terrain obstacles in deterministic id order; empty on open arenas. */
  obstacles: ObstacleObservation[];
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

  const obstacles = world.arena.obstacles
    .map(
      (obstacle): ObstacleObservation => ({
        id: obstacle.id,
        type: obstacle.type,
        ...footprint(obstacle.collision),
        blocksSight: obstacle.blocksSight,
        blocksProjectiles: obstacle.blocksProjectiles,
        blocksMovement: obstacle.blocksMovement,
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
    obstacles,
    match: {
      blueAlive: world.countLiving(Team.Player),
      redAlive: world.countLiving(Team.Enemy),
    },
  };
}

function snowTeam(team: Team): SnowTeam {
  return team === Team.Player ? 'blue' : 'red';
}

/** Normalizes a collision shape to a center + half-extents footprint. */
function footprint(shape: Shape): { x: number; y: number; halfWidth: number; halfHeight: number } {
  if (shape.kind === 'circle') {
    return { x: shape.x, y: shape.y, halfWidth: shape.radius, halfHeight: shape.radius };
  }
  if (shape.kind === 'rect') {
    return { x: shape.x, y: shape.y, halfWidth: shape.halfW, halfHeight: shape.halfH };
  }
  // Capsule: use the segment midpoint and its axis-aligned bounding box.
  return {
    x: (shape.x1 + shape.x2) / 2,
    y: (shape.y1 + shape.y2) / 2,
    halfWidth: Math.abs(shape.x2 - shape.x1) / 2 + shape.radius,
    halfHeight: Math.abs(shape.y2 - shape.y1) / 2 + shape.radius,
  };
}
