import { SIM, THROW } from '../../src/game/config';
import { PlayerState, Team, type ObstacleType, type Player } from '../../src/game/types';
import type { World } from '../../src/game/World';
import type { Shape } from '../../src/physics/shapes';

export type SnowTeam = 'blue' | 'red';
export const OBSERVATION_VERSION = 'snowgym.observation.v1' as const;
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
  moveTarget?: { x: number; y: number } | null;
  steeringTarget?: { x: number; y: number } | null;
  aimDirection?: { x: number; y: number };
  stunRemaining?: number;
  throwPhaseRemaining?: number;
  immunityRemaining?: number;
  speedRemaining?: number;
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
  age?: number;
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
  observationVersion?: typeof OBSERVATION_VERSION;
  tick: number;
  selfTeam: SnowTeam;
  simulationHz: number;
  arena: { width: number; height: number };
  allies: UnitObservation[];
  enemies: UnitObservation[];
  projectiles: ProjectileObservation[];
  /** Terrain obstacles in deterministic id order; empty on open arenas. */
  obstacles: ObstacleObservation[];
  decision?: {
    hz: number;
    dt: number;
    maxTicks: number;
    remainingFraction: number;
  };
  match: {
    blueAlive: number;
    redAlive: number;
  };
}

export interface ObservationContext {
  readonly decisionHz?: number;
  readonly maxTicks?: number;
  readonly steeringTarget?: (player: Player) => { readonly x: number; readonly y: number } | null;
}

/** Returns a detached, deterministic snapshot of simulation state. */
export function observeWorld(
  world: World,
  selfTeam: Team,
  tick: number,
  context: ObservationContext = {},
): Observation {
  const decisionHz = context.decisionHz ?? SIM.hz;
  const maxTicks = context.maxTicks ?? Number.MAX_SAFE_INTEGER;
  if (!Number.isFinite(decisionHz) || decisionHz <= 0 || decisionHz > SIM.hz) {
    throw new RangeError('observation decisionHz must be in (0, simulationHz]');
  }
  if (!Number.isSafeInteger(maxTicks) || maxTicks <= 0) {
    throw new RangeError('observation maxTicks must be a positive safe integer');
  }
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
        moveTarget: point(player.moveTarget),
        steeringTarget: point(context.steeringTarget?.(player) ?? player.moveTarget),
        aimDirection: { x: player.aimDirection.x, y: player.aimDirection.y },
        stunRemaining: player.stunTimer,
        throwPhaseRemaining: throwPhaseRemaining(player),
        immunityRemaining: player.immunityTimer,
        speedRemaining: player.speedTimer,
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
        age: snowball.age,
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
    observationVersion: OBSERVATION_VERSION,
    tick,
    selfTeam: snowTeam(selfTeam),
    simulationHz: SIM.hz,
    arena: { width: world.arena.width, height: world.arena.height },
    allies: units.filter((unit) => unit.team === snowTeam(selfTeam)),
    enemies: units.filter((unit) => unit.team !== snowTeam(selfTeam)),
    projectiles,
    obstacles,
    decision: {
      hz: decisionHz,
      dt: 1 / decisionHz,
      maxTicks,
      remainingFraction: Math.min(1, Math.max(0, (maxTicks - tick) / maxTicks)),
    },
    match: {
      blueAlive: world.countLiving(Team.Player),
      redAlive: world.countLiving(Team.Enemy),
    },
  };
}

function point(value: { readonly x: number; readonly y: number } | null): { x: number; y: number } | null {
  return value === null ? null : { x: value.x, y: value.y };
}

function throwPhaseRemaining(player: Player): number {
  if (player.state === PlayerState.Throwing) return Math.max(0, THROW.windup - player.throwTimer);
  if (player.state === PlayerState.Recovering) return Math.max(0, THROW.recovery - player.throwTimer);
  return 0;
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
