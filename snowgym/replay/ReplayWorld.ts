import { EventBus } from '../../src/core/EventBus';
import { createEmptyArena } from '../../src/game/Arena';
import { createPlayer } from '../../src/game/Player';
import { createSnowball } from '../../src/game/Snowball';
import { PLAYER, SNOWBALL } from '../../src/game/config';
import {
  PlayerState,
  Team,
  type Obstacle,
  type Player,
  type Snowball,
} from '../../src/game/types';
import { circle, rect } from '../../src/physics/shapes';
import { World } from '../../src/game/World';
import { Vector2 } from '../../src/utils/Vector2';
import type {
  ObstacleObservation,
  ProjectileObservation,
  UnitObservation,
} from '../observations/Observation';
import type { ReplayRecording } from './ReplayRecording';

export interface ReplayWorld {
  world: World;
  events: EventBus;
}

export function createReplayWorld(recording: ReplayRecording): ReplayWorld {
  const arena = recording.frames[0].arena;
  const world = new World(createEmptyArena(arena.width, arena.height), recording.seed);
  // Terrain is static over an episode; rebuild render obstacles from the first frame.
  world.arena.obstacles = (recording.frames[0].obstacles ?? []).map(buildReplayObstacle);
  world.playerLives = 0;
  world.playerLivesMax = 0;
  const events = new EventBus();
  applyReplayTick(world, recording, recording.frames[0].tick);
  return { world, events };
}

function buildReplayObstacle(observation: ObstacleObservation): Obstacle {
  const round = observation.halfWidth === observation.halfHeight;
  const collision = round
    ? circle(observation.x, observation.y, observation.halfWidth)
    : rect(observation.x, observation.y, observation.halfWidth, observation.halfHeight);
  return {
    id: observation.id,
    type: observation.type,
    position: new Vector2(observation.x, observation.y),
    collision,
    cover: null,
    blocksSight: observation.blocksSight,
    blocksProjectiles: observation.blocksProjectiles,
    blocksMovement: observation.blocksMovement,
  };
}

/** Applies an interpolated, render-only snapshot without advancing simulation. */
export function applyReplayTick(world: World, recording: ReplayRecording, tick: number): void {
  const finalTick = recording.frames.at(-1)?.tick ?? 0;
  const boundedTick = Math.min(Math.max(tick, recording.frames[0].tick), finalTick);
  const upperIndex = recording.frames.findIndex((frame) => frame.tick >= boundedTick);
  const nextIndex = upperIndex === -1 ? recording.frames.length - 1 : upperIndex;
  const previousIndex = Math.max(0, nextIndex - 1);
  const previous = recording.frames[previousIndex];
  const next = recording.frames[nextIndex];
  const span = Math.max(1, next.tick - previous.tick);
  const alpha = previousIndex === nextIndex ? 0 : (boundedTick - previous.tick) / span;

  world.time = boundedTick / recording.simulationHz;
  syncPlayers(
    world,
    [...previous.allies, ...previous.enemies],
    [...next.allies, ...next.enemies],
    alpha,
  );
  syncSnowballs(world, previous.projectiles, next.projectiles, alpha, world.time);
}

function syncPlayers(
  world: World,
  previousUnits: UnitObservation[],
  nextUnits: UnitObservation[],
  alpha: number,
): void {
  const previousById = new Map(previousUnits.map((unit) => [unit.id, unit]));
  const nextById = new Map(nextUnits.map((unit) => [unit.id, unit]));
  const ids = new Set([...previousById.keys(), ...nextById.keys()]);

  for (const id of ids) {
    const from = previousById.get(id) ?? nextById.get(id);
    const to = nextById.get(id) ?? from;
    if (!from || !to) continue;
    let player = world.getPlayer(id);
    if (!player) {
      player = createPlayer(id, team(from.team), from.x, from.y);
      world.players.push(player);
    }
    applyUnit(player, from, to, alpha, world.time);
  }
}

function applyUnit(
  player: Player,
  from: UnitObservation,
  to: UnitObservation,
  alpha: number,
  time: number,
): void {
  player.position.set(lerp(from.x, to.x, alpha), lerp(from.y, to.y, alpha));
  player.velocity.set(lerp(from.vx, to.vx, alpha), lerp(from.vy, to.vy, alpha));
  if (player.velocity.lengthSq() > 1e-6) player.rotation = player.velocity.angle();
  player.health = lerp(from.health, to.health, alpha);
  player.maxHealth = to.maxHealth;
  player.alive = alpha < 1 ? from.alive : to.alive;
  player.state = (alpha < 1 ? from.state : to.state) as PlayerState;
  player.currentAnimation = animationFor(player.state);
  player.animationTime = time;
  player.throwCooldown = lerp(from.throwCooldown, to.throwCooldown, alpha);
  player.throwCharge = lerp(from.charge, to.charge, alpha);
  player.radius = PLAYER.radius;
  player.selected = false;
}

function syncSnowballs(
  world: World,
  previousProjectiles: ProjectileObservation[],
  nextProjectiles: ProjectileObservation[],
  alpha: number,
  time: number,
): void {
  const previousById = new Map(
    previousProjectiles.map((projectile) => [projectile.id, projectile]),
  );
  const nextById = new Map(nextProjectiles.map((projectile) => [projectile.id, projectile]));
  const visible = new Set<number>();

  for (const [id, to] of nextById) {
    const from = previousById.get(id) ?? to;
    let snowball = world.snowballs.find((candidate) => candidate.id === id);
    if (!snowball) {
      snowball = createSnowball();
      (snowball as { id: number }).id = id;
      world.snowballs.push(snowball);
    }
    applyProjectile(snowball, from, to, alpha, time);
    visible.add(id);
  }

  for (const snowball of world.snowballs) snowball.alive = visible.has(snowball.id);
}

function applyProjectile(
  snowball: Snowball,
  from: ProjectileObservation,
  to: ProjectileObservation,
  alpha: number,
  time: number,
): void {
  snowball.ownerId = to.ownerId;
  snowball.team = team(to.team);
  snowball.position.set(lerp(from.x, to.x, alpha), lerp(from.y, to.y, alpha));
  snowball.velocity.set(lerp(from.vx, to.vx, alpha), lerp(from.vy, to.vy, alpha));
  snowball.height = Math.max(0, lerp(from.height, to.height, alpha));
  snowball.heightVelocity = lerp(from.heightVelocity, to.heightVelocity, alpha);
  snowball.damage = SNOWBALL.damage;
  snowball.radius = SNOWBALL.radius;
  snowball.age = time;
  snowball.alive = true;
}

function team(value: 'blue' | 'red'): Team {
  return value === 'blue' ? Team.Player : Team.Enemy;
}

function animationFor(state: PlayerState): Player['currentAnimation'] {
  if (state === PlayerState.Moving) return 'walk';
  if (state === PlayerState.PreparingThrow || state === PlayerState.Throwing) return 'throw';
  if (state === PlayerState.Hit || state === PlayerState.Frozen) return 'hit';
  if (state === PlayerState.Defeated) return 'defeated';
  return 'idle';
}

function lerp(from: number, to: number, alpha: number): number {
  return from + (to - from) * alpha;
}
