import type {
  ObstacleObservation,
  Observation,
  ProjectileObservation,
  UnitObservation,
} from '../observations/Observation';
import { STATE_HASH_VERSION } from '../protocol/Version';

const SCALE = 1_000_000_000;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const UINT64_MASK = 0xffffffffffffffffn;

/**
 * Hashes the detached public simulator state using a versioned canonical form.
 * This is a regression checksum, not a cryptographic integrity signature.
 */
export function hashObservation(observation: Observation): string {
  const canonical = canonicalObservation(observation);
  let hash = FNV_OFFSET;
  for (const byte of new TextEncoder().encode(canonical)) {
    hash ^= BigInt(byte);
    hash = (hash * FNV_PRIME) & UINT64_MASK;
  }
  return `fnv1a64:${hash.toString(16).padStart(16, '0')}`;
}

/** Canonical token stream shared with the Python client implementation. */
export function canonicalObservation(observation: Observation): string {
  const tokens: string[] = [
    STATE_HASH_VERSION,
    integer(observation.tick),
    observation.selfTeam,
    integer(observation.simulationHz),
    quantized(observation.arena.width),
    quantized(observation.arena.height),
  ];
  appendUnits(tokens, 'allies', observation.allies);
  appendUnits(tokens, 'enemies', observation.enemies);
  appendProjectiles(tokens, observation.projectiles);
  appendObstacles(tokens, observation.obstacles ?? []);
  tokens.push('match', integer(observation.match.blueAlive), integer(observation.match.redAlive));
  return tokens.join('|');
}

function appendUnits(tokens: string[], label: string, units: readonly UnitObservation[]): void {
  const ordered = [...units].sort((a, b) => a.id - b.id);
  tokens.push(label, integer(ordered.length));
  for (const unit of ordered) {
    tokens.push(
      integer(unit.id),
      unit.team,
      quantized(unit.x),
      quantized(unit.y),
      quantized(unit.vx),
      quantized(unit.vy),
      quantized(unit.health),
      quantized(unit.maxHealth),
      unit.alive ? '1' : '0',
      unit.state,
      quantized(unit.throwCooldown),
      quantized(unit.charge),
    );
  }
}

function appendProjectiles(tokens: string[], projectiles: readonly ProjectileObservation[]): void {
  const ordered = [...projectiles].sort((a, b) => a.id - b.id);
  tokens.push('projectiles', integer(ordered.length));
  for (const projectile of ordered) {
    tokens.push(
      integer(projectile.id),
      integer(projectile.ownerId),
      projectile.team,
      quantized(projectile.x),
      quantized(projectile.y),
      quantized(projectile.vx),
      quantized(projectile.vy),
      quantized(projectile.height),
      quantized(projectile.heightVelocity),
    );
  }
}

function appendObstacles(tokens: string[], obstacles: readonly ObstacleObservation[]): void {
  const ordered = [...obstacles].sort((a, b) => a.id - b.id);
  tokens.push('obstacles', integer(ordered.length));
  for (const obstacle of ordered) {
    tokens.push(
      integer(obstacle.id),
      obstacle.type,
      quantized(obstacle.x),
      quantized(obstacle.y),
      quantized(obstacle.halfWidth),
      quantized(obstacle.halfHeight),
      obstacle.blocksSight ? '1' : '0',
      obstacle.blocksProjectiles ? '1' : '0',
      obstacle.blocksMovement ? '1' : '0',
    );
  }
}

function quantized(value: number): string {
  if (!Number.isFinite(value)) throw new RangeError('state hash values must be finite');
  return Math.floor(value * SCALE + 0.5).toString();
}

function integer(value: number): string {
  if (!Number.isSafeInteger(value)) throw new RangeError('state hash integers must be safe');
  return value.toString();
}
