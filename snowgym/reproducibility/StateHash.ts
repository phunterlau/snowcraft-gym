import type {
  ObstacleObservation,
  Observation,
  ProjectileObservation,
  UnitObservation,
} from '../observations/Observation';
import {
  LEGACY_STATE_HASH_VERSION,
  STATE_HASH_VERSION,
  type StateHashVersion,
} from '../protocol/Version';

const SCALE = 1_000_000_000;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const UINT64_MASK = 0xffffffffffffffffn;

/**
 * Hashes the detached public simulator state using a versioned canonical form.
 * This is a regression checksum, not a cryptographic integrity signature.
 */
export function hashObservation(
  observation: Observation,
  version: StateHashVersion = STATE_HASH_VERSION,
): string {
  const canonical = canonicalObservation(observation, version);
  let hash = FNV_OFFSET;
  for (const byte of new TextEncoder().encode(canonical)) {
    hash ^= BigInt(byte);
    hash = (hash * FNV_PRIME) & UINT64_MASK;
  }
  return `fnv1a64:${hash.toString(16).padStart(16, '0')}`;
}

/** Canonical token stream shared with the Python client implementation. */
export function canonicalObservation(
  observation: Observation,
  version: StateHashVersion = STATE_HASH_VERSION,
): string {
  if (version === LEGACY_STATE_HASH_VERSION) return canonicalObservationV1(observation);
  const decision = observation.decision ?? {
    hz: observation.simulationHz,
    dt: 1 / observation.simulationHz,
    maxTicks: Number.MAX_SAFE_INTEGER,
    remainingFraction: 1,
  };
  const tokens: string[] = [
    STATE_HASH_VERSION,
    observation.observationVersion ?? 'legacy',
    integer(observation.tick),
    observation.selfTeam,
    integer(observation.simulationHz),
    quantized(observation.arena.width),
    quantized(observation.arena.height),
    'decision',
    quantized(decision.hz),
    quantized(decision.dt),
    integer(decision.maxTicks),
    quantized(decision.remainingFraction),
  ];
  appendUnits(tokens, 'allies', observation.allies, true);
  appendUnits(tokens, 'enemies', observation.enemies, true);
  appendProjectiles(tokens, observation.projectiles, true);
  appendObstacles(tokens, observation.obstacles ?? []);
  tokens.push('match', integer(observation.match.blueAlive), integer(observation.match.redAlive));
  return tokens.join('|');
}

export function canonicalObservationV1(observation: Observation): string {
  const tokens: string[] = [
    LEGACY_STATE_HASH_VERSION,
    integer(observation.tick),
    observation.selfTeam,
    integer(observation.simulationHz),
    quantized(observation.arena.width),
    quantized(observation.arena.height),
  ];
  appendUnits(tokens, 'allies', observation.allies, false);
  appendUnits(tokens, 'enemies', observation.enemies, false);
  appendProjectiles(tokens, observation.projectiles, false);
  appendObstacles(tokens, observation.obstacles ?? []);
  tokens.push('match', integer(observation.match.blueAlive), integer(observation.match.redAlive));
  return tokens.join('|');
}

function appendUnits(
  tokens: string[],
  label: string,
  units: readonly UnitObservation[],
  includeController: boolean,
): void {
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
    if (includeController) {
      appendPoint(tokens, 'moveTarget', unit.moveTarget);
      appendPoint(tokens, 'steeringTarget', unit.steeringTarget);
      appendPoint(tokens, 'aimDirection', unit.aimDirection);
      tokens.push(
        quantized(unit.stunRemaining ?? 0),
        quantized(unit.throwPhaseRemaining ?? 0),
        quantized(unit.immunityRemaining ?? 0),
        quantized(unit.speedRemaining ?? 0),
      );
    }
  }
}

function appendProjectiles(
  tokens: string[],
  projectiles: readonly ProjectileObservation[],
  includeAge: boolean,
): void {
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
    if (includeAge) tokens.push(quantized(projectile.age ?? 0));
  }
}

function appendPoint(
  tokens: string[],
  label: string,
  value: { readonly x: number; readonly y: number } | null | undefined,
): void {
  if (value === null || value === undefined) {
    tokens.push(label, '0');
    return;
  }
  tokens.push(label, '1', quantized(value.x), quantized(value.y));
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
