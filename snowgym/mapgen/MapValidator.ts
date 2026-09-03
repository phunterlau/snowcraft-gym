import { createHash } from 'node:crypto';
import { IdAllocator } from '../../src/ecs/Entity';
import { PLAYER } from '../../src/game/config';
import { MapLoader } from '../../src/game/MapLoader';
import { Team, type MapData, type MapObjectData, type MapSpawnData } from '../../src/game/types';
import { PathGrid } from '../../src/physics/Pathfinding';
import {
  MAP_CANDIDATE_VERSION,
  MAP_VALIDATION_VERSION,
  type MapCandidate,
  type MapGenerationRequest,
  type MapMetrics,
  type MapValidationFinding,
  type MapValidationReport,
} from './types';

export const MAX_GENERATED_OBSTACLES = 64;
export const MIN_GENERATED_ARENA_SIZE = 12;
export const MAX_GENERATED_ARENA_SIZE = 120;

const OBJECT_KEYS = new Set(['type', 'x', 'y', 'radius', 'width', 'height']);
const SPAWN_KEYS = new Set(['team', 'x', 'y']);
const MAP_KEYS = new Set(['name', 'width', 'height', 'objects', 'spawns']);
const CANDIDATE_KEYS = new Set(['schemaVersion', 'intentSummary', 'map']);

export function parseMapCandidate(value: unknown): MapCandidate {
  const candidate = record(value, 'candidate');
  rejectUnknown(candidate, CANDIDATE_KEYS, 'candidate');
  if (candidate.schemaVersion !== MAP_CANDIDATE_VERSION) {
    throw new RangeError(`candidate.schemaVersion must be ${MAP_CANDIDATE_VERSION}`);
  }
  const intentSummary = nonEmptyString(candidate.intentSummary, 'candidate.intentSummary', 500);
  return { schemaVersion: MAP_CANDIDATE_VERSION, intentSummary, map: parseMapData(candidate.map) };
}

export function parseMapData(value: unknown): MapData {
  const source = record(value, 'map');
  rejectUnknown(source, MAP_KEYS, 'map');
  const name = nonEmptyString(source.name, 'map.name', 100);
  const width = finite(source.width, 'map.width');
  const height = finite(source.height, 'map.height');
  if (!Array.isArray(source.objects)) throw new RangeError('map.objects must be an array');
  if (!Array.isArray(source.spawns)) throw new RangeError('map.spawns must be an array');
  return {
    name,
    width,
    height,
    objects: source.objects.map((item, index) => parseObject(item, index)),
    spawns: source.spawns.map((item, index) => parseSpawn(item, index)),
  };
}

export function validateGeneratedMap(
  input: unknown,
  request?: MapGenerationRequest,
): MapValidationReport {
  const findings: MapValidationFinding[] = [];
  let map: MapData;
  try {
    map = parseMapData(input);
  } catch (error) {
    findings.push(finding('error', 'schema_invalid', '$', errorMessage(error)));
    return emptyReport(findings);
  }

  const canonicalMap = canonicalizeMap(map);
  validateDimensions(canonicalMap, findings);
  validateObjects(canonicalMap, findings);
  validateSpawns(canonicalMap, findings);
  if (request) validateAgainstRequest(canonicalMap, request, findings);

  let metrics: MapMetrics | null = null;
  if (!findings.some((item) => item.severity === 'error')) {
    metrics = analyzeMap(canonicalMap, findings, request?.symmetry ?? 'mirror');
    if (request) validateQualityAgainstRequest(metrics, request, findings);
  }
  const valid = !findings.some((item) => item.severity === 'error');
  return {
    schemaVersion: MAP_VALIDATION_VERSION,
    valid,
    mapDigest: valid ? digestJson(canonicalMap) : null,
    findings,
    metrics,
    ...(valid ? { canonicalMap } : {}),
  };
}

export function canonicalizeMap(map: MapData): MapData {
  const objects = map.objects.map((object) => canonicalObject(object)).sort(compareObjects);
  const spawns = [...(map.spawns ?? [])]
    .map(({ team, x, y }) => ({ team, x: normalizedNumber(x), y: normalizedNumber(y) }))
    .sort((a, b) => a.team.localeCompare(b.team) || a.x - b.x || a.y - b.y);
  return {
    name: map.name?.trim(),
    width: normalizedNumber(map.width),
    height: normalizedNumber(map.height),
    objects,
    spawns,
  };
}

export function digestJson(value: unknown): string {
  return `sha256:${createHash('sha256').update(stableJson(value)).digest('hex')}`;
}

function validateDimensions(map: MapData, findings: MapValidationFinding[]): void {
  for (const [name, value] of [
    ['width', map.width],
    ['height', map.height],
  ] as const) {
    if (value < MIN_GENERATED_ARENA_SIZE || value > MAX_GENERATED_ARENA_SIZE) {
      findings.push(
        finding(
          'error',
          'arena_size_out_of_range',
          `map.${name}`,
          `${name} must be in [${MIN_GENERATED_ARENA_SIZE}, ${MAX_GENERATED_ARENA_SIZE}]`,
        ),
      );
    }
  }
}

function validateObjects(map: MapData, findings: MapValidationFinding[]): void {
  if (map.objects.length === 0) {
    findings.push(
      finding('error', 'objects_empty', 'map.objects', 'at least one object is required'),
    );
  }
  if (map.objects.length > MAX_GENERATED_OBSTACLES) {
    findings.push(
      finding(
        'error',
        'obstacle_capacity_exceeded',
        'map.objects',
        `at most ${MAX_GENERATED_OBSTACLES} objects fit the Gym obstacle tensor`,
      ),
    );
  }
  map.objects.forEach((object, index) => {
    const footprint = objectFootprint(object);
    if (
      footprint.minX < -map.width / 2 ||
      footprint.maxX > map.width / 2 ||
      footprint.minY < -map.height / 2 ||
      footprint.maxY > map.height / 2
    ) {
      findings.push(
        finding(
          'error',
          'object_out_of_bounds',
          `map.objects[${index}]`,
          'the complete collision footprint must be inside the arena',
        ),
      );
    }
  });
}

function validateSpawns(map: MapData, findings: MapValidationFinding[]): void {
  const spawns = map.spawns ?? [];
  const blue = spawns.filter((spawn) => spawn.team === Team.Player);
  const red = spawns.filter((spawn) => spawn.team === Team.Enemy);
  if (blue.length < 1 || blue.length > 10 || red.length < 1 || red.length > 10) {
    findings.push(
      finding(
        'error',
        'spawn_capacity_invalid',
        'map.spawns',
        'each team must have between 1 and 10 explicit spawns',
      ),
    );
  }
  spawns.forEach((spawn, index) => {
    if (
      spawn.x < -map.width / 2 + PLAYER.radius ||
      spawn.x > map.width / 2 - PLAYER.radius ||
      spawn.y < -map.height / 2 + PLAYER.radius ||
      spawn.y > map.height / 2 - PLAYER.radius
    ) {
      findings.push(
        finding(
          'error',
          'spawn_out_of_bounds',
          `map.spawns[${index}]`,
          'spawn must clear the arena border',
        ),
      );
    }
    const blocker = map.objects.findIndex(
      (object) =>
        object.type !== 'prop' && pointIntersectsObject(spawn.x, spawn.y, PLAYER.radius, object),
    );
    if (blocker >= 0) {
      findings.push(
        finding(
          'error',
          'spawn_blocked',
          `map.spawns[${index}]`,
          `spawn intersects blocking object ${blocker}`,
        ),
      );
    }
  });
  for (let i = 0; i < spawns.length; i++) {
    for (let j = i + 1; j < spawns.length; j++) {
      if (Math.hypot(spawns[i].x - spawns[j].x, spawns[i].y - spawns[j].y) < 1) {
        findings.push(
          finding(
            'error',
            'spawn_overlap',
            `map.spawns[${j}]`,
            `spawn is within 1 unit of spawn ${i}`,
          ),
        );
      }
    }
  }
}

function validateAgainstRequest(
  map: MapData,
  request: MapGenerationRequest,
  findings: MapValidationFinding[],
): void {
  if (map.width !== request.width || map.height !== request.height) {
    findings.push(
      finding(
        'error',
        'requested_size_mismatch',
        'map',
        `map must be exactly ${request.width}x${request.height}`,
      ),
    );
  }
  const blue = map.spawns?.filter((spawn) => spawn.team === Team.Player).length ?? 0;
  const red = map.spawns?.filter((spawn) => spawn.team === Team.Enemy).length ?? 0;
  if (blue !== request.blueCapacity || red !== request.redCapacity) {
    findings.push(
      finding(
        'error',
        'requested_capacity_mismatch',
        'map.spawns',
        `expected ${request.blueCapacity} player and ${request.redCapacity} enemy spawns`,
      ),
    );
  }
  if (map.objects.length > request.objectBudget) {
    findings.push(
      finding(
        'error',
        'requested_budget_exceeded',
        'map.objects',
        `object count exceeds requested budget ${request.objectBudget}`,
      ),
    );
  }
}

function validateQualityAgainstRequest(
  metrics: MapMetrics,
  request: MapGenerationRequest,
  findings: MapValidationFinding[],
): void {
  const budgetUse = metrics.obstacleCount / request.objectBudget;
  const densityMatches =
    request.density === 'sparse'
      ? budgetUse <= 0.5
      : request.density === 'medium'
        ? budgetUse >= 0.35 && budgetUse <= 0.85
        : budgetUse >= 0.7;
  if (!densityMatches) {
    findings.push(
      finding(
        'warning',
        'claimed_density_mismatch',
        'map.objects',
        `${request.density} density used ${(budgetUse * 100).toFixed(1)}% of the object budget`,
      ),
    );
  }
  if (request.symmetry !== 'asymmetric' && metrics.symmetryError > 0.02) {
    findings.push(
      finding(
        'warning',
        'claimed_symmetry_mismatch',
        'map.objects',
        `${request.symmetry} symmetry error ${metrics.symmetryError.toFixed(4)} exceeds 0.02`,
      ),
    );
  }
  const coverMatches =
    request.desiredCover === 'low'
      ? metrics.blockingFootprintFraction <= 0.03
      : request.desiredCover === 'medium'
        ? metrics.blockingFootprintFraction >= 0.015 && metrics.blockingFootprintFraction <= 0.12
        : metrics.blockingFootprintFraction >= 0.08;
  if (!coverMatches) {
    findings.push(
      finding(
        'warning',
        'claimed_cover_mismatch',
        'map.objects',
        `${request.desiredCover} cover request produced footprint fraction ${metrics.blockingFootprintFraction.toFixed(4)}`,
      ),
    );
  }
}

function analyzeMap(
  map: MapData,
  findings: MapValidationFinding[],
  symmetry: MapGenerationRequest['symmetry'],
): MapMetrics {
  const arena = new MapLoader(new IdAllocator()).build(map);
  const grid = new PathGrid(arena);
  const blue = (map.spawns ?? []).filter((spawn) => spawn.team === Team.Player);
  const red = (map.spawns ?? []).filter((spawn) => spawn.team === Team.Enemy);
  const bluePaths = pathLengths(grid, blue, red);
  const redPaths = pathLengths(grid, red, blue);
  if (bluePaths.some((value) => value === null) || redPaths.some((value) => value === null)) {
    findings.push(
      finding(
        'error',
        'engagement_space_disconnected',
        'map.objects',
        'every spawn must have a traversable path to at least one opposing spawn',
      ),
    );
  }

  const overlapPairs = countOverlapPairs(map.objects.filter((object) => object.type !== 'prop'));
  if (overlapPairs > 0) {
    findings.push(
      finding(
        'warning',
        'blocking_objects_overlap',
        'map.objects',
        `${overlapPairs} blocking object pairs overlap`,
      ),
    );
  }
  const symmetryError = measureSymmetry(map, symmetry === 'rotational');
  const blueMean = meanNullable(bluePaths);
  const redMean = meanNullable(redPaths);
  const sightCover = map.objects.filter(
    (object) => object.type === 'tree' || object.type === 'rock' || object.type === 'fort',
  );
  const blueCover = meanCoverDistance(blue, sightCover);
  const redCover = meanCoverDistance(red, sightCover);
  return {
    obstacleCount: map.objects.length,
    blockingObstacleCount: map.objects.filter((object) => object.type !== 'prop').length,
    decorativeObstacleCount: map.objects.filter((object) => object.type === 'prop').length,
    obstacleDensity: map.objects.length / (map.width * map.height),
    blockingFootprintFraction:
      map.objects
        .filter((object) => object.type !== 'prop')
        .reduce((sum, object) => {
          const bounds = objectFootprint(object);
          return sum + (bounds.maxX - bounds.minX) * (bounds.maxY - bounds.minY);
        }, 0) /
      (map.width * map.height),
    overlapPairs,
    symmetryError,
    blueMeanPathLength: blueMean,
    redMeanPathLength: redMean,
    pathLengthImbalance:
      blueMean === null || redMean === null
        ? null
        : Math.abs(blueMean - redMean) / Math.max(1, blueMean, redMean),
    blueMeanCoverDistance: blueCover,
    redMeanCoverDistance: redCover,
    coverAccessImbalance:
      blueCover === null || redCover === null
        ? null
        : Math.abs(blueCover - redCover) / Math.max(1, blueCover, redCover),
  };
}

function pathLengths(
  grid: PathGrid,
  starts: MapSpawnData[],
  goals: MapSpawnData[],
): Array<number | null> {
  return starts.map((start) => {
    let best: number | null = null;
    for (const goal of goals) {
      const path = grid.findPath(start.x, start.y, goal.x, goal.y);
      if (!path) continue;
      let length = 0;
      let previous = { x: start.x, y: start.y };
      for (const point of path) {
        length += Math.hypot(point.x - previous.x, point.y - previous.y);
        previous = point;
      }
      best = best === null ? length : Math.min(best, length);
    }
    return best;
  });
}

function parseObject(value: unknown, index: number): MapObjectData {
  const object = record(value, `map.objects[${index}]`);
  rejectUnknown(object, OBJECT_KEYS, `map.objects[${index}]`);
  const type = object.type;
  if (
    type !== 'tree' &&
    type !== 'rock' &&
    type !== 'fort' &&
    type !== 'fence' &&
    type !== 'prop'
  ) {
    throw new RangeError(`map.objects[${index}].type is invalid`);
  }
  if ('rotation' in object) throw new RangeError(`map.objects[${index}].rotation is unsupported`);
  const x = finite(object.x, `map.objects[${index}].x`);
  const y = finite(object.y, `map.objects[${index}].y`);
  if (type === 'tree' || type === 'rock') {
    rejectPresent(object, ['width', 'height'], `map.objects[${index}]`);
    return {
      type,
      x,
      y,
      ...(object.radius === undefined
        ? {}
        : { radius: positive(object.radius, `map.objects[${index}].radius`) }),
    };
  }
  if (type === 'fort' || type === 'fence') {
    rejectPresent(object, ['radius'], `map.objects[${index}]`);
    return {
      type,
      x,
      y,
      ...(object.width === undefined
        ? {}
        : { width: positive(object.width, `map.objects[${index}].width`) }),
      ...(object.height === undefined
        ? {}
        : { height: positive(object.height, `map.objects[${index}].height`) }),
    };
  }
  rejectPresent(object, ['radius', 'width', 'height'], `map.objects[${index}]`);
  return { type, x, y };
}

function parseSpawn(value: unknown, index: number): MapSpawnData {
  const spawn = record(value, `map.spawns[${index}]`);
  rejectUnknown(spawn, SPAWN_KEYS, `map.spawns[${index}]`);
  if (spawn.team !== Team.Player && spawn.team !== Team.Enemy) {
    throw new RangeError(`map.spawns[${index}].team must be player or enemy`);
  }
  return {
    team: spawn.team,
    x: finite(spawn.x, `map.spawns[${index}].x`),
    y: finite(spawn.y, `map.spawns[${index}].y`),
  };
}

function canonicalObject(object: MapObjectData): MapObjectData {
  const base = { type: object.type, x: normalizedNumber(object.x), y: normalizedNumber(object.y) };
  if (object.type === 'tree' || object.type === 'rock') {
    return {
      ...base,
      ...(object.radius === undefined ? {} : { radius: normalizedNumber(object.radius) }),
    };
  }
  if (object.type === 'fort' || object.type === 'fence') {
    return {
      ...base,
      ...(object.width === undefined ? {} : { width: normalizedNumber(object.width) }),
      ...(object.height === undefined ? {} : { height: normalizedNumber(object.height) }),
    };
  }
  return base;
}

function compareObjects(a: MapObjectData, b: MapObjectData): number {
  return (
    a.type.localeCompare(b.type) ||
    a.x - b.x ||
    a.y - b.y ||
    (a.radius ?? 0) - (b.radius ?? 0) ||
    (a.width ?? 0) - (b.width ?? 0) ||
    (a.height ?? 0) - (b.height ?? 0)
  );
}

function objectFootprint(object: MapObjectData): Bounds {
  if (object.type === 'tree' || object.type === 'rock' || object.type === 'prop') {
    const radius =
      object.radius ?? (object.type === 'tree' ? 0.35 : object.type === 'rock' ? 0.6 : 0.3);
    return {
      minX: object.x - radius,
      maxX: object.x + radius,
      minY: object.y - radius,
      maxY: object.y + radius,
    };
  }
  const width = object.width ?? (object.type === 'fort' ? 2.4 : 2);
  const height = object.height ?? (object.type === 'fort' ? 1.2 : 0.24);
  return {
    minX: object.x - width / 2,
    maxX: object.x + width / 2,
    minY: object.y - height / 2,
    maxY: object.y + height / 2,
  };
}

function pointIntersectsObject(
  x: number,
  y: number,
  padding: number,
  object: MapObjectData,
): boolean {
  if (object.type === 'tree' || object.type === 'rock') {
    const radius = object.radius ?? (object.type === 'tree' ? 0.35 : 0.6);
    return Math.hypot(x - object.x, y - object.y) < radius + padding;
  }
  const bounds = objectFootprint(object);
  return (
    x > bounds.minX - padding &&
    x < bounds.maxX + padding &&
    y > bounds.minY - padding &&
    y < bounds.maxY + padding
  );
}

function countOverlapPairs(objects: MapObjectData[]): number {
  let count = 0;
  for (let i = 0; i < objects.length; i++) {
    const a = objectFootprint(objects[i]);
    for (let j = i + 1; j < objects.length; j++) {
      const b = objectFootprint(objects[j]);
      if (a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY) count++;
    }
  }
  return count;
}

function measureSymmetry(map: MapData, rotational: boolean): number {
  if (map.objects.length === 0) return 0;
  let total = 0;
  for (const object of map.objects) {
    let best = Number.POSITIVE_INFINITY;
    for (const other of map.objects) {
      if (other.type !== object.type) continue;
      const a = objectFootprint(object);
      const b = objectFootprint(other);
      const distance =
        Math.hypot(other.x + object.x, rotational ? other.y + object.y : other.y - object.y) +
        Math.abs(a.maxX - a.minX - (b.maxX - b.minX)) +
        Math.abs(a.maxY - a.minY - (b.maxY - b.minY));
      best = Math.min(best, distance);
    }
    total += best;
  }
  return total / map.objects.length / Math.max(map.width, map.height);
}

function meanCoverDistance(spawns: MapSpawnData[], cover: MapObjectData[]): number | null {
  if (spawns.length === 0 || cover.length === 0) return null;
  return (
    spawns.reduce(
      (sum, spawn) =>
        sum +
        Math.min(
          ...cover.map((object) =>
            Math.max(0, Math.hypot(spawn.x - object.x, spawn.y - object.y) - coverRadius(object)),
          ),
        ),
      0,
    ) / spawns.length
  );
}

function coverRadius(object: MapObjectData): number {
  const bounds = objectFootprint(object);
  return Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) / 2;
}

function meanNullable(values: Array<number | null>): number | null {
  if (values.some((value) => value === null) || values.length === 0) return null;
  return values.reduce<number>((sum, value) => sum + (value ?? 0), 0) / values.length;
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}

function emptyReport(findings: MapValidationFinding[]): MapValidationReport {
  return {
    schemaVersion: MAP_VALIDATION_VERSION,
    valid: false,
    mapDigest: null,
    findings,
    metrics: null,
  };
}

function finding(
  severity: 'error' | 'warning',
  code: string,
  path: string,
  message: string,
): MapValidationFinding {
  return { severity, code, path, message };
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new RangeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function rejectUnknown(value: Record<string, unknown>, allowed: Set<string>, name: string): void {
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0)
    throw new RangeError(`${name} has unknown fields: ${unknown.sort().join(', ')}`);
}

function rejectPresent(value: Record<string, unknown>, keys: string[], name: string): void {
  const present = keys.filter((key) => value[key] !== undefined);
  if (present.length > 0) throw new RangeError(`${name} does not support: ${present.join(', ')}`);
}

function finite(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new RangeError(`${name} must be finite`);
  return value;
}

function positive(value: unknown, name: string): number {
  const result = finite(value, name);
  if (result <= 0) throw new RangeError(`${name} must be positive`);
  return result;
}

function nonEmptyString(value: unknown, name: string, maxLength: number): string {
  if (typeof value !== 'string' || value.trim().length === 0 || value.length > maxLength) {
    throw new RangeError(`${name} must be a non-empty string of at most ${maxLength} characters`);
  }
  return value.trim();
}

function normalizedNumber(value: number): number {
  const rounded = Math.round(value * 1_000_000) / 1_000_000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}
