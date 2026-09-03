import {
  MAP_GENERATION_REQUEST_VERSION,
  type DatasetSplit,
  type MapDensity,
  type MapGenerationRequest,
  type MapSymmetry,
  type MapTopology,
} from './types';

export interface MapGenerationRequestOptions {
  brief: string;
  blueCapacity?: number;
  redCapacity?: number;
  width?: number;
  height?: number;
  topology?: MapTopology;
  symmetry?: MapSymmetry;
  density?: MapDensity;
  objectBudget?: number;
  desiredCover?: 'low' | 'medium' | 'high';
  split?: DatasetSplit;
  tags?: string[];
}

export function createMapGenerationRequest(
  options: MapGenerationRequestOptions,
): MapGenerationRequest {
  const brief = boundedText(options.brief, 'brief', 2_000);
  const blueCapacity = boundedInteger(options.blueCapacity ?? 10, 'blueCapacity', 1, 10);
  const redCapacity = boundedInteger(options.redCapacity ?? 10, 'redCapacity', 1, 10);
  const width = boundedNumber(options.width ?? 64, 'width', 12, 120);
  const height = boundedNumber(options.height ?? 48, 'height', 12, 120);
  const objectBudget = boundedInteger(options.objectBudget ?? 40, 'objectBudget', 1, 64);
  const tags = (options.tags ?? []).map((tag, index) => boundedText(tag, `tags[${index}]`, 64));
  if (new Set(tags).size !== tags.length) throw new RangeError('tags must be unique');
  return {
    schemaVersion: MAP_GENERATION_REQUEST_VERSION,
    brief,
    blueCapacity,
    redCapacity,
    width,
    height,
    topology: member(
      options.topology ?? 'mixed',
      ['open', 'lanes', 'chokepoint', 'pockets', 'mixed'],
      'topology',
    ),
    symmetry: member(
      options.symmetry ?? 'mirror',
      ['mirror', 'rotational', 'asymmetric'],
      'symmetry',
    ),
    density: member(options.density ?? 'medium', ['sparse', 'medium', 'dense'], 'density'),
    objectBudget,
    desiredCover: member(
      options.desiredCover ?? 'medium',
      ['low', 'medium', 'high'],
      'desiredCover',
    ),
    split: member(options.split ?? 'development', ['development', 'evaluation'], 'split'),
    tags,
  };
}

function boundedInteger(value: number, name: string, minimum: number, maximum: number): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer in [${minimum}, ${maximum}]`);
  }
  return value;
}

function boundedNumber(value: number, name: string, minimum: number, maximum: number): number {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be in [${minimum}, ${maximum}]`);
  }
  return value;
}

function boundedText(value: string, name: string, maximum: number): string {
  if (typeof value !== 'string' || value.trim().length === 0 || value.length > maximum) {
    throw new RangeError(`${name} must be non-empty and at most ${maximum} characters`);
  }
  return value.trim();
}

function member<const T extends string>(value: string, allowed: readonly T[], name: string): T {
  if (!allowed.includes(value as T))
    throw new RangeError(`${name} must be one of: ${allowed.join(', ')}`);
  return value as T;
}
