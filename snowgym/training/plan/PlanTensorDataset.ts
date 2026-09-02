import { createHash } from 'node:crypto';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import { validateCommandPlan } from '../../orchestration/command/PlanValidator';
import { PlanGrounder } from '../../orchestration/grounding/PlanGrounder';
import type { Scenario } from '../../scenarios/Scenario';
import {
  PLAN_FEATURE_VECTOR_SIZE,
  PLAN_GROUP_SLOTS,
  encodePlanTensor,
} from './PlanTensorEncoder';
import {
  SYNTHETIC_PLAN_CURRICULUM_FORMAT,
  generateSyntheticPlanCurriculum,
  type SyntheticPlanCurriculum,
} from './SyntheticPlanCurriculum';

export const PLAN_TENSOR_DATASET_FORMAT = 'snowgym.plan-tensor-dataset.v0' as const;

export interface PlanTensorDataset {
  readonly format: typeof PLAN_TENSOR_DATASET_FORMAT;
  readonly apiVersion: string;
  readonly simulationVersion: string;
  readonly stateHashVersion: string;
  readonly upstreamBaseCommit: string;
  readonly scenario: string;
  readonly environmentSeed: number;
  readonly sourceStateHash: string;
  readonly curriculum: SyntheticPlanCurriculum;
  readonly tensors: readonly {
    readonly sourceSeed: number;
    readonly groups: readonly number[];
    readonly groupMask: readonly number[];
  }[];
  readonly datasetDigest: string;
}

export interface PlanTensorDatasetOptions {
  readonly scenario: Scenario;
  readonly environmentSeed: number;
  readonly basePlanSeed: number;
  readonly sampleCount: number;
}

/** Build plan tensors from one detached authoritative simulator state. */
export function buildPlanTensorDataset(options: PlanTensorDatasetOptions): PlanTensorDataset {
  const environment = new SnowEnvironment({ scenario: options.scenario });
  const observation = environment.reset(options.environmentSeed);
  const status = environment.status();
  const curriculum = generateSyntheticPlanCurriculum(observation, {
    baseSeed: options.basePlanSeed,
    sampleCount: options.sampleCount,
    sourceStateHash: status.stateHash,
  });
  const grounder = new PlanGrounder();
  const tensors = curriculum.samples.map((sample) => {
    const grounded = grounder.ground(
      {
        planId: sample.planId,
        source: {
          requestId: `synthetic-request-${sample.sourceSeed}`,
          sourceTick: observation.tick,
          sourceStateHash: status.stateHash,
        },
        decision: sample.plan,
      },
      observation,
    );
    assertAssignmentsMatch(sample.assignments, grounded.groups);
    const encoded = encodePlanTensor(
      { plan: grounded, activatedAtTick: observation.tick, version: 1 },
      observation,
      observation.tick,
    );
    return {
      sourceSeed: sample.sourceSeed,
      groups: [...encoded.groups],
      groupMask: [...encoded.groupMask],
    };
  });
  const body = {
    format: PLAN_TENSOR_DATASET_FORMAT,
    apiVersion: status.apiVersion,
    simulationVersion: status.simulationVersion,
    stateHashVersion: status.stateHashVersion,
    upstreamBaseCommit: status.upstreamBaseCommit,
    scenario: status.scenario,
    environmentSeed: options.environmentSeed,
    sourceStateHash: status.stateHash,
    curriculum,
    tensors,
  };
  return { ...body, datasetDigest: digest(body) };
}

export function auditPlanTensorDataset(value: unknown): PlanTensorDataset {
  if (!isRecord(value) || value.format !== PLAN_TENSOR_DATASET_FORMAT) {
    throw new ValueError(`plan tensor dataset format must be ${PLAN_TENSOR_DATASET_FORMAT}`);
  }
  const dataset = value as unknown as PlanTensorDataset;
  if (dataset.curriculum?.format !== SYNTHETIC_PLAN_CURRICULUM_FORMAT) {
    throw new ValueError('plan tensor dataset curriculum format is invalid');
  }
  if (
    dataset.curriculum.sampleCount !== dataset.curriculum.samples.length ||
    dataset.tensors.length !== dataset.curriculum.samples.length
  ) {
    throw new ValueError('plan tensor dataset sample counts do not match');
  }
  for (let index = 0; index < dataset.curriculum.samples.length; index++) {
    const sample = dataset.curriculum.samples[index];
    const tensor = dataset.tensors[index];
    if (!validateCommandPlan(sample.plan).ok) {
      throw new ValueError(`plan tensor dataset sample ${index} has an invalid plan`);
    }
    if (tensor.sourceSeed !== sample.sourceSeed) {
      throw new ValueError(`plan tensor dataset sample ${index} seed is misaligned`);
    }
    if (
      tensor.groups.length !== PLAN_GROUP_SLOTS * PLAN_FEATURE_VECTOR_SIZE ||
      tensor.groupMask.length !== PLAN_GROUP_SLOTS
    ) {
      throw new ValueError(`plan tensor dataset sample ${index} tensor shape is invalid`);
    }
    if (
      !tensor.groups.every((item) => Number.isFinite(item) && item >= -1 && item <= 1) ||
      !tensor.groupMask.every((item) => item === 0 || item === 1)
    ) {
      throw new ValueError(`plan tensor dataset sample ${index} tensor value is invalid`);
    }
  }
  const { datasetDigest, ...body } = dataset;
  if (datasetDigest !== digest(body)) {
    throw new ValueError('plan tensor dataset digest mismatch');
  }
  return dataset;
}

function assertAssignmentsMatch(
  expected: SyntheticPlanCurriculum['samples'][number]['assignments'],
  grounded: ReturnType<PlanGrounder['ground']>['groups'],
): void {
  const actual = grounded.map(({ role, assignment }) => ({
    role,
    unitIds: [...assignment.unitIds],
  }));
  if (canonicalPlanJson(expected) !== canonicalPlanJson(actual)) {
    throw new ValueError('regrounded assignments do not match curriculum assignments');
  }
}

function digest(value: unknown): string {
  return planArtifactDigest(value);
}

export function planArtifactDigest(value: unknown): string {
  return `sha256:${createHash('sha256').update(canonicalPlanJson(value)).digest('hex')}`;
}

export function canonicalPlanJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalPlanJson).join(',')}]`;
  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalPlanJson(value[key])}`)
      .join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

class ValueError extends Error {}
