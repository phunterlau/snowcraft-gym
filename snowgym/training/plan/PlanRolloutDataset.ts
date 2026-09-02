import type { ActionResult } from '../../adapters/SnowCraftActionAdapter';
import type { TeamAction } from '../../actions/UnitAction';
import { SnowEnvironment } from '../../core/SnowEnvironment';
import type { Observation } from '../../observations/Observation';
import { validateCommandPlan } from '../../orchestration/command/PlanValidator';
import { PlanAwareTeamController } from '../../orchestration/execution/PlanAwareTeamController';
import { ReactiveUnitPolicy } from '../../orchestration/execution/ReactiveUnitPolicy';
import { PlanGrounder } from '../../orchestration/grounding/PlanGrounder';
import { PlanStore } from '../../orchestration/runtime/PlanStore';
import { hashObservation } from '../../reproducibility/StateHash';
import type { Scenario } from '../../scenarios/Scenario';
import {
  PLAN_FEATURE_VECTOR_SIZE,
  PLAN_GROUP_SLOTS,
  encodePlanTensor,
} from './PlanTensorEncoder';
import { planArtifactDigest } from './PlanTensorDataset';
import {
  SYNTHETIC_PLAN_CURRICULUM_FORMAT,
  generateSyntheticPlanCurriculum,
  type SyntheticPlanCurriculum,
  type SyntheticPlanSample,
} from './SyntheticPlanCurriculum';

export const PLAN_ROLLOUT_DATASET_FORMAT = 'snowgym.plan-rollout-dataset.v0' as const;

export interface PlanRolloutTransition {
  readonly decision: number;
  readonly observation: Observation;
  readonly action: TeamAction;
  readonly planGroups: readonly number[];
  readonly planGroupMask: readonly number[];
  readonly reward: number;
  readonly terminated: boolean;
  readonly truncated: boolean;
  readonly preStateHash: string;
  readonly postStateHash: string;
  readonly nextTick: number;
  readonly actionResults: readonly ActionResult[];
}

export interface PlanRolloutEpisode {
  readonly sourceSeed: number;
  readonly planId: string;
  readonly plan: SyntheticPlanSample['plan'];
  readonly assignments: SyntheticPlanSample['assignments'];
  readonly initialStateHash: string;
  readonly transitions: readonly PlanRolloutTransition[];
  readonly outcome: {
    readonly decisions: number;
    readonly terminated: boolean;
    readonly truncated: boolean;
    readonly decisionLimited: boolean;
    readonly winner: 'blue' | 'red' | null;
    readonly blueAlive: number;
    readonly redAlive: number;
    readonly finalTick: number;
    readonly finalStateHash: string;
  };
}

export interface PlanRolloutDataset {
  readonly format: typeof PLAN_ROLLOUT_DATASET_FORMAT;
  readonly apiVersion: string;
  readonly simulationVersion: string;
  readonly stateHashVersion: string;
  readonly upstreamBaseCommit: string;
  readonly scenario: string;
  readonly environmentSeed: number;
  readonly decisionHz: number;
  readonly configuration: ReturnType<SnowEnvironment['status']>['configuration'];
  readonly sourceStateHash: string;
  readonly maxDecisions: number;
  readonly curriculum: SyntheticPlanCurriculum;
  readonly episodes: readonly PlanRolloutEpisode[];
  readonly datasetDigest: string;
}

export interface PlanRolloutDatasetOptions {
  readonly scenario: Scenario;
  readonly environmentSeed: number;
  readonly basePlanSeed: number;
  readonly sampleCount: number;
  readonly maxDecisions: number;
  readonly decisionHz?: number;
  readonly redDifficulty?: 'easy' | 'normal' | 'hard';
}

/**
 * Execute synthetic commander plans from the same authoritative initial state.
 * The resulting labels are produced by the production plan-aware controller,
 * so plans are causes of actions rather than metadata attached after the fact.
 */
export function buildPlanRolloutDataset(options: PlanRolloutDatasetOptions): PlanRolloutDataset {
  positiveInteger(options.maxDecisions, 'maxDecisions');
  const environmentConfig = {
    scenario: options.scenario,
    ...(options.decisionHz === undefined ? {} : { decisionHz: options.decisionHz }),
    ...(options.redDifficulty === undefined ? {} : { redDifficulty: options.redDifficulty }),
  };
  const sourceEnvironment = new SnowEnvironment(environmentConfig);
  const sourceObservation = sourceEnvironment.reset(options.environmentSeed);
  const sourceStatus = sourceEnvironment.status();
  const curriculum = generateSyntheticPlanCurriculum(sourceObservation, {
    baseSeed: options.basePlanSeed,
    sampleCount: options.sampleCount,
    sourceStateHash: sourceStatus.stateHash,
  });
  const episodes = curriculum.samples.map((sample) =>
    executePlan(sample, options.environmentSeed, options.maxDecisions, environmentConfig),
  );
  const body = {
    format: PLAN_ROLLOUT_DATASET_FORMAT,
    apiVersion: sourceStatus.apiVersion,
    simulationVersion: sourceStatus.simulationVersion,
    stateHashVersion: sourceStatus.stateHashVersion,
    upstreamBaseCommit: sourceStatus.upstreamBaseCommit,
    scenario: sourceStatus.scenario,
    environmentSeed: options.environmentSeed,
    decisionHz: sourceStatus.decisionHz,
    configuration: sourceStatus.configuration,
    sourceStateHash: sourceStatus.stateHash,
    maxDecisions: options.maxDecisions,
    curriculum,
    episodes,
  };
  return { ...body, datasetDigest: planArtifactDigest(body) };
}

export function auditPlanRolloutDataset(value: unknown): PlanRolloutDataset {
  if (!isRecord(value) || value.format !== PLAN_ROLLOUT_DATASET_FORMAT) {
    throw new PlanRolloutFormatError(
      `plan rollout dataset format must be ${PLAN_ROLLOUT_DATASET_FORMAT}`,
    );
  }
  const dataset = value as unknown as PlanRolloutDataset;
  positiveInteger(dataset.maxDecisions, 'maxDecisions');
  if (dataset.curriculum?.format !== SYNTHETIC_PLAN_CURRICULUM_FORMAT) {
    throw new PlanRolloutFormatError('plan rollout curriculum format is invalid');
  }
  if (
    dataset.curriculum.samples.length !== dataset.curriculum.sampleCount ||
    dataset.episodes.length !== dataset.curriculum.sampleCount
  ) {
    throw new PlanRolloutFormatError('plan rollout sample counts do not match');
  }
  for (let index = 0; index < dataset.episodes.length; index++) {
    auditEpisode(dataset, dataset.curriculum.samples[index], dataset.episodes[index], index);
  }
  const { datasetDigest, ...body } = dataset;
  if (datasetDigest !== planArtifactDigest(body)) {
    throw new PlanRolloutFormatError('plan rollout dataset digest mismatch');
  }
  return dataset;
}

function executePlan(
  sample: SyntheticPlanSample,
  environmentSeed: number,
  maxDecisions: number,
  environmentConfig: ConstructorParameters<typeof SnowEnvironment>[0],
): PlanRolloutEpisode {
  const environment = new SnowEnvironment(environmentConfig);
  let observation = environment.reset(environmentSeed);
  let status = environment.status();
  const initialStateHash = status.stateHash;
  const grounded = new PlanGrounder().ground(
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
  const planStore = new PlanStore(grounded, observation.tick);
  const controller = new PlanAwareTeamController(planStore, new ReactiveUnitPolicy());
  const transitions: PlanRolloutTransition[] = [];

  while (!status.terminated && !status.truncated && transitions.length < maxDecisions) {
    const preStateHash = status.stateHash;
    const encoded = encodePlanTensor(planStore.current(), observation, observation.tick);
    const action = controller.act(observation, 1 / environment.decisionHz);
    const result = environment.step(action);
    status = environment.status();
    if (result.info.actionResults.some(({ accepted }) => !accepted)) {
      throw new PlanRolloutFormatError(
        `plan ${sample.planId} produced a rejected physical action at decision ${transitions.length}`,
      );
    }
    transitions.push({
      decision: transitions.length,
      observation,
      action,
      planGroups: [...encoded.groups],
      planGroupMask: [...encoded.groupMask],
      reward: result.reward,
      terminated: result.terminated,
      truncated: result.truncated,
      preStateHash,
      postStateHash: status.stateHash,
      nextTick: result.observation.tick,
      actionResults: result.info.actionResults,
    });
    observation = result.observation;
  }
  const decisionLimited = !status.terminated && !status.truncated;
  return {
    sourceSeed: sample.sourceSeed,
    planId: sample.planId,
    plan: sample.plan,
    assignments: sample.assignments,
    initialStateHash,
    transitions,
    outcome: {
      decisions: transitions.length,
      terminated: status.terminated,
      truncated: status.truncated,
      decisionLimited,
      winner: status.winner,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      finalTick: status.tick,
      finalStateHash: status.stateHash,
    },
  };
}

function auditEpisode(
  dataset: PlanRolloutDataset,
  sample: SyntheticPlanSample,
  episode: PlanRolloutEpisode,
  index: number,
): void {
  if (!validateCommandPlan(episode.plan).ok) {
    throw new PlanRolloutFormatError(`plan rollout episode ${index} has an invalid plan`);
  }
  if (
    episode.sourceSeed !== sample.sourceSeed ||
    episode.planId !== sample.planId ||
    planArtifactDigest(episode.plan) !== planArtifactDigest(sample.plan) ||
    planArtifactDigest(episode.assignments) !== planArtifactDigest(sample.assignments)
  ) {
    throw new PlanRolloutFormatError(`plan rollout episode ${index} is misaligned`);
  }
  if (episode.initialStateHash !== dataset.sourceStateHash) {
    throw new PlanRolloutFormatError(`plan rollout episode ${index} initial state hash differs`);
  }
  if (
    episode.outcome.decisions !== episode.transitions.length ||
    episode.transitions.length > dataset.maxDecisions ||
    episode.outcome.decisionLimited ===
      (episode.outcome.terminated || episode.outcome.truncated)
  ) {
    throw new PlanRolloutFormatError(`plan rollout episode ${index} outcome is inconsistent`);
  }
  let expectedHash = episode.initialStateHash;
  for (let decision = 0; decision < episode.transitions.length; decision++) {
    const transition = episode.transitions[decision];
    if (
      transition.decision !== decision ||
      transition.preStateHash !== expectedHash ||
      hashObservation(transition.observation) !== transition.preStateHash
    ) {
      throw new PlanRolloutFormatError(
        `plan rollout episode ${index} transition ${decision} state hash is invalid`,
      );
    }
    if (
      transition.planGroups.length !== PLAN_GROUP_SLOTS * PLAN_FEATURE_VECTOR_SIZE ||
      transition.planGroupMask.length !== PLAN_GROUP_SLOTS ||
      !transition.planGroups.every((item) => Number.isFinite(item) && item >= -1 && item <= 1) ||
      !transition.planGroupMask.every((item) => item === 0 || item === 1)
    ) {
      throw new PlanRolloutFormatError(
        `plan rollout episode ${index} transition ${decision} plan tensor is invalid`,
      );
    }
    if (
      transition.nextTick <= transition.observation.tick ||
      transition.actionResults.length !== transition.action.actions.length ||
      transition.actionResults.some(({ accepted }) => !accepted)
    ) {
      throw new PlanRolloutFormatError(
        `plan rollout episode ${index} transition ${decision} action result is invalid`,
      );
    }
    expectedHash = transition.postStateHash;
  }
  if (expectedHash !== episode.outcome.finalStateHash) {
    throw new PlanRolloutFormatError(`plan rollout episode ${index} final state hash differs`);
  }
}

function positiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

class PlanRolloutFormatError extends Error {}
