import type { TeamAction } from '../../actions/UnitAction';
import type { Observation } from '../../observations/Observation';
import {
  benchmarkDigest,
  prepareBackendFixture,
  restoreBackendFixture,
  type BackendCase,
  type BackendFixture,
} from '../benchmark/CommanderBackendBenchmark';
import { summarizeStrategy } from '../commander/StrategicSummary';
import { PlanLifecycle } from '../lifecycle/PlanLifecycle';
import { createFallbackEnvelope } from '../lifecycle/FallbackPlan';
import { PlanGrounder } from '../grounding/PlanGrounder';
import {
  RECOVERY_FAMILIES,
  summarizeRecovery,
  type RecoveryEvidence,
  type RecoveryFamily,
} from './RecoveryEvidence';

export const RECOVERY_BENCHMARK_VERSION = 'snowgym.commander-recovery.v0' as const;
/** Diagnostic-only seeds; not a claim of untouched training/qualification data. */
export const RECOVERY_SCAN_CASES: readonly BackendCase[] = [
  {
    id: 'recovery-open-5v5',
    seed: 620001,
    blueUnits: 5,
    redUnits: 5,
    map: null,
    prefixDecisions: 0,
  },
  {
    id: 'recovery-terrain-5v5',
    seed: 620002,
    blueUnits: 5,
    redUnits: 5,
    map: 'arena6.json',
    prefixDecisions: 0,
  },
  {
    id: 'recovery-terrain-10v10',
    seed: 620003,
    blueUnits: 10,
    redUnits: 10,
    map: 'arena6.json',
    prefixDecisions: 0,
  },
  {
    id: 'recovery-terrain-6v10',
    seed: 620004,
    blueUnits: 6,
    redUnits: 10,
    map: 'arena6.json',
    prefixDecisions: 0,
  },
];
export interface RecoveryFixture {
  format: typeof RECOVERY_BENCHMARK_VERSION;
  family: RecoveryFamily;
  base: BackendFixture;
  activation: Observation;
  evidence: RecoveryEvidence;
  digest: string;
}

/** First qualifying opportunity per family per world; never select by model outcome. */
export function collectRecoveryFixtures(cases = RECOVERY_SCAN_CASES, horizon = 300) {
  boundedInteger(horizon, 1, 1000, 'scan horizon');
  const fixtures: RecoveryFixture[] = [];
  const scans = [];
  for (const configuration of cases) {
    if (configuration.prefixDecisions !== 0 || configuration.switchAt !== undefined)
      throw new Error('recovery scan requires an unswitched reset');
    const initial = prepareBackendFixture(configuration);
    const world = restoreBackendFixture(initial);
    const prefix: TeamAction[] = [];
    const prefixStateHashes = [world.environment.status().stateHash];
    const found = new Set<RecoveryFamily>();
    while (prefix.length < horizon && running(world)) {
      const action = world.controller.act(world.observation, 0.1);
      advance(world, action);
      prefix.push(action);
      prefixStateHashes.push(world.environment.status().stateHash);
      if (!running(world)) break;
      const evidence = summarizeRecovery(
        world.observation,
        initial.observation,
        world.store.current(),
        world.monitor.digest(),
      );
      for (const family of evidence.detectedFamilies) {
        if (found.has(family)) continue;
        found.add(family);
        const body = {
          ...initial,
          configuration: { ...configuration, prefixDecisions: prefix.length },
          prefix: structuredClone(prefix),
          prefixStateHashes: [...prefixStateHashes],
          observation: world.observation,
          snapshot: world.store.current(),
          request: {
            ...initial.request,
            summary: summarizeStrategy(world.observation, world.store.current()),
            trajectory: world.monitor.digest(),
          },
        };
        const { fixtureDigest: _old, ...baseBody } = body;
        const outer = {
          format: RECOVERY_BENCHMARK_VERSION,
          family,
          base: { ...baseBody, fixtureDigest: benchmarkDigest(baseBody) },
          activation: initial.observation,
          evidence,
        };
        const fixture = { ...outer, digest: benchmarkDigest(outer) };
        restoreRecoveryFixture(fixture);
        fixtures.push(fixture);
      }
    }
    scans.push({
      configuration,
      decisions: prefix.length,
      found: [...found],
      finalStateHash: world.environment.status().stateHash,
    });
  }
  return {
    format: RECOVERY_BENCHMARK_VERSION,
    fixtures,
    scans,
    missingFamilies: RECOVERY_FAMILIES.filter(
      (family) => !fixtures.some((f) => f.family === family),
    ),
  };
}

export function restoreRecoveryFixture(fixture: RecoveryFixture) {
  const { digest, ...body } = fixture;
  if (fixture.format !== RECOVERY_BENCHMARK_VERSION || benchmarkDigest(body) !== digest)
    throw new Error('recovery fixture digest mismatch');
  const initial = prepareBackendFixture({ ...fixture.base.configuration, prefixDecisions: 0 });
  if (benchmarkDigest(initial.observation) !== benchmarkDigest(fixture.activation))
    throw new Error('recovery activation mismatch');
  const world = restoreBackendFixture(fixture.base);
  const evidence = summarizeRecovery(
    world.observation,
    fixture.activation,
    world.store.current(),
    world.monitor.digest(),
  );
  if (
    benchmarkDigest(evidence) !== benchmarkDigest(fixture.evidence) ||
    !evidence.detectedFamilies.includes(fixture.family)
  )
    throw new Error('recovery evidence mismatch');
  return world;
}

export function recoveryRequest(fixture: RecoveryFixture, enriched: boolean) {
  restoreRecoveryFixture(fixture);
  return { ...fixture.base.request, ...(enriched ? { recoveryEvidence: fixture.evidence } : {}) };
}

/** One request opportunity, fixed continuation; delay is part of the shared total horizon. */
export function continueRecoveryPlan(
  fixture: RecoveryFixture,
  decision: unknown | null,
  delayDecisions = 0,
  horizon = 300,
) {
  boundedInteger(delayDecisions, 0, 80, 'delay decisions');
  boundedInteger(horizon, 1, 1000, 'continuation horizon');
  if (delayDecisions >= horizon) throw new Error('delay must be shorter than continuation horizon');
  const world = restoreRecoveryFixture(fixture);
  const initial = world.observation;
  const oldVersion = world.store.current().version;
  const actions: TeamAction[] = [];
  const stateHashes = [world.environment.status().stateHash];
  const stages: {
    decision: number;
    tick: number;
    stage: number;
    inRangeFraction: number | null;
  }[] = [];
  const stageFirstDecision: (number | null)[] = [0, null, null, null, null];
  const damageDecisions: number[] = [];
  let maximumStage = 0;
  let issuedActions = 0;
  let rejectedActions = 0;
  let activationTick: number | null = null;
  let activationStatus = 'not_reached';
  let activationError: string | null = null;
  let planChanged = false;
  const step = () => {
    const before = world.observation;
    const result = advance(world, world.controller.act(before, 0.1));
    actions.push(result.action);
    stateHashes.push(result.info.stateHash);
    issuedActions += result.info.actionResults.length;
    rejectedActions += result.info.actionResults.filter((row) => !row.accepted).length;
    const evidence = summarizeRecovery(
      world.observation,
      fixture.activation,
      fixture.base.snapshot,
    ); // Original frozen objective remains the scoring reference.
    const inRange =
      evidence.groups.reduce(
        (sum, group) => sum + (group.inExecutorRangeFraction ?? 0) * group.living,
        0,
      ) /
      Math.max(
        evidence.groups.reduce((sum, group) => sum + group.living, 0),
        1,
      );
    if (teamHealth(before.enemies) > teamHealth(world.observation.enemies))
      damageDecisions.push(actions.length);
    const damageCount = damageDecisions.filter((at) => actions.length - at < 20).length;
    const stage = evidence.groups.some((group) => group.frozenTargetEliminated)
      ? 4
      : damageCount >= 2
        ? 3
        : damageDecisions.length
          ? 2
          : inRange > 0
            ? 1
            : 0;
    maximumStage = Math.max(maximumStage, stage);
    // Thresholds can be skipped by in-flight projectiles or already eliminated targets.
    if (stageFirstDecision[stage] === null) stageFirstDecision[stage] = actions.length;
    stages.push({
      decision: actions.length,
      tick: world.observation.tick,
      stage,
      inRangeFraction: inRange,
    });
  };
  while (actions.length < delayDecisions && running(world)) step();
  if (running(world)) {
    activationTick = world.observation.tick;
    if (decision === null) activationStatus = 'kept';
    else {
      const result = new PlanLifecycle(world.store).activateCandidate(
        {
          planId: 'recovery-candidate',
          source: {
            requestId: fixture.base.request.requestId,
            sourceTick: initial.tick,
            sourceStateHash: fixture.base.request.summary.sourceStateHash,
          },
          decision,
        },
        world.observation,
      );
      activationStatus = result.status;
      if (result.status === 'rejected') {
        activationError = result.error;
        world.store.activate(
          new PlanGrounder().ground(
            createFallbackEnvelope(world.observation, 'recovery_invalid_plan', 1),
            world.observation,
          ),
          world.observation.tick,
        );
      }
      planChanged =
        benchmarkDigest(world.store.current().plan.envelope.decision.groups) !==
        benchmarkDigest(fixture.base.request.currentPlan.groups);
    }
  }
  while (actions.length < horizon && running(world)) step();
  const status = world.environment.status();
  return {
    format: RECOVERY_BENCHMARK_VERSION,
    fixtureDigest: fixture.digest,
    family: fixture.family,
    delayDecisions,
    horizon,
    activationStatus,
    activationError,
    activationTick,
    sourceAgeAtActivationSeconds:
      activationTick === null ? null : (activationTick - initial.tick) / initial.simulationHz,
    planChanged,
    planReactivated: world.store.current().version !== oldVersion,
    actions,
    stateHashes,
    stages,
    metrics: {
      winner: status.winner,
      censored: !status.terminated && !status.truncated,
      decisions: actions.length,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      damageDealt: teamHealth(initial.enemies) - teamHealth(world.observation.enemies),
      damageReceived: teamHealth(initial.allies) - teamHealth(world.observation.allies),
      maximumStage,
      stageFirstDecision,
      // Stage 4 may predate this request: never present it as recovery caused by the candidate.
      frozenTargetAlreadyEliminatedAtRequest: fixture.evidence.groups.some(
        (group) => group.frozenTargetEliminated,
      ),
      issuedActions,
      rejectedActions,
      rejectedActionRate: rejectedActions / Math.max(issuedActions, 1),
    },
  };
}

function running(world: ReturnType<typeof restoreBackendFixture>): boolean {
  const status = world.environment.status();
  return !status.terminated && !status.truncated;
}
function advance(world: ReturnType<typeof restoreBackendFixture>, action: TeamAction) {
  const before = world.observation;
  const result = world.environment.step(action);
  world.monitor.record({
    before,
    after: result.observation,
    actionResults: result.info.actionResults,
    plan: world.store.current(),
  });
  world.observation = result.observation;
  return { ...result, action };
}
function teamHealth(units: Observation['allies']): number {
  return units.reduce((sum, unit) => sum + (unit.alive ? Math.max(0, unit.health) : 0), 0);
}
function boundedInteger(value: number, min: number, max: number, name: string) {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    throw new Error(`${name} must be an integer in [${min}, ${max}]`);
}
