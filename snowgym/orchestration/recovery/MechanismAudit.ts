import type { TeamAction } from '../../actions/UnitAction';
import type { Observation, UnitObservation } from '../../observations/Observation';
import { benchmarkDigest } from '../benchmark/CommanderBackendBenchmark';
import { PlanAwareTeamController } from '../execution/PlanAwareTeamController';
import { ReactiveUnitPolicy, throwRange } from '../execution/ReactiveUnitPolicy';
import { TargetResolver } from '../grounding/TargetResolver';
import { PlanLifecycle } from '../lifecycle/PlanLifecycle';
import type { PlanSnapshot } from '../runtime/PlanStore';
import { restoreRecoveryFixture, type RecoveryFixture } from './RecoveryBenchmark';
import { BindingRefreshResolver, LocalFirePolicy } from './SelectiveRepair';

export const MECHANISM_VERSION = 'snowgym.selective-repair-audit.v0' as const;
export const MECHANISM_ARMS = ['keep', 'refresh_binding', 'local_fire', 'reactivate'] as const;
export type MechanismArm = (typeof MECHANISM_ARMS)[number];
export const MECHANISM_DELAYS = [0, 10, 20, 40, 80] as const;
export const MECHANISM_CASES = Array.from({ length: 120 }, (_, index) => ({
  id: `mechanism-${index < 40 ? '5v5' : index < 80 ? '10v10' : '6v10'}-${630000 + index}`,
  seed: 630000 + index,
  blueUnits: index < 40 ? 5 : index < 80 ? 10 : 6,
  redUnits: index < 40 ? 5 : 10,
  map: 'arena6.json',
  prefixDecisions: 0,
}));

export function runMechanismBranch(
  fixture: RecoveryFixture,
  arm: MechanismArm,
  delayDecisions: number,
  horizon = 300,
) {
  if (
    !MECHANISM_ARMS.includes(arm) ||
    !MECHANISM_DELAYS.includes(delayDecisions as never) ||
    !Number.isSafeInteger(horizon) ||
    horizon <= delayDecisions ||
    horizon > 1000
  )
    throw new Error('invalid mechanism arm/delay/horizon');
  const world = restoreRecoveryFixture(fixture);
  const source = world.observation;
  const sourceSnapshot = world.store.current();
  const normalResolver = new TargetResolver();
  let resolver: TargetResolver = normalResolver;
  const localFire = new LocalFirePolicy();
  const actions: TeamAction[] = [];
  const stateHashes = [world.environment.status().stateHash];
  const timeline: ReturnType<typeof measure>[] = [];
  const opportunisticActions: {
    tick: number;
    unitId: number;
    targetId: number;
    accepted: boolean;
  }[] = [];
  let interventionTick: number | null = null;
  let interventionStatus = 'not_reached';
  let bindingRevision = 0;
  let actualBindingChanges = 0;
  let beforeIntervention: ReturnType<typeof bindingState> | null = null;
  let afterIntervention: ReturnType<typeof bindingState> | null = null;
  let postInterventionSnapshot: PlanSnapshot | null = null;
  let issued = 0;
  let rejected = 0;
  let canonicalReturn = 0;
  const alive = () =>
    !world.environment.status().terminated && !world.environment.status().truncated;
  const frozen = sourceSnapshot.plan.groups.map((group) => ({
    role: group.role,
    enemyIds: group.objective.kind === 'enemy_cluster' ? [...group.objective.enemyIds] : [],
    initialHealth:
      group.objective.kind === 'enemy_cluster'
        ? teamHealth(
            fixture.activation.enemies.filter(
              (unit) =>
                group.objective.kind === 'enemy_cluster' &&
                group.objective.enemyIds.includes(unit.id),
            ),
          )
        : 0,
  }));
  const step = () => {
    const before = world.observation;
    const shotStart = localFire.shots.length;
    const action = world.controller.act(before, 0.1);
    const result = world.environment.step(action);
    actions.push(action);
    stateHashes.push(result.info.stateHash);
    issued += result.info.actionResults.length;
    rejected += result.info.actionResults.filter((row) => !row.accepted).length;
    canonicalReturn += result.reward;
    for (const shot of localFire.shots.slice(shotStart))
      opportunisticActions.push({
        ...shot,
        accepted: result.info.actionResults.some(
          (row) => row.action.unitId === shot.unitId && row.accepted,
        ),
      });
    world.observation = result.observation;
    timeline.push(measure());
  };
  function measure() {
    const observation = world.observation;
    const active = bindingState(world.store.current(), observation, resolver);
    return {
      tick: observation.tick,
      blueHealth: teamHealth(observation.allies),
      redHealth: teamHealth(observation.enemies),
      originalRangeOccupancy: rangeOccupancy(sourceSnapshot, observation, frozen),
      activeRangeOccupancy: rangeOccupancy(world.store.current(), observation, active),
      activeBindings: active,
      frozenTargets: frozen.map((group) => {
        const targets = observation.enemies.filter((unit) => group.enemyIds.includes(unit.id));
        return {
          ...group,
          living: targets.filter((unit) => unit.alive).length,
          healthFraction: group.initialHealth ? teamHealth(targets) / group.initialHealth : null,
          eliminated: group.enemyIds.length > 0 && targets.every((unit) => !unit.alive),
        };
      }),
    };
  }
  const initialMeasure = measure();
  while (actions.length < delayDecisions && alive()) step();
  if (alive()) {
    interventionTick = world.observation.tick;
    beforeIntervention = bindingState(world.store.current(), world.observation, normalResolver);
    if (arm === 'reactivate') {
      const outcome = new PlanLifecycle(world.store).activateCandidate(
        {
          planId: 'recovery-candidate',
          source: {
            requestId: fixture.base.request.requestId,
            sourceTick: source.tick,
            sourceStateHash: fixture.base.request.summary.sourceStateHash,
          },
          decision: fixture.base.request.currentPlan,
        },
        world.observation,
      );
      if (outcome.status === 'rejected')
        throw new Error(`declared same-symbol plan rejected: ${outcome.error}`);
      interventionStatus = outcome.status;
    } else {
      interventionStatus = arm === 'keep' ? 'kept' : 'applied';
      if (arm === 'refresh_binding') {
        resolver = new BindingRefreshResolver(sourceSnapshot, world.observation);
        bindingRevision = 1;
      }
      world.controller = new PlanAwareTeamController(
        world.store,
        arm === 'local_fire' ? localFire : new ReactiveUnitPolicy(),
        resolver,
      );
    }
    postInterventionSnapshot = world.store.current();
    if (
      arm !== 'reactivate' &&
      benchmarkDigest(postInterventionSnapshot) !== benchmarkDigest(sourceSnapshot)
    )
      throw new Error('isolated repair mutated the plan contract');
    afterIntervention = bindingState(world.store.current(), world.observation, resolver);
    actualBindingChanges = afterIntervention.filter(
      (group, i) =>
        benchmarkDigest(group.enemyIds) !== benchmarkDigest(beforeIntervention![i].enemyIds),
    ).length;
  }
  while (actions.length < horizon && alive()) step();
  const status = world.environment.status();
  const final = timeline.at(-1) ?? initialMeasure;
  const newCompletions = final.frozenTargets.filter(
    (group, i) => group.eliminated && !initialMeasure.frozenTargets[i].eliminated,
  ).length;
  return {
    format: MECHANISM_VERSION,
    fixtureDigest: fixture.digest,
    seed: fixture.base.configuration.seed,
    roster: `${fixture.base.configuration.blueUnits}v${fixture.base.configuration.redUnits}`,
    arm,
    delayDecisions,
    horizon,
    interventionTick,
    interventionStatus,
    bindingRevision,
    actualBindingChanges,
    sourceSnapshot,
    postInterventionSnapshot,
    beforeIntervention,
    afterIntervention,
    initialMeasure,
    actions,
    stateHashes,
    timeline,
    opportunisticActions,
    metrics: {
      winner: status.winner,
      terminated: status.terminated,
      truncated: status.truncated,
      censored: !status.terminated && !status.truncated,
      blueAlive: status.blueAlive,
      redAlive: status.redAlive,
      decisions: actions.length,
      damageDealt: teamHealth(source.enemies) - teamHealth(world.observation.enemies),
      damageReceived: teamHealth(source.allies) - teamHealth(world.observation.allies),
      issuedActions: issued,
      rejectedActions: rejected,
      rejectedActionRate: rejected / Math.max(issued, 1),
      canonicalReturn,
      opportunisticShots: opportunisticActions.length,
      newFrozenTargetCompletions: newCompletions,
      meanOriginalRangeOccupancy: mean(timeline.map((row) => row.originalRangeOccupancy)),
      meanActiveRangeOccupancy: mean(timeline.map((row) => row.activeRangeOccupancy)),
      sourceAgeAtInterventionSeconds:
        interventionTick === null ? null : (interventionTick - source.tick) / source.simulationHz,
    },
  };
}

export type MechanismResult = ReturnType<typeof runMechanismBranch>;
export function firstChangedAction(
  result: MechanismResult,
  baseline: MechanismResult,
): number | null {
  for (let i = 0; i < Math.max(result.actions.length, baseline.actions.length); i++)
    if (benchmarkDigest(result.actions[i] ?? null) !== benchmarkDigest(baseline.actions[i] ?? null))
      return i + 1;
  return null;
}
function bindingState(snapshot: PlanSnapshot, observation: Observation, resolver: TargetResolver) {
  return snapshot.plan.groups.map((group) => {
    const objective = resolver.refresh(
      group.objective,
      observation,
      snapshot.plan.groups.map((row) => row.assignment),
    );
    return {
      role: group.role,
      objective,
      enemyIds:
        objective.kind === 'enemy_cluster'
          ? [...objective.enemyIds]
          : observation.enemies.filter((unit) => unit.alive).map((unit) => unit.id),
    };
  });
}
function rangeOccupancy(
  snapshot: PlanSnapshot,
  observation: Observation,
  bindings: readonly { role: string; enemyIds: readonly number[] }[],
): number | null {
  let living = 0;
  let inRange = 0;
  for (const group of snapshot.plan.groups) {
    const ids = bindings.find((row) => row.role === group.role)?.enemyIds ?? [];
    const targets = observation.enemies.filter((unit) => unit.alive && ids.includes(unit.id));
    for (const unit of observation.allies.filter(
      (unit) => unit.alive && group.assignment.unitIds.includes(unit.id),
    )) {
      living++;
      if (
        targets.some(
          (enemy) =>
            Math.hypot(unit.x - enemy.x, unit.y - enemy.y) <=
            throwRange(group.command.order.engagement.preferredRange),
        )
      )
        inRange++;
    }
  }
  return living ? inRange / living : null;
}
function teamHealth(units: readonly UnitObservation[]) {
  return units.reduce((sum, unit) => sum + (unit.alive ? Math.max(0, unit.health) : 0), 0);
}
function mean(values: (number | null)[]) {
  const known = values.filter((x): x is number => x !== null);
  return known.length ? known.reduce((a, b) => a + b, 0) / known.length : null;
}
