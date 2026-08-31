import { EventBus } from '../../src/core/EventBus';
import { IdAllocator } from '../../src/ecs/Entity';
import { SIM } from '../../src/game/config';
import { Team } from '../../src/game/types';
import { World } from '../../src/game/World';
import type { AiDifficulty } from '../../src/systems/AISystem';
import { CollisionSystem } from '../../src/systems/CollisionSystem';
import { DamageSystem } from '../../src/systems/DamageSystem';
import { MovementSystem } from '../../src/systems/MovementSystem';
import { ProjectileSystem } from '../../src/systems/ProjectileSystem';
import { RoundSystem } from '../../src/systems/RoundSystem';
import { ThrowSystem } from '../../src/systems/ThrowSystem';
import type { ActionResult } from '../adapters/SnowCraftActionAdapter';
import { SnowCraftActionAdapter } from '../adapters/SnowCraftActionAdapter';
import type { TeamAction } from '../actions/UnitAction';
import {
  createRedController,
  DEFAULT_RED_CONTROLLER,
  type RedControllerType,
} from '../agents/opponents';
import type { TeamController } from '../agents/TeamController';
import { observeWorld, type Observation } from '../observations/Observation';
import { SIMULATION_VERSION, STATE_HASH_VERSION, UPSTREAM_BASE_COMMIT } from '../protocol/Version';
import { hashObservation } from '../reproducibility/StateHash';
import { buildArena, THREE_VS_THREE_OPEN, type Scenario } from '../scenarios/Scenario';

export interface EnvironmentConfig {
  scenario?: Scenario;
  decisionHz?: number;
  redDifficulty?: AiDifficulty;
  redController?: RedControllerType;
}

export interface EnvironmentStatus {
  apiVersion: 'snowgym.v0';
  simulationVersion: typeof SIMULATION_VERSION;
  stateHashVersion: typeof STATE_HASH_VERSION;
  upstreamBaseCommit: typeof UPSTREAM_BASE_COMMIT;
  stateHash: string;
  scenario: string;
  seed: number;
  tick: number;
  simulationHz: number;
  decisionHz: number;
  ticksPerDecision: number;
  configuration: {
    blueUnits: number;
    redUnits: number;
    arenaWidth: number;
    arenaHeight: number;
    maxTicks: number;
    redDifficulty: AiDifficulty;
    redController: RedControllerType;
    map: string | null;
  };
  blueAlive: number;
  redAlive: number;
  terminated: boolean;
  truncated: boolean;
  winner: 'blue' | 'red' | null;
}

export interface StepInfo extends EnvironmentStatus {
  actionResults: ActionResult[];
}

export interface StepResult {
  observation: Observation;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: StepInfo;
}

/**
 * DOM-free, Gym-like simulation host. One call to {@link step} applies a blue
 * team action and advances several fixed physics ticks while the red team is
 * driven by a TeamController (scripted squad AI by default).
 */
export class SnowEnvironment {
  readonly scenario: Scenario;
  readonly decisionHz: number;
  readonly ticksPerDecision: number;

  private seed: number;
  private tick = 0;
  private truncated = false;
  private world!: World;
  private events!: EventBus;
  private throwing!: ThrowSystem;
  private movement!: MovementSystem;
  private redController!: TeamController;
  private projectile!: ProjectileSystem;
  private collision!: CollisionSystem;
  private damage!: DamageSystem;
  private round!: RoundSystem;
  private actionAdapter!: SnowCraftActionAdapter;
  private readonly redDifficulty: AiDifficulty;
  private readonly redControllerType: RedControllerType;

  constructor(config: EnvironmentConfig = {}) {
    this.scenario = config.scenario ?? THREE_VS_THREE_OPEN;
    this.decisionHz = config.decisionHz ?? 10;
    this.ticksPerDecision = validateDecisionFrequency(this.decisionHz);
    this.redDifficulty = config.redDifficulty ?? 'normal';
    this.redControllerType = config.redController ?? DEFAULT_RED_CONTROLLER;
    this.seed = this.scenario.seed;
    this.reset(this.seed);
  }

  reset(seed = this.scenario.seed): Observation {
    if (!Number.isSafeInteger(seed)) throw new RangeError('seed must be a safe integer');

    this.seed = seed;
    this.tick = 0;
    this.truncated = false;
    this.events = new EventBus();
    this.world = new World(buildArena(this.scenario, new IdAllocator()), seed);
    // No reserve lives means blue elimination is terminal without registering
    // the browser game's single-hero RespawnSystem.
    this.world.playerLives = 0;
    this.world.playerLivesMax = 0;
    for (const spawn of this.scenario.blueSpawns) {
      this.world.addPlayer(Team.Player, spawn.x, spawn.y);
    }
    for (const spawn of this.scenario.redSpawns) {
      this.world.addPlayer(Team.Enemy, spawn.x, spawn.y);
    }

    this.throwing = new ThrowSystem(this.world, this.events);
    this.movement = new MovementSystem(this.world);
    this.redController = createRedController(
      { type: this.redControllerType, difficulty: this.redDifficulty },
      this.world,
      this.throwing,
      Team.Enemy,
      Team.Player,
      this.redDifficulty,
    );
    this.projectile = new ProjectileSystem(this.world, this.events);
    this.collision = new CollisionSystem(this.world, this.events);
    this.damage = new DamageSystem(this.world, this.events);
    this.round = new RoundSystem(this.world, this.events);
    this.actionAdapter = new SnowCraftActionAdapter(this.world, this.movement, this.throwing);

    return this.observe(Team.Player);
  }

  observe(team: Team): Observation {
    return observeWorld(this.world, team, this.tick);
  }

  step(blueAction: TeamAction): StepResult {
    if (this.round.isOver || this.truncated) {
      throw new EpisodeCompleteError('reset the environment before stepping a completed episode');
    }

    const actionResults = this.actionAdapter.apply(Team.Player, blueAction);
    for (let i = 0; i < this.ticksPerDecision && !this.round.isOver; i++) {
      this.physicsStep();
      if (this.tick >= this.scenario.maxTicks) {
        this.truncated = !this.round.isOver;
        break;
      }
    }

    const status = this.status();
    return {
      observation: this.observe(Team.Player),
      reward: terminalReward(status.winner, status.terminated),
      terminated: status.terminated,
      truncated: status.truncated,
      info: { ...status, actionResults },
    };
  }

  status(): EnvironmentStatus {
    const observation = this.observe(Team.Player);
    return {
      apiVersion: 'snowgym.v0',
      simulationVersion: SIMULATION_VERSION,
      stateHashVersion: STATE_HASH_VERSION,
      upstreamBaseCommit: UPSTREAM_BASE_COMMIT,
      stateHash: hashObservation(observation),
      scenario: this.scenario.name,
      seed: this.seed,
      tick: this.tick,
      simulationHz: SIM.hz,
      decisionHz: this.decisionHz,
      ticksPerDecision: this.ticksPerDecision,
      configuration: {
        blueUnits: this.scenario.blueSpawns.length,
        redUnits: this.scenario.redSpawns.length,
        arenaWidth: this.scenario.arena.width,
        arenaHeight: this.scenario.arena.height,
        maxTicks: this.scenario.maxTicks,
        redDifficulty: this.redDifficulty,
        redController: this.redControllerType,
        map: this.scenario.map ?? null,
      },
      blueAlive: this.world.countLiving(Team.Player),
      redAlive: this.world.countLiving(Team.Enemy),
      terminated: this.round.isOver,
      truncated: this.truncated,
      winner: teamName(this.round.result),
    };
  }

  private physicsStep(): void {
    this.world.time += SIM.dt;
    // The red TeamController holds internal per-tick state (decision timers,
    // dodges), so physicsStep delegates to the composed red behavior rather
    // than calling controller.act once per decision. The reported semantic
    // actions stay on the controller for inspection; applying them through the
    // adapter is unnecessary for the scripted bridge (its orders already
    // reached the world) and tryThrow cannot be re-issued without
    // double-firing a snowball.
    this.redController.act(observeWorld(this.world, Team.Enemy, this.tick), SIM.dt);
    this.movement.update(SIM.dt);
    this.throwing.update(SIM.dt);
    this.projectile.update(SIM.dt);
    this.collision.update(SIM.dt);
    this.damage.update(SIM.dt);
    this.round.update();
    this.world.reclaimSnowballs();
    this.tick++;
  }
}

export class EpisodeCompleteError extends Error {}

function validateDecisionFrequency(decisionHz: number): number {
  if (!Number.isInteger(decisionHz) || decisionHz <= 0 || decisionHz > SIM.hz) {
    throw new RangeError(`decisionHz must be an integer in [1, ${SIM.hz}]`);
  }
  if (SIM.hz % decisionHz !== 0) {
    throw new RangeError(`decisionHz must divide the ${SIM.hz} Hz simulation rate exactly`);
  }
  return SIM.hz / decisionHz;
}

function teamName(team: Team | null): 'blue' | 'red' | null {
  if (team === null) return null;
  return team === Team.Player ? 'blue' : 'red';
}

function terminalReward(winner: EnvironmentStatus['winner'], terminated: boolean): number {
  if (!terminated) return 0;
  return winner === 'blue' ? 1 : -1;
}
