import type { System } from '../../src/ecs/System';
import { SIM } from '../../src/game/config';
import type { Team } from '../../src/game/types';
import type { World } from '../../src/game/World';
import type { SnowCraftActionAdapter, ActionResult } from '../adapters/SnowCraftActionAdapter';
import type { TeamController } from '../agents/TeamController';
import { observeWorld } from '../observations/Observation';

/** Runs a team policy at a lower decision frequency than the physics systems. */
export class TeamControllerSystem implements System {
  readonly name: string;
  private ticks = 0;
  private ticksUntilDecision = 0;
  private readonly ticksPerDecision: number;
  lastResults: readonly ActionResult[] = [];

  constructor(
    private readonly world: World,
    private readonly team: Team,
    private readonly controller: TeamController,
    private readonly adapter: SnowCraftActionAdapter,
    decisionHz = 10,
  ) {
    if (!Number.isFinite(decisionHz) || decisionHz <= 0 || decisionHz > SIM.hz) {
      throw new RangeError(`decisionHz must be in (0, ${SIM.hz}]`);
    }
    this.name = `snowgym-controller-${team}`;
    this.ticksPerDecision = Math.max(1, Math.round(SIM.hz / decisionHz));
  }

  update(): void {
    this.ticks++;
    if (this.ticksUntilDecision > 0) {
      this.ticksUntilDecision--;
      return;
    }

    const observation = observeWorld(this.world, this.team, this.ticks);
    this.lastResults = this.adapter.apply(this.team, this.controller.act(observation, SIM.dt));
    this.ticksUntilDecision = this.ticksPerDecision - 1;
  }
}
