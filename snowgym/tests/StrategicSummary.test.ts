import { describe, expect, it } from 'vitest';
import type { Observation, UnitObservation } from '../observations/Observation';
import { hashObservation } from '../reproducibility/StateHash';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlan,
  type CommandPlanEnvelope,
} from '../orchestration/command/CommandPlan';
import {
  STRATEGIC_SUMMARY_VERSION,
  summarizeStrategy,
} from '../orchestration/commander/StrategicSummary';
import { PlanGrounder } from '../orchestration/grounding/PlanGrounder';
import { PlanStore } from '../orchestration/runtime/PlanStore';
import { observationWith } from './orchestrationTestHelpers';

describe('summarizeStrategy', () => {
  it('produces compact ID-free force, threat, terrain, and assignment summaries', () => {
    const observation: Observation = {
      ...observationWith({
        allies: [unit(1, 'blue', -8, -2), unit(2, 'blue', -8, 2, false)],
        enemies: [unit(100, 'red', 8, -1), unit(101, 'red', 8, 1, true, 50)],
      }),
      tick: 30,
      projectiles: [
        {
          id: 500,
          ownerId: 100,
          team: 'red',
          x: 0,
          y: 0,
          vx: -1,
          vy: 0,
          height: 1,
          heightVelocity: 0,
        },
      ],
      obstacles: [
        {
          id: 600,
          type: 'tree',
          x: 0,
          y: 0,
          halfWidth: 1,
          halfHeight: 1,
          blocksSight: true,
          blocksProjectiles: true,
          blocksMovement: true,
        },
      ],
    };
    const store = storeWith(oneGroupPlan(), observation);

    const summary = summarizeStrategy(observation, store.current());

    expect(summary).toMatchObject({
      schemaVersion: STRATEGIC_SUMMARY_VERSION,
      sourceTick: 30,
      sourceStateHash: hashObservation(observation),
      arena: { width: 40, height: 30, obstacleCount: 1 },
      ownForce: { alive: 1, healthFraction: 1, centroid: { x: -8, y: -2 }, spread: 0 },
      enemyForce: { alive: 2, healthFraction: 0.75, centroid: { x: 8, y: 0 }, spread: 1 },
      hostileProjectileCount: 1,
      groups: [
        {
          role: 'main',
          mission: 'engage',
          assigned: 1,
          living: 1,
          objectiveKind: 'enemy_cluster',
        },
      ],
    });
    expect(JSON.stringify(summary)).not.toContain('"unitIds"');
    expect(JSON.stringify(summary)).not.toContain('"enemyIds"');
  });
});

function storeWith(decision: CommandPlan, observation: Observation): PlanStore {
  const envelope: CommandPlanEnvelope = {
    planId: 'summary-plan',
    source: { requestId: 'summary-request', sourceTick: observation.tick },
    decision,
  };
  return new PlanStore(new PlanGrounder().ground(envelope, observation), observation.tick);
}

function oneGroupPlan(): CommandPlan {
  return {
    schemaVersion: COMMAND_PLAN_VERSION,
    intentSummary: null,
    groups: [
      {
        role: 'main',
        allocationWeight: 1,
        selection: 'balanced',
        order: {
          mission: 'engage',
          objective: { kind: 'enemy_cluster', select: 'nearest' },
          approach: 'direct',
          engagement: {
            posture: 'balanced',
            fire: 'focus',
            preferredRange: 'medium',
            cohesion: 'normal',
          },
        },
      },
    ],
  };
}

function unit(
  id: number,
  team: UnitObservation['team'],
  x: number,
  y: number,
  alive = true,
  health = 100,
): UnitObservation {
  return {
    id,
    team,
    x,
    y,
    vx: 0,
    vy: 0,
    health: alive ? health : 0,
    maxHealth: 100,
    alive,
    state: alive ? 'idle' : 'defeated',
    throwCooldown: 0,
    charge: 0,
  };
}
