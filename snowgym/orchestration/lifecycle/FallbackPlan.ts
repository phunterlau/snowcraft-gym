import type { Observation } from '../../observations/Observation';
import { hashObservation } from '../../reproducibility/StateHash';
import {
  COMMAND_PLAN_VERSION,
  type CommandPlanEnvelope,
  type GroupOrder,
} from '../command/CommandPlan';

/** Builds a deterministic one-group doctrine when no commander plan is viable. */
export function createFallbackEnvelope(
  observation: Observation,
  reason: string,
  sequence: number,
): CommandPlanEnvelope {
  const suffix = `${observation.tick}-${sequence}`;
  return {
    planId: `fallback-${suffix}`,
    source: {
      requestId: `fallback-${normalizedReason(reason)}-${suffix}`,
      sourceTick: observation.tick,
      sourceStateHash: hashObservation(observation),
    },
    decision: {
      schemaVersion: COMMAND_PLAN_VERSION,
      intentSummary: `Deterministic fallback: ${reason}`.slice(0, 160),
      groups: [
        {
          role: 'main',
          allocationWeight: 1,
          selection: 'balanced',
          order: fallbackOrder(observation),
        },
      ],
    },
  };
}

export function fallbackOrder(observation: Observation): GroupOrder {
  if (observation.enemies.some(({ alive }) => alive)) {
    return {
      mission: 'engage',
      objective: { kind: 'enemy_cluster', select: 'nearest' },
      approach: 'direct',
      engagement: {
        posture: 'balanced',
        fire: 'distributed',
        preferredRange: 'medium',
        cohesion: 'normal',
      },
    };
  }
  return {
    mission: 'hold',
    objective: { kind: 'current_position' },
    approach: 'direct',
    engagement: {
      posture: 'conservative',
      fire: 'opportunistic',
      preferredRange: 'long',
      cohesion: 'tight',
    },
  };
}

function normalizedReason(reason: string): string {
  const normalized = reason
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
  return normalized || 'unspecified';
}
