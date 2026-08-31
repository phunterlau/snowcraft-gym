export const COMMAND_PLAN_VERSION = 'snowgym.command-plan.v0' as const;

export const GROUP_ROLES = ['main', 'maneuver', 'reserve'] as const;
export type GroupRole = (typeof GROUP_ROLES)[number];

export const GROUP_SELECTIONS = [
  'balanced',
  'frontline',
  'rearline',
  'healthiest',
  'nearest_objective',
  'nearest_left_lane',
  'nearest_right_lane',
] as const;
export type GroupSelection = (typeof GROUP_SELECTIONS)[number];

export const ENEMY_CLUSTER_SELECTORS = [
  'nearest',
  'largest',
  'weakest',
  'leftmost',
  'rightmost',
] as const;
export type EnemyClusterSelector = (typeof ENEMY_CLUSTER_SELECTORS)[number];

export const REGIONS = [
  'left_lane',
  'center_lane',
  'right_lane',
  'own_backfield',
  'enemy_backfield',
] as const;
export type Region = (typeof REGIONS)[number];

export const APPROACHES = ['direct', 'left_flank', 'right_flank', 'avoid_center'] as const;
export type Approach = (typeof APPROACHES)[number];

export const POSTURES = ['aggressive', 'balanced', 'conservative'] as const;
export type Posture = (typeof POSTURES)[number];

export const FIRE_POLICIES = ['focus', 'distributed', 'opportunistic'] as const;
export type FirePolicy = (typeof FIRE_POLICIES)[number];

export const PREFERRED_RANGES = ['close', 'medium', 'long'] as const;
export type PreferredRange = (typeof PREFERRED_RANGES)[number];

export const COHESION_LEVELS = ['tight', 'normal', 'loose'] as const;
export type Cohesion = (typeof COHESION_LEVELS)[number];

export interface EngagementPolicy {
  readonly posture: Posture;
  readonly fire: FirePolicy;
  readonly preferredRange: PreferredRange;
  readonly cohesion: Cohesion;
}

export interface EnemyClusterObjective {
  readonly kind: 'enemy_cluster';
  readonly select: EnemyClusterSelector;
}

export interface RegionObjective {
  readonly kind: 'region';
  readonly region: Region;
}

export interface CurrentPositionObjective {
  readonly kind: 'current_position';
}

export interface AllyGroupObjective {
  readonly kind: 'ally_group';
  readonly role: GroupRole;
}

export interface EngageOrder {
  readonly mission: 'engage';
  readonly objective: EnemyClusterObjective;
  readonly approach: Approach;
  readonly engagement: EngagementPolicy;
}

export interface AdvanceOrder {
  readonly mission: 'advance';
  readonly objective: RegionObjective;
  readonly approach: Approach;
  readonly engagement: EngagementPolicy;
}

export interface HoldOrder {
  readonly mission: 'hold';
  readonly objective: RegionObjective | CurrentPositionObjective;
  readonly approach: 'direct';
  readonly engagement: EngagementPolicy;
}

export interface WithdrawOrder {
  readonly mission: 'withdraw';
  readonly objective: { readonly kind: 'region'; readonly region: 'own_backfield' };
  readonly approach: 'direct' | 'avoid_center';
  readonly engagement: EngagementPolicy;
}

export interface SupportOrder {
  readonly mission: 'support';
  readonly objective: AllyGroupObjective;
  readonly approach: Approach;
  readonly engagement: EngagementPolicy;
}

export type GroupOrder = EngageOrder | AdvanceOrder | HoldOrder | WithdrawOrder | SupportOrder;

export interface GroupCommand {
  readonly role: GroupRole;
  /** Relative share of the living roster. The allocator normalizes all group weights. */
  readonly allocationWeight: number;
  readonly selection: GroupSelection;
  readonly order: GroupOrder;
}

/** The complete model-owned output. IDs, timing, coordinates, and provenance are host-owned. */
export interface CommandPlan {
  readonly schemaVersion: typeof COMMAND_PLAN_VERSION;
  /** Trace-only explanation. Null keeps the strict model-output schema compact when unused. */
  readonly intentSummary: string | null;
  readonly groups: readonly GroupCommand[];
}

export interface CommandPlanSource {
  readonly requestId: string;
  readonly sourceTick: number;
  readonly sourceStateHash?: string;
}

/** Trusted host envelope added only after model output has passed validation. */
export interface CommandPlanEnvelope {
  readonly planId: string;
  readonly source: CommandPlanSource;
  readonly decision: CommandPlan;
}

export const GROUP_ROLE_ORDER: Readonly<Record<GroupRole, number>> = {
  main: 0,
  maneuver: 1,
  reserve: 2,
};
