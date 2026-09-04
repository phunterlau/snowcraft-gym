/** Version of deterministic simulation behavior, independent of HTTP/Gym versions. */
export const SIMULATION_VERSION = 'snowgym.sim.v2' as const;

/** SnowCraft commit from which the SnowGym extension work began. */
export const UPSTREAM_BASE_COMMIT = '7d9fca5' as const;

/** Legacy public-state serialization retained for committed replay verification. */
export const LEGACY_STATE_HASH_VERSION = 'snowgym.state.v1' as const;

/** Version of the canonical actuator-complete public-state serialization. */
export const STATE_HASH_VERSION = 'snowgym.state.v2' as const;

export type StateHashVersion =
  | typeof LEGACY_STATE_HASH_VERSION
  | typeof STATE_HASH_VERSION;
