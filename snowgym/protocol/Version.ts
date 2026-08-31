/** Version of deterministic simulation behavior, independent of HTTP/Gym versions. */
export const SIMULATION_VERSION = 'snowgym.sim.v1' as const;

/** SnowCraft commit from which the SnowGym extension work began. */
export const UPSTREAM_BASE_COMMIT = '7d9fca5' as const;

/** Version of the canonical public-state serialization used for regression hashes. */
export const STATE_HASH_VERSION = 'snowgym.state.v1' as const;
