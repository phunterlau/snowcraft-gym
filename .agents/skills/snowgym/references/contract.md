# Guarded JSON contract

The server binds to `127.0.0.1:8787` and owns one shared mutable episode.

1. Read `GET /capabilities` for supported maps, capacities, versions, action
   fields, decision rates, and Gym IDs.
2. Read `GET /status` immediately before a mutation.
3. Send the returned `status.stateHash` as `expectedStateHash`.
4. Generate a unique `idempotencyKey` for each logical mutation. Reuse that key
   only when retrying the exact same request.

Mutations reject unknown fields. `POST /step` requires `action`; the built-in
policy is isolated at `POST /step-scripted`. A stale hash returns HTTP 409
`stale_state`. Reusing a key for different input returns HTTP 409
`idempotency_conflict`. An exact duplicate returns the cached response without
advancing again.

Canonical actions are `noop(type, unitId)`, `move(type, unitId, x, y)`, and
`throw(type, unitId, x, y, power)`. Coordinates are world-space. A `noop` and
an omitted unit retain any prior movement order; neither means hold position.

The Python Gym client applies state-hash and idempotency guards automatically.
