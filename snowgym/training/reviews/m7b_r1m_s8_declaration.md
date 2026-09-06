# R1m-S8: same-state continuation after squad correction

Commit this protocol and tested implementation before running the experiment.
Reuse all 24 S6 training seeds and their archived squad-30 trajectories. Include
every trajectory with a nonterminal state after exactly 30 correction decisions;
exclude only those terminating at or before that boundary, recording the reason.
Do not select on final success, remaining health, or budget. S7 indicates 23
eligible states and one pre-handoff timeout. These are previously exposed
training states, not new development or qualification environments.

Restore the original exact first-hit prefix and replay the 30 archived correction
actions. Validate every correction transition's hash, reward, option state,
action results and nontermination. Both arms start with identical physical,
plan, option and observation identities. Preserve the original 200-decision
option horizon, activated target membership, and all reward definitions.

Two arms, each executed twice independently:

- `source`: frozen R1h source with the same corrected shots as S6. Generate its
  actions and require exact equality with every saved S6 squad-30 suffix action,
  state hash, reward, option record, action result and terminal boundary.
- `move-rest`: same frozen classifier and corrected shots. At each remaining
  decision replace only living source-selected MOVE destinations with the
  production-compatible conditional teacher recommendation on that branch's
  current state. Do not replace other action dimensions or action choices.

Recommendation availability, action legality and shot readiness remain separate.
Verify read-only labeling, helper agreement and no source parameter changes.
Record complete restore prefixes, start identities, actions, source actions,
hashes, option records, reward components, damage, per-unit geometry, override
dose and final outcomes. Count actual prefix-plus-continuation decisions across
both repeats; cap at 19,200 (=24 seeds x 2 arms x 2 repeats x 200). Tests are
separate reproducibility checks, not extra observations in the experiment.
Refuse existing output directories and bind source/data/implementation digests
in a self-verifying manifest. Recheck S3/S5/S6/S7 lineage after collection.

Primary comparison is move-rest minus source on final mission success and full
discounted executor return from the handoff state. Average paired differences
over eligible seeds, with 10,000 paired bootstrap samples and RNG 992001.
The diagnostic continuation gate requires mean success gain >=10 percentage
points, strictly positive 95% lower bounds for both success and return gains,
and rejected actions below 0.1% in both arms. No alternative contrast replaces
that gate. Repeats verify determinism and are not independent environments.

Report damage dealt/received, final progress, survival, exposure, range occupancy
within 9 and error from 6.5, readiness/action choice and actual correction dose
as secondary descriptive metrics. Range references are teacher conventions,
not hit guarantees. Report recovered and lost seeds. Preserve negative results;
no failed episode receives extra time. Opponent actions may diverge once the
Blue trajectories diverge despite the equal seed and initial state.

All artifacts are explicitly teacher-assisted and autonomous-ineligible. A
passed gate supports predeclaring a sustained learned-movement comparison; it
does not qualify a fighter or launch training. A failed gate calls for inspecting
remaining targeting/timing/pursuit evidence before another PPO run. No provider
calls, browser, new action decoder, reward changes, PPO sweep or commander work.

Before implementation and results commits: targeted tests of exact handoff,
prefix tamper rejection, source-suffix parity, action-channel isolation,
casualties/missing labels, repeats, budgets and report arithmetic; then npm test,
npm run build, Python client tests and Python training tests.
