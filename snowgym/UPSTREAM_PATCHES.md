# Upstream patch ledger

## `refs/snowgym_m8_s3_dev_notes.md`

Reason: the user asked for a note on every development step. Local log for
M8-S3, the 3v3 plan-conditioned path audit on the batch transport.

Change: dated entries covering the transport decision, checks, and gates. The
file stays local; the committed protocol is
`snowgym/training/reviews/m8_s3_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_m8_s2_dev_notes.md`

Reason: the user asked for a note on every development step. Local log for
M8-S2, the contract repair of `SnowGymUnitParallelEnv` after an external review.

Change: dated entries covering the review's verification, decisions, and
gates. The file stays local; the committed protocol is
`snowgym/training/reviews/m8_s2_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_handoff_2026-09-20.md`

Reason: user-requested handoff of what ran since the 2026-09-18 review, with
pending ideas. Corrected after review to demote R1n-i's movement finding.

Change: local note only; no source behavior, sealed evidence or thresholds
changed.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_handoff_review_and_next_steps_2026-09-20.md`

Reason: user-requested review of the September 20 handoff and pending ideas
against the objective of LLM orchestration of local learned agents.

Change: added a source-linked review of R1n-i and M8-S1, reproduced wrapper
contract issues, qualified causal/verification claims, and proposed a staged
training/command-conditioning path with PPO, observability and latency contracts.
No source behavior, sealed evidence, thresholds or existing notes were changed;
no new research collection, training, provider call, commit or push was performed.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_m8_s1_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for M8-S1, the new per-unit PettingZoo environment
(`SnowGymUnitParallelEnv`) that begins the M8 unit-level CTDE/MAPPO
milestone.

Change: added dated entries per commit or run, covering decisions,
deviations, verification, and open questions. The file stays local; the
committed protocol is `snowgym/training/reviews/m8_s1_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_since_handoff_review_2026-09-18.md`

Reason: user-requested review of changes since the last whole-repository handoff.

Change: added a source-linked R1n-b through R1n-h review, current local/remote
revision boundaries, repaired training contracts, PPO interpretation, fresh
archive-only mixture failure analysis, statistical and causal caveats, test
results, and proposed next decisions. Preserved sealed artifacts and existing
local notes; no research training, provider call, commit, or push was performed.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_rl_llm_cotraining_plan.md`

Reason: save the requested RL/LLM co-training discussion as a standalone local note.

Change: documented the proposed hierarchy, mathematical objectives, alternating
training sequence, evidence boundaries and evaluation requirements. The note
does not approve new experiments or change roadmap qualification gates.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_selective_repair_review_and_plan.md`

Reason: preserve the requested loss-recovery interpretation and predeclare the
selective tactical repair mechanism audit before collecting new results.

Change: added a source-linked review, isolated intervention definitions, fixed
seed/delay budgets and verification gates. Upstream behavior is unchanged.

SnowGym dependency: documents a headless scripted-executor diagnostic without
changing command semantics or fighter qualification thresholds.

## `src/systems/AISystem.ts`

Reason: the red squad had to run behind SnowGym's `TeamController` boundary
without changing its behavior in the browser game.

Change: added an optional `AiSquad` parameter (`{ controlled, target }`) to the
constructor, defaulting to the classic `{ Enemy, Player }` pairing, and widened
the `events` parameter to accept `null` (it was already unused). The three
hardcoded `Team.Enemy`-drives-`Team.Player` checks now read from the squad.

Upstream behavior: unchanged; the browser game constructs `AISystem` with the
default squad and identical difficulty tuning.

SnowGym dependency: `ScriptedAiAgent` re-runs the classic AI per-tick inside a
`TeamController` and reports its orders as semantic actions; full-episode
traces are bit-identical to direct registration.

## `src/systems/MovementSystem.ts`

Reason: semantic controllers need to order one known unit without selection or
UI commands.

Change: added public `tryMove(player, x, y)`, routed the existing group move
command through it, and made rejected state-incompatible orders atomic. Later
added public `tryHold(player)` so a controller that reports orders on behalf of
a unit can cancel a stale move target without disturbing other states. M7a also
added the read-only `currentSteeringTarget(player)` seam so renderer-free full
state observations can expose the active path waypoint without owning pathfinding.

Upstream behavior: unchanged; group formation, clamping, state transitions, and
movement integration use the same logic as before.

SnowGym dependency: `SnowCraftActionAdapter` uses the command seams and
`SnowEnvironment` uses the waypoint seam for actuator-complete observations.

## `src/systems/ThrowSystem.ts`

Reason: arbitrary Gym actions can arrive while a unit is in a state that cannot
start a throw.

Change: `tryThrow` now rejects state-incompatible orders before acquiring or
launching a projectile.

Upstream behavior: valid human and AI throws are unchanged; invalid throws no
longer create a projectile before a failed state transition.

SnowGym dependency: keeps rejected semantic actions free of partial side
effects.

## `vite.config.ts`

Reason: SnowGym tests must participate in normal verification.

Change: added `snowgym/tests/**/*.test.ts` to Vitest discovery and added
`replay.html` as a second Vite build entry.

Upstream behavior: source test discovery remains enabled.

SnowGym dependency: keeps extension-layer tests in the standard test command
and builds the visual replay viewer without changing the normal game entry.

## `tsconfig.json`

Reason: the headless environment and server must be checked by the repository's
strict TypeScript command.

Change: added `snowgym` to the compiler include list.

Upstream behavior: existing `src` and configuration checking is unchanged.

SnowGym dependency: covers the new server and environment sources.

## `package.json` and `package-lock.json`

Reason: the TypeScript server needs a direct Node execution entry point and Node
type declarations.

Change: added `npm run snowgym:server`, `npm run snowgym:example`, the SnowGym
replay browser-smoke command, `tsx`, and `@types/node`.

Upstream behavior: existing browser scripts and runtime dependencies are
unchanged.

SnowGym dependency: runs `snowgym/server/main.ts` without a separate build
pipeline.

## `public/maps/arena6.json` and `src/main.ts`

Reason: the terrain-backed 10v10 SnowGym acceptance scenario needs a map whose
native schema contains ten valid spawn points for each team.

Change: added the 64x48 `Winter Front` arena with symmetric cover and 20 spawn
points, then registered it in the browser map menu.

Upstream behavior: existing maps and the single-player defaults are unchanged;
the browser continues to cap spawned units according to its match settings.

SnowGym dependency: the headless mirror in `snowgym/scenarios/maps.ts` feeds the
same JSON data to `MapLoader.build`, so server physics and browser rendering use
identical terrain and spawn definitions.

## `AGENTS.md` and `.agents/skills/snowgym/`

Related review documentation (2026-09-04): preserve the supplied R1j review at
`refs/snowgym_r1j_review_and_next_rl_experiments.md` and its standalone handoff
`refs/snowgym_rl_recovery_r1j_handoff_for_gpt_pro_2026-09-04.md`. Correct the
handoff's seed-exposure description using R1e's archived continuation manifest.
These documentation-only changes do not alter the upstream game.

Reason: coding agents and LLM policy operators need one discoverable source for
the repository boundary, safe mutation workflow, environment versions, and
verification commands.

Change: added a concise root agent guide plus a repo-local SnowGym skill with
guarded HTTP contract, troubleshooting, and a state-hash/idempotency-aware step
wrapper.

Upstream behavior: unchanged; these files are operational documentation and a
local command wrapper only.

SnowGym dependency: directs agents to the renderer-free interfaces, capability
endpoint, safe artifact workflow, and required verification gates.

## `AGENTS.md` (accepted `npm test` known failure)

Reason: the user accepted the selective-repair preflight collision caused by
the R1n-b archive as a documented known failure. Agents running the milestone
gate need the exact pass condition.

Change: added a paragraph under Verification naming the single allowed
failing test and colliding file. Any other failure still blocks.

Upstream behavior: unchanged; documentation only. No test, source, or
archive was modified.

## `refs/run_snowgym_demo.sh`

Reason: provide the requested one-command blue-team demo bootstrap without
requiring users to coordinate separate server and Gym client terminals.

Change: added a short script that resolves the repository root, prepares local
dependencies when absent, reuses or starts loopback SnowGym and Vite servers,
runs and records the Python demo, opens the recording in the existing replay UI,
forwards optional demo arguments, and cleans up only its temporary resources.

Upstream behavior: unchanged; the script is an opt-in development helper.

SnowGym dependency: combines the existing JSON server, `snowgym-demo` CLI, and
Three.js replay UI; it does not add another simulation or rendering path.

## `package.json`, `src/main.ts`, and `src/physics/Pathfinding.ts` (map generator)

Reason: the constrained map-generation side tool needs a root entry point,
explicit promotion into the browser catalog, and authoritative reachability
checks for blockers whose player-radius inflation reaches an arena edge.

Change: added `npm run snowgym:mapgen`, a generated-map insertion marker in the
browser map list, and clamped partially out-of-grid inflated obstacle bounds in
the existing path grid instead of dropping those obstacles from pathfinding.

Upstream behavior: existing maps, menu entries, and controls are unchanged. The
pathfinding correction only affects blockers touching an arena boundary; they
now remain blockers instead of disappearing from the coarse grid.

SnowGym dependency: generated maps remain isolated artifacts until an explicit
promotion, and their deterministic playability validator uses the same path
geometry as simulation movement.

## `refs/snowgym_rl_ppo_review_2026-09-04.md`

Reason: document the requested review of the current fighter RL/PPO approach,
results, and remaining issues against source and archived evidence.

Change: added a standalone review covering R1e results, reproducible contract
and measurement findings, teacher-state diagnostics, and recommended next checks.

Upstream behavior: unchanged; this is review documentation only.

SnowGym dependency: records evidence for future recovery work without changing
training code, historical artifacts, or milestone acceptance thresholds.

## `refs/snowgym_r1m_s4_ppo_action_space_review.md`

Reason: preserve the requested S3/S4 PPO mathematical review and discussion of
destination-space versus physical-control-space design.

Change: added an evidence-backed review with equations, archive-only diagnostic
commands, action-interface tradeoffs and a proposed gated physical probe.

Upstream behavior: unchanged; documentation only. Existing review and run
artifacts, training contracts and milestone gates remain untouched.

## `refs/tiny-doom-defender/` and `refs/tiny_doom_defender_ppo_review_2026-09-08.md`

Reason: user-requested read-only reference checkout and review of its Doom PPO
training pipeline in relation to SnowGym fighter control.

Change: cloned the public Apache-2.0 repository at
`b2e8daa3b9259f2c7f2975354b185b11f2961044`, without changing its working tree,
and added a standalone source-pinned review. No dependencies, weights, training
or gameplay were executed; the clone is not registered as a submodule.

Upstream behavior: unchanged. These are reference materials only; SnowGym source,
training contracts, historical evidence and qualification gates remain unchanged.

## `refs/snowgym_fighter_rl_ppo_handoff_for_claude_opus_2026-09-12.md`

Reason: user-requested standalone design, findings and source-code handoff for
an external Claude Opus review of fighter RL/PPO.

Change: added a source-linked review packet covering the environment/action
contract, assisted movement policy, reward and PPO mathematics, historical
interventions, S9/S10 evidence, unresolved hypotheses and qualification gates.
It records the local/remote snapshot difference so local-only evidence is not
mistaken for published code.

Upstream behavior: unchanged; documentation only. No training, provider request,
source modification, checkpoint promotion or protocol change was performed.

## `refs/snowgym_fighter_rl_ppo_review_claude_opus_2026-09-12_reanalysis.py`

Reason: the committed R1m-S11 declaration/results and R1n results cite this
read-only reanalysis script (for example, its section C Monte Carlo critic-fit
check), so it is committed as code. Its companion review note stays local.

Change: added the reviewer's archive-only reanalysis of R1m S5/S9/S10
artifacts. It reads archived runs only: no simulator, provider call, or policy
update.

Upstream behavior: unchanged; analysis script only.

## `refs/snowgym_whole_repo_handoff_2026-09-13.md`

Reason: user-requested whole-repository handoff from the research objective
through the latest implementation and remaining milestones.

Change: added an architecture, workflow, source and evidence guide covering
headless control, replay, commander orchestration, map generation, historical
BC/PPO gates and the new S11/S12/E3 experiments. It distinguishes reusable
checkpoints from assisted measurements and stopped runs, records the published
snapshot with direct key-file GitHub links, and flags source/declaration and
artifact-availability gaps. Historical performance is context rather than the
criterion for selecting the implementation described.

Upstream behavior: unchanged; documentation only. Existing notes, source,
protocols and research artifacts were preserved. No new experiment, browser,
provider request, commit or push was performed for this handoff.

## `refs/snowgym_r1n_i_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for R1n-i, the frozen-checkpoint failure diagnosis
reading R1n-h's own archived checkpoints for the contact-vs-finishing
divergence a 2026-09-18 external review and erratum surfaced.

Change: added dated entries per commit or run, covering decisions,
deviations, verification, and open questions. The file stays local; the
committed protocol is `snowgym/training/reviews/m7b_r1n_i_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_r1n_h_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for R1n-h, the mixture-imitation curriculum with a
single-opponent control.

Change: added dated entries per commit or run, covering decisions,
deviations, verification, and open questions. The file stays local; the
committed protocol is `snowgym/training/reviews/m7b_r1n_h_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_r1n_g_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for R1n-g, the label-error diagnostic characterizing why
blue's offense scores zero hits against scripted red.

Change: added dated entries per commit or run, covering decisions,
deviations, verification, and open questions. The file stays local; the
committed protocol is `snowgym/training/reviews/m7b_r1n_g_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_r1n_f_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for R1n-f, the opponent-transfer evaluation of the R1n-e
policies.

Change: added dated entries per commit or run, covering decisions,
deviations, verification, and open questions. The file stays local; the
committed protocol is `snowgym/training/reviews/m7b_r1n_f_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_r1n_e_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for R1n-e, PPO from the imitation policies with death rate
as the primary test.

Change: added dated entries per commit or run, covering decisions,
deviations, verification, and open questions. The file stays local; the
committed protocol is `snowgym/training/reviews/m7b_r1n_e_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_r1n_d_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for R1n-d, the diagnostics step before PPO.

Change: added dated entries per commit or run, covering decisions,
deviations, verification, and open questions. The file stays local; the
committed protocol is `snowgym/training/reviews/m7b_r1n_d_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_r1n_c_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for R1n-c, D3 step 1.

Change: added dated entries per commit or run, covering decisions,
deviations, verification, and open questions. The file stays local; the
committed protocol is `snowgym/training/reviews/m7b_r1n_c_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_r1n_b_dev_notes.md`

Reason: the user asked for a note on every development step. This is the
running local log for R1n-b.

Change: added dated entries per commit or run, covering decisions, deviations,
verification, and open questions. The file stays local; the committed protocol
is `snowgym/training/reviews/m7b_r1n_b_declaration.md`.

Upstream behavior: unchanged; documentation only.

## `refs/snowgym_continuation_plan_2026-09-13.md`

Reason: user-requested development plan continuing from the whole-repository
handoff and the stopped R1n/E3 experiment.

Change: added a proposed plan. It records source-verified E3 findings with
line references:
- critic steps gated by the actor's KL stop;
- throw exploration sharing the movement log-std;
- explained variance reported as R²;
- bootstrapped warm-start targets;
- the critic not reading `option_state`;
- recoverable 1v1 teacher candidates.

It sequences contract repair in new versioned modules, small gated
diagnostics, a branch decision on reward sparsity, an artifact-retaining
supervised pipeline, and the route back to the R1 gates. It also added a
matching proposed, undeclared R1n-b entry to `snowgym/PLAN.md`.

Upstream behavior: unchanged; documentation only. No source, training,
collection, provider request, commit or push was performed; the plan
authorizes no experiment.
