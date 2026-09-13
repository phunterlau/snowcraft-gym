# R1m-S12: movement representability under dense on-policy supervision

Commit this protocol and tested implementation before collection. This is
Experiment E2 of the reviewer handoff
(`refs/snowgym_fighter_rl_ppo_review_claude_opus_2026-09-12.md`, section 6):
can a residual on the unchanged R1f source represent a controller that
recovers the teacher-MOVE benefit, and does fighter-specific entity context
matter? This is supervised and teacher-labeled throughout;
`autonomousQualificationEligible: false`. It does not authorize training,
change a protocol, or promote a checkpoint.

## Deviations from the reviewer's E2 sketch, stated up front

- **Three arms, not two.** The review specified A (current absolute
  `GeometryProbe`) versus B (a new egocentric-attention module). Before
  implementing, `GeometryProbe(source, relative=True)` was found to already
  exist as a one-flag variant of A that already varies pooled context per
  querying fighter -- it costs nothing to include and it is the only way to
  tell whether a B-over-A win comes from the egocentric frame or from
  attention specifically, which matters because E3 is specified to inherit
  "E2's winning input type." Arm **A-rel** is added: same class as A, same
  parameter count as A, `relative=True` instead of `False`.
- **From-reset evaluation only, not first-hit-snapshot continuation.** The
  review's floor/ceiling figures (13/38, 16/36, 35/38, 40/40) are anchored to
  S3's first-hit snapshots and S2's continuation harness. Building a third
  first-hit-continuation harness for three new deterministic-only (no PPO,
  no recovery-time input) arms was judged not to justify its implementation
  risk relative to what it would add. All training, evaluation, floor, and
  ceiling numbers in this declaration use from-reset Engage episodes on the
  existing `EngageOptionBatchV1` / `teacher_option_plan("engage")` path
  (`movement_train.evaluate`'s own harness), on the historical (200000-200039)
  and replication-development (210000-210039) seeds -- 40 seeds each, the
  same "R1h setting" the review names for its third eval split. This is a
  genuine loss of numeric comparability with S2/S3/S9's first-hit-anchored
  floor/ceiling; every number in this experiment's results is a fresh
  from-reset measurement, not a citation of those archived figures.
- **DAgger data collection uses training seeds 100000-100063 as full
  from-reset episodes**, not as S3 first-hit restorations. The review's text
  said "training frames 100000-100063 (first-hit) plus from-reset states on
  training seeds"; this declaration collects only the from-reset half, for
  the same reason as above (avoiding a second post-hit-restoration harness
  for non-`RecoveryPolicy` deterministic probes).
- **Because historical/replication/training-from-reset seeds are never used
  as DAgger training states**, they are a genuine label-free holdout for the
  falsifier that needs one ("both below 60% of ceiling despite held-out
  heading error < 10 degrees").

## Shared architecture, decoder, and loss

All three arms wrap the same frozen, unchanged R1f source
(`checkpointDigest sha256:10d924ec...`), the same residual decoder
`q = c + a * tanh(raw_source + delta)`, and zero-initialized final move/shot
layers, matching `GeometryProbe`'s existing contract exactly (forward/act
signatures are identical across A, A-rel, and B so one loss and one loop
serve all three). Shots stay corrected at execution time in every arm and
round; only the move head is fit.

- **A:** `GeometryProbe(source, relative=False)`, unmodified, imported as-is.
- **A-rel:** `GeometryProbe(source, relative=True)`, unmodified, imported as-is.
- **B:** `AttentionGeometryProbe` (new). Per entity type (allies, enemies,
  projectiles), pair features add relative position, relative velocity
  (velocity is *not* made relative by the existing `relative=True` path;
  B does that too), distance, a unit bearing vector, and closing speed;
  projectiles additionally get time-to-closest-approach and the resulting
  closest-approach distance (both clamped to a 3-second horizon). Pooling
  for these three types is a single-head, scaled dot-product attention
  query built from the querying fighter's own state, concatenated with the
  existing masked max-pool (so pooled width per type stays 32, matching A's
  mean+max, and the move/shot heads have identical input width and
  parameter count across all three arms). Obstacles keep A-rel's plain
  relative-position mean/max pooling unchanged (the review named attention
  only over "enemies, allies and projectiles"). Total parameter count is
  within 5% of A/A-rel (one shared query projection plus wider per-type
  first encoder layers).
- **Loss (`heading_loss`, new -- not `geometry_probe.geometry_loss`, which
  computes plain endpoint MSE for moves and reserves its cosine term for
  throws):** for each living, classifier-chosen-MOVE, recommendation-available
  unit, let distance be the world-unit gap from the fighter's current
  position to the teacher's recommended destination. Outside the 2.1-unit
  slow radius (`MovementSystem.ts` `ARRIVAL_SLOW_RADIUS = PLAYER.spacing *
  1.5 = 1.4 * 1.5`), use `1 - cosine` between the learned and teacher
  heading vectors (both measured from the fighter's current position).
  Inside the radius, use squared world-unit endpoint error, clipping the
  raw distance (not the square) at 3 units first. `moveAvailable`
  (`recommend_movement`'s validity) gates both terms; a unit the classifier
  did not choose to move never enters the loss.

## DAgger loop

Four rounds, one Adam optimizer per arm (not reset between rounds), 3,000
steps/round, minibatch 256, learning rate 3e-4, gradient clip 0.5, training
seed 95201 (offset per round). Round 0's from-reset trajectory (the 64
training seeds, teacher-forced: frozen classifier's own action-type choice,
movement target replaced by `recommend_movement`'s recommendation, shots
corrected as always) is collected once and shared unmodified across all
three arms, since it depends on none of their weights. Rounds 1-3 are
per-arm: fit on all data aggregated so far, then collect one more from-reset
pass over the same 64 training seeds under that arm's own just-fitted
policy (deterministic act, shots still corrected). The final fit (after
round 3's collection) is each arm's evaluated checkpoint. Labels
(`recommend_movement`'s target and validity) are read-only and are never
used to change which action type or shot is issued; they never enter PPO.

## Evaluation (final round only)

Deterministic from-reset closed-loop success, progress, and rejected-action
rate on training (100000-100063, in-sample), historical-development
(200000-200039), and replication-development (210000-210039) -- 64, 40, and
40 seeds. Controls, also from reset on the two development splits:
**floor** = each arm's zero-initialized (pre-DAgger) checkpoint; **ceiling**
= the shared teacher-forced trajectory (frozen classifier + teacher MOVE +
corrected shots), computed once and shared by all arms as in round 0.
Held-out mean heading-error degrees (the `heading_loss` diagnostic, not the
optimized objective) is reported on the development splits' MOVE-eligible
decisions for each arm's final checkpoint.

## Prediction and falsifiers (reviewer's, extended to three arms)

Prediction: B success on both development splits at or above A-rel, which
is at or above A; a large A -> A-rel gap versus a small A-rel -> B gap would
say the frame mattered more than attention. Falsifiers / decisions:

- A ~= A-rel ~= B: representation is not the bottleneck; keep the current
  absolute path in production; prioritize authority (E3) over representation.
- A-rel ~= B > A: the egocentric frame was the whole story; E3 uses
  `relative=True` geometry and the attention module is not carried forward.
- B > A-rel > A: attention earns its added parameters; E3 uses B's features.
- All three below 60% of the from-reset ceiling despite held-out heading
  error under 10 degrees on development seeds: representation is not the
  limiting factor at all; the ceiling is closed-loop timing or classifier
  coupling, so E3 should prioritize full action authority over any
  representation choice.

## Budget and gates

Ceiling (40) + floor (3 arms x 40) + shared round-0 collection (64) + 3 arms
x 3 collection rounds (64 each) + 3 arms x final evaluation (64 + 40 + 40).
Declared cap: 250,000 simulator decisions, enforced by an `account()` guard.
No qualification seeds, provider calls, browser input, or protocol changes.
Stopping rule: fixed four rounds, final-round evaluation only, no arm or
round selection; two optimizer replications (seeds offset from 95201) only
if the predeclared B-over-A-rel and B-over-A development-success comparisons
both hold on the first seed.

Before implementation and results commits: targeted
`geometry_representation_probe` tests, `npm test`, `npm run build`, Python
client tests, Python training tests.
