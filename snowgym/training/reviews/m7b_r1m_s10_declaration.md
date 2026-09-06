# R1m-S10: frozen late-state learning audit

Inspect the negative S9 pair, RNG 99301, without fitting or changing any policy,
reward, seed allocation, or control contract. Commit this protocol and tested
implementation before measurement. This is an assisted training-state diagnostic;
there is no promotion gate or new evaluation result.

Audit selection coverage in all 30 archived updates of both arms. Early means
offset 0–29 after first hit; late means offset >=30. Report collected decision
counts, selected occurrences and unique selected decisions, with per-update and
pooled denominators. Repeated occurrences are not independent states. Short has
no learned late rows; its frozen tail is already folded into its last reward.

Reconstruct updates 1, 11 and 21, eight episodes each, in both arms (48 episodes,
at most 9,600 simulator decisions including prefixes). Restore pre-update weights
and frame/optimizer RNG from initialization or checkpoints 10/20. Restore the
isolated episode sampling seeds. Require exact complete archived trajectory
equality, including actions, state hashes, option records, rewards, values,
latents, densities and GAE. Check source and checkpoint digests before/after.
No alternate continuation is allowed. Test trajectories have separate budgets.

For every living selected MOVE, record read-only teacher recommendation geometry,
availability, controller readiness and action legality separately. Validate the
existing teacher helper in its supported open single-main Engage scenario.
Report sample displacement, recommendation gap, distance, saturation and the
fraction within three latent standard deviations. Recommendations are not
assumed executable or combat-optimal.

Reconstruct selected-row multiplicities and the exact first optimizer minibatch.
Require its loss to match the archived loss (rtol 1e-5, atol 1e-6). At frozen
behavior weights, compute independent mean-output ascent scores for the first
epoch, using each minibatch's advantage normalization and living-unit divisor.
Preserve duplicate occurrences and sum their contributions. These ratio-one
scores are not sequential Adam updates; label only the first minibatch as the
actual pre-update batch. Test the analytic score against autograd, including
duplicates and unused heads. Report normalized-advantage/sample-direction
correlation, projection toward the recommendation and between-episode advantage
variance. No confidence intervals based on independent fighter opportunities.

On the same reconstructed states compare final-checkpoint deterministic MOVE
means with behavior means. Report cumulative world-space target shift and
recommendation-gap reduction by arm, update, early/late and outside range (>9).
This is neither a one-update change nor physical displacement or a causal
performance estimate. Also report critic explained variance by time stratum.

Archive versioned opportunity, selection, reconstruction and report files in a
new immutable directory, bound to S9 inventory, implementation and declaration
digests. Abort on lineage, replay, nonfinite-data or loss mismatches. Run targeted
checks and all four milestone gates before implementation and results commits.
No provider calls, browser, new training, automatic sweep or gate relaxation.
