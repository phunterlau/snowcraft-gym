# R1m-S4: frozen exploration and advantage audit

Declared before measurement. No optimization, new policy continuation, provider
call or qualification is included. Preserve all S3 artifacts and weights.

## Fixed evidence

Audit S3 training RNGs 94101–94103 at collection updates 1, 11 and 21. Recreate
the initial policy or load the preceding update-10/update-20 checkpoint. Use all
eight archived trajectories at each point: 72 trajectories total, including
repeated snapshot seeds. Restore each original prefix, verify physical/plan/
option/tensor identity, then replay only its recorded learned window. Reproduce
its sampled latents, log probabilities, values and post-action hashes. Check the
snapshot sampler sequence. Use the archived, digest-verified frozen-tail rewards;
do not claim to have replayed those tails in this audit.

On the same 57 training first-hit snapshots, compare each RNG's initial,
update-10, update-20 and final policy. All comparisons use identical states and
remaining recovery fraction one. No development or qualification states are
needed. The simulator budget is at most 25,800 decisions: 72 reset-prefix plus
learned-window reconstructions, and 57 common-state reset-prefix reconstructions,
each bounded by 200. Inference on cached tensors adds no simulator transitions.

## Measurements

For living units choosing movement, report world-space sampled displacement,
saturation, distance from the behavior mean to the valid teacher movement
recommendation, and the recommendation's latent Mahalanobis distance under the
fixed Normal standard deviation. The same-covariance Normal KL for moving its
mean to that recommendation is half the squared Mahalanobis distance. Report
the fraction inside a radius-three latent ball. This is a geometric coverage
measure, not the probability of hitting an exact point or succeeding in battle.

Validate teacher/helper agreement on visited states. A valid recommendation
does not guarantee an accepted or useful action. S2 established an aggregate
benefit on other, development states; it did not validate every recommendation
in this training-state audit. Keep availability/readiness and actual selection
separate. Do not introduce another movement decoder or formula.

On the common snapshots, report policy displacement from initialization,
distance reduction toward the recommendation, and fractions closer/farther.
Aggregate within each snapshot first, then across snapshots, so roster size
does not create extra independent observations.

Reconstruct S3 GAE from the recorded rewards and behavior values. Reproduce the
first-epoch minibatch permutation from the restored Torch RNG and normalize
advantages within each minibatch as the training implementation does. At the
frozen behavior weights, compute the movement-mean score direction proportional
to normalized advantage times `(latent - mean) / variance`, with the actual
per-living-unit and minibatch normalization. Project it through the tanh-to-world
Jacobian toward the recommendation. The first minibatch corresponds to the
initial PPO gradient at ratio one; later minibatches are frozen-weight diagnostics,
not the sequential optimizer's actual gradients. No optimizer steps are taken.

Report the sign and magnitude of these output-space projections, alongside
sample-direction/advantage correlations. This diagnostic omits shared-parameter
coupling and Adam/clipping, and is not a causal test of recommendation quality.
Keep initial/final policy changes distinct from this local score diagnostic.

## Reporting and stop

Use descriptive means, medians and tails; report opportunity, decision, trajectory
and unique-seed counts separately. Repeated episodes/units are not independent
environment seeds. No retrospective promotion threshold or best checkpoint is
selected. Archive per-opportunity data, paired common-state summaries, reconstructed
advantage diagnostics and source/checkpoint/artifact digests. Any proposed change
to exploration, reward, critic or policy requires a separate declaration after
this audit. R1n remains open.

Before implementation and results commits, run targeted reconstruction, scale,
mask, finite-boundary, gradient-direction and immutability tests plus the full
TypeScript/build/client/training gate.
