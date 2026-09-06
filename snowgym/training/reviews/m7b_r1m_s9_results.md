# R1m-S9: longer learnable control did not pass recovery gates

Completed 2026-09-05. The [declaration](m7b_r1m_s9_declaration.md) and tested
implementation were committed as `1549f8c` before fitting. Both arms completed
30 updates under training RNG 99301. The predeclared development gates failed;
conditional optimizer-seed replications 99302/99303 were not run. No thresholds,
noise, reward, decoder, classifier, or provider settings were changed afterward.

## Matched comparison

Both arms use the same R1h/R1m source, RecoveryPolicy architecture and initial
parameter digest, corrected-shot assistance, and identical sampled first-hit
frame schedules. There are 57 training, 38 historical-development, and 36
replication-development frames from the existing S3 archive. These development
sets are previously exposed; the name replication-development does not imply a
new optimizer-seed replication or untouched qualification set.

Short learns movement for 30 decisions and then uses the source continuation.
Full learns movement through termination at the original option horizon. The
control-time fraction reflects each arm's window; option budget is independently
visible. No teacher MOVE destinations enter either learned segment. Both zero
residuals reproduce every archived frame's source action/hash sequence: 262
complete parity episodes, using 47,016 prefix-plus-continuation decisions.

Each arm collects eight paired training frames/update and presents 240 decision
rows to the same four-epoch, minibatch-120 PPO update, subject to existing KL
stopping. GAE is calculated before subsampling. Short includes all its rows plus
deficit resampling when necessary; full generally subsamples its larger collection.
This is a matched new experiment, not a rerun of S3's original sampling procedure.

## Final-only evaluation

All figures refer to Engage mission completion, not full battle wins. Both final
checkpoints were reloaded before evaluation. Differences favor full over short;
intervals are 10,000 paired environment-seed bootstrap resamples, RNG 993001.

| Development split | Initialization | Short control | Full control | Full minus short success [95% interval] | Full minus short return [95% interval] |
| --- | ---: | ---: | ---: | --- | --- |
| Historical, 38 frames | 13/38 | 16/38 | 13/38 | -7.9 points [-23.7, +7.9] | -0.1210 [-0.3764, +0.1347] |
| Replication-development, 36 frames | 16/36 | 17/36 | 17/36 | 0.0 points [-11.1, +11.1] | +0.0097 [-0.1731, +0.1917] |

Full improves over initialization by 0.0 success points [-10.5, +10.5] on the
historical set and +2.8 [0.0, +8.3] on replication-development. Its success is
34.2% and 47.2%, below the 50% gate, and it fails the declared initializer and
short-control improvement requirements. Parameter-change and rejection checks
pass. Evaluation rejected zero of 59,255 action results across both arms and
sets; prefix actions and preflight are excluded from that count.

There is no reliable full-control improvement under this bounded configuration.
The intervals also do not establish that full control is generally worse.
Optimizer-seed variability remains unmeasured because the replication trigger
failed. Do not select an intermediate checkpoint or substitute training success
for the final evaluation.

## Optimization and collection accounting

Counts below cover 240 stochastic training episodes per arm. Unique selected rows
are unique within each update, summed over updates; they are not globally unique
physical states. Repeated prefixes occur because the frame schedule samples with
replacement.

| Quantity | Short | Full |
| --- | ---: | ---: |
| Collected learnable decisions | 7,154 | 18,119 |
| Presented optimizer rows | 7,200 | 7,200 |
| Unique selected rows within updates | 7,154 | 7,200 |
| Selected living MOVE opportunities | 27,256 | 26,344 |
| Actual optimizer steps / maximum 240 | 103 | 125 |
| Updates stopped by KL / 30 | 26 | 23 |
| Frozen-tail decisions | 11,635 | 0 |
| Prefix decisions | 24,915 | 24,915 |
| Total training simulator decisions | 43,704 | 43,034 |
| Actor parameter L2 change | 0.4066 | 0.4547 |
| Mean pre-update critic explained variance | 0.0184 | 0.0542 |
| Mean common-state world-target change per update | 0.1243 | 0.1123 |

The world-target statistic compares before/after policy means on that update's
collected states. It is not distance traveled or cumulative final displacement
from initialization. Critic explained variance is averaged over per-update
on-policy return fits, not an independent value-function evaluation.

All executed actor updates exceed the 0.5 pre-clipping norm threshold in both
arms; median raw norms are 4.64 and 4.48. This and frequent KL stopping show that
the safeguards were active, but do not establish them as the cause of failure.
Full receives more actual optimizer steps than short, so its result cannot be
explained by fewer gradient updates alone. The experiment matches row caps and
episode schedules, not realized optimizer steps or unique collection support.

The largest stored-versus-reevaluated log-density errors are 0.0000839 and
0.0002108, below the existing 0.001 tolerance. Different inference batch sizes
can produce small float differences at the fixed latent standard deviation.
Unused-action densities remain zero, and source/shot parameters remain frozen.

Stochastic training successes are 86/240 for short and 100/240 for full. These
are sampled training states under noisy execution; final development evaluation
uses deterministic means. They cannot demonstrate generalization or establish
overfitting without separating state-distribution and execution-mode differences.

## Interpretation relative to S8

S8 established a useful teacher continuation from teacher-corrected handoff
states. S9 tested whether extending learned movement control from original
first-hit states lets the existing PPO configuration acquire useful behavior.
Both findings can hold:

- A high-return sustained controller exists for many archived states.
- This bounded on-policy learner does not reliably discover a comparable controller.

S9 removes the mandatory early handoff in its full arm, but that change alone
does not recover the teacher benefit. Exploration, advantage quality, policy
representation and the state distribution remain unresolved. The low measured
critic explained variance and small per-update physical-output changes motivate
inspection; they are not a demonstrated single cause.

For the current movement Normal, the local score is

$$
\nabla_\mu\log\pi(z\mid s)=\frac{z-\mu}{\sigma^2}.
$$

Useful updates require sampled deviations to align with useful advantages.
Allowing more late decisions to be learned creates those opportunities but
does not guarantee that useful pursuit sequences enter the sampled distribution
or receive reliable credit. Full collected over twice as many learnable rows,
while only 7,200 entered the capped optimization sets. Inspect the selected rows
as well as the entire collection before interpreting exploration coverage.

Keep the verified latent-density contract and terminal reward. Neither S9's
negative result nor S8's teacher success justifies an automatic larger-noise,
looser-KL, new-decoder or reward sweep. Runtime assistance still prevents
autonomous qualification, irrespective of this diagnostic's outcome.

## Recommended next inspection, not executed

Use the frozen S9 archive to inspect early versus late selected MOVE opportunities:
coverage, advantage/return distributions, critic residuals and signed output-score
alignment with independently valid movement recommendations. Check whether
late pursuit states are collected but rarely selected, selected with weak credit,
or updated without useful physical direction. Compare parameter gradients and
actual mean changes; aggregate loss reductions alone are insufficient.

If noise-versus-deterministic execution appears material, predeclare a separate
matched evaluation rather than replacing the existing deterministic gate. If
representation or local exploration is implicated, select one bounded mechanism
test with its own controls. Preserve both remaining S8 failure cases and all S9
negative outcomes. No new inspection, training or promotion is launched by this note.

## Reproducibility and archive

The [manifest](../runs/m7b_engage_r1m_s9_v0/manifest.json) binds 491 artifacts,
approximately 40.5 MB including the manifest. It includes all preflight, training
and final-evaluation trajectories, three checkpoints per arm, histories, selected
row indices, source/data/declaration digests and final comparisons. Dataset and
ancestor manifests remain unchanged. The complete run uses 160,301 simulator
decisions: 47,016 preflight, 86,738 training and 26,547 final evaluation, below
the 430,000 overall cap. Replication runs were not spent after the failed trigger.

The implementation gate passed 367 TypeScript, 51 client and 298 training tests
plus build. Targeted tests additionally recompute the archived gates, sample
budgets, action hashes, reward sums and simulator ledger. Both arms' tiny tests
reproduce uninterrupted checkpoints exactly after update-boundary resume. This
does not claim recovery from arbitrary mid-episode interruptions.

The final results gate passed 367 TypeScript, 51 client and 299 training tests
(717 total) plus build. Existing bundle-size and legacy Gym-version warnings
remain. Environment observation contracts did not change.
