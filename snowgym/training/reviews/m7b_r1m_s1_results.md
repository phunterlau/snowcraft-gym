# R1m-S1: critic schedule isolation

Completed 2026-09-05. The predeclared consistency gate failed. Preserve the
coupled default; stop this schedule branch at its fixed budget. R1m remains
unreplicated and R1n autonomous Engage remains open.

## Protocol and provenance

The [declaration](m7b_r1m_s1_declaration.md) was committed in `e68c487` before
collection. Six runs used two schedules, three training RNGs and 20 updates per
run. Each update collected 8 worlds × 200 decisions. Source policy, corrected
shots, action classifier, actor KL limit, fixed latent standard deviation,
learning rates and rewards were unchanged. The candidate alone finished critic
updates after the actor stopped. All artifacts explicitly mark teacher-shot
assistance and `autonomousQualificationEligible: false`.

The [archive](../runs/m7b_engage_r1m_s1_v0) contains source/configuration digests,
three common-rollout probes, 120 compressed event files, six final checkpoints,
480 evaluation episodes and a digest-bound manifest. Training used the existing
100000–107999 allocation; every run ended with seed cursor 100320. Common-rollout
probes add 4,800 world decisions to the 192,000 training decisions. Development
blocks 200000–200039 and 210000–210039 were already exposed. No qualification
seeds, provider calls, new teacher-label dataset or browser input were used.

## Isolated first-update result

All three controls exactly reproduced their archived first update. Within each
pair, actor parameters, actor Adam state and the post-update Torch RNG were
identical. The control performed one update; the candidate critic performed 16.

| Training RNG | Initial value MSE | Coupled final MSE | Independent final MSE |
|---|---:|---:|---:|
| 94001 | 0.245137 | 0.242105 | 0.218303 |
| 94002 | 0.288828 | 0.283324 | 0.248872 |
| 94003 | 0.285885 | 0.281695 | 0.247497 |

These errors use the same full rollout and fixed return targets for each pair.
They demonstrate additional in-sample value fitting, not held-out critic accuracy
or a policy benefit. Later advantages and trajectories can differ as the critic
changes; equal simulator seeds do not guarantee equal subsequent actions.

## Final-checkpoint mission success

Each entry uses 40 paired environments. Differences and paired-bootstrap 95%
intervals are in percentage points, with 10,000 resamples and RNG 950001.

| Training RNG | Historical: coupled → independent | Difference [95% CI] | Replication-development: coupled → independent | Difference [95% CI] |
|---|---:|---:|---:|---:|
| 94001 | 21/40 → 17/40 | −10 [−25.06, 7.5] | 25/40 → 23/40 | −5 [−20, 10] |
| 94002 | 14/40 → 17/40 | +7.5 [−7.5, 22.5] | 17/40 → 16/40 | −2.5 [−22.5, 17.5] |
| 94003 | 11/40 → 19/40 | +20 [2.5, 37.5] | 13/40 → 21/40 | +20 [5, 35] |

The replication-development mean difference across training RNGs is +4.17
points, below the declared +10-point threshold. Two point estimates are
negative. Only RNG 94003 has a positive interval lower bound. Three fits on the
same 40 environments are not 120 independent evaluation environments.

Replication-development progress differences were −0.014 [−0.076, 0.055],
+0.030 [−0.065, 0.119] and +0.117 [0.065, 0.170]. The complete physical-win and
progress metrics remain in [report.json](../runs/m7b_engage_r1m_s1_v0/report.json).
Engage success uses the frozen target-health criterion; it is not synonymous
with eliminating the entire Red team.

| Training RNG | Control actor/critic steps | Candidate actor steps | Candidate critic steps | Actor parameter L2 change: control / candidate |
|---|---:|---:|---:|---:|
| 94001 | 190 | 168 | 320 | 0.51473 / 0.46885 |
| 94002 | 193 | 174 | 320 | 0.55834 / 0.50463 |
| 94003 | 172 | 156 | 320 | 0.43284 / 0.45780 |

The candidate uses more critic computation. Actor update counts can change
after the first rollout because critic-dependent advantages affect subsequent
policy updates and KL stops. Frozen inherited source parameters remain unchanged.
All 960,000 training and 427,435 evaluation action results were accepted.

## Interpretation and next boundary

Actor KL stopping does truncate critic fitting, and separating the schedules
can help one training RNG. This experiment does not establish a robust benefit
from completing that budget. The observed additional value fitting is insufficient
evidence for changing the training default or selecting RNG 94003 for promotion.
The short diagnostic also does not establish that independent scheduling would
fail at every longer budget.

Together with R1l's improved conditional regression without autonomous success,
the result argues for measuring sustained control consequences before another
optimizer sweep. A useful next proposal is a bounded diagnostic of action timing,
range retention and post-hit continuation under frozen policies. It needs its
own interventions, scoring windows and declaration; no such run was started here.
Aiming/composition and autonomous qualification remain separately gated.

## Verification

All 149 manifest-listed files and the manifest's own digest verified, with exact
inventory coverage. All six checkpoint loaders passed their digest checks.
All three 20-update control traces matched the first 20 archived R1m updates;
all 60 corresponding decompressed event files were byte-identical as well.
Tests cover no-stop parity, actor/Adam/RNG preservation after stopping, frozen
source/shot parameters, selective reset, interrupted/resumed collection and
checkpoint tamper rejection. No observation contract changed.

Before implementation commit and again before result commit: 367 TypeScript
tests, 51 Python client tests and 261 Python training tests passed; production
build passed. Existing bundle-size and legacy-Gym deprecation warnings remain.
