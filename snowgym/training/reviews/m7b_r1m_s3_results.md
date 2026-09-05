# R1m-S3: short recovery PPO did not pass

Completed 2026-09-05. None of the three final checkpoints passed the declared
assisted learning gate on either development set. Stop at the fixed budget;
preserve every checkpoint and keep R1n autonomous Engage open.

## Implemented contract

The [declaration](m7b_r1m_s3_declaration.md), implementation and tests were
committed in `e8e8ca5` before collection. The recovery actor controls movement
for at most 30 decisions after a frozen baseline first hit, then the original
source resumes until the existing Engage option ends. Categorical weights and
teacher-corrected shots remain fixed. No optimizer acts on the frozen tail.

The new `snowgym.recovery-time.v0` input exposes remaining recovery time to a
zero-initialized actor path and an independent role-aware critic. Existing
option-time/target state remains intact. The new checkpoint format records the
contract, source, snapshot digest, optimizer, Torch RNG, sampler RNG and update
history. Legacy environment tensors and checkpoint loaders are unchanged.

All rewards after handoff are observed and folded, with their original discount,
into the final learned transition before GAE. The simulator does not terminate
at handoff and the original option budget is not reset. See the
[equations](../math/README.md#short-recovery-with-a-frozen-continuation).

## Data and budget

| Partition | Declared seeds | Eligible first-hit snapshots | Use |
|---|---|---:|---|
| Training | 100000–100063 | 57/64 | Uniform snapshot sampling with replacement |
| Historical development | 200000–200039 | 38/40 | Evaluation only; reused S2 baselines |
| Replication-development | 210000–210039 | 36/40 | Evaluation only; reproduced R1m baselines |

No missing trigger was replaced. All partitions were already exposed in prior
work; no source-unseen or untouched qualification claim is made. Snapshot digest
and seed-range checks prevent either development set from entering training.
Zero-residual recovery exactly reproduced all 74 eligible development baseline
suffixes, including action digests and state hashes.

Three RNGs each ran 30 updates, eight trajectories per update, four PPO epochs,
minibatches of 120 and the unchanged coupled KL stop. The 720 training trajectories
contained 21,314 learner-controlled decisions and 33,408 frozen-tail decisions.
Including 75,143 prefix-reconstruction decisions, training used 129,865 of its
144,000 maximum simulator decisions. It stayed below the 21,600 learned-decision
limit. Snapshot construction used 104 newly executed baseline episodes; 40
historical baseline artifacts were reused. Evaluation comprised 74 initializer
and 222 final-checkpoint continuations, separate from training.

## Final results

The matched initializer succeeds on 13/38 historical and 16/36 replication-
development episodes. Differences are percentage points; intervals use 10,000
episode-paired bootstrap resamples with RNG 970001.

| Training RNG | Historical success | Difference [95% CI] | Replication-development success | Difference [95% CI] |
|---|---:|---:|---:|---:|
| 94101 | 14/38 | +2.6 [−5.3, 10.5] | 19/36 | +8.3 [−2.8, 19.4] |
| 94102 | 14/38 | +2.6 [−5.3, 10.5] | 17/36 | +2.8 [−5.6, 11.1] |
| 94103 | 11/38 | −5.3 [−21.1, 10.5] | 17/36 | +2.8 [−8.3, 13.9] |

Every success interval includes zero, and no point estimate reaches the required
+20-point gain. All actor parameter-change and rejected-action checks passed.
Final progress differences also have intervals spanning zero on both sets for
all RNGs. Exact results remain in the [paired report](../runs/m7b_engage_r1m_s3_v0/report.json).
The three optimizer runs use the same development environments; they are not
three independent sets of environments.

| RNG | Actor parameter L2 change | Optimizer steps | Unique training snapshots used |
|---|---:|---:|---:|
| 94101 | 0.59116 | 159 | 54 |
| 94102 | 0.51405 | 159 | 57 |
| 94103 | 0.42487 | 114 | 57 |

There were zero rejected actions among 273,610 training and 89,215 final-evaluation
action results. These counts exclude snapshot construction, prefix replay and
initializer evaluation. Every checkpoint remains explicitly teacher-assisted
and `autonomousQualificationEligible: false`.

## Interpretation and remaining issue

S2 established that a prescribed brief movement correction can improve the
remaining trajectory. S3 did not recover that effect with this fixed PPO setup.
The learned parameters changed, but mission success and progress stayed close
to the matched initializer. Shortening the controlled window and preserving
the actual tail return were insufficient at the declared budget.

This does not establish that brief recovery is unlearnable. It leaves several
mechanisms unresolved: whether fixed latent noise explores physically useful
destinations, whether KL-constrained updates permit sufficient movement changes,
and whether estimated advantages favor the locally useful corrections. Correct
discount accounting does not by itself solve variance or credit assignment.

The next justified proposal is a read-only audit of world-space exploration,
physical policy change and advantage alignment using these frozen checkpoints
and reproducible training snapshots. Compare sampled displacement and update
direction with the corrections that were useful in S2, without selecting a new
checkpoint or widening the PPO budget. Declare any resulting intervention before
training. No new noise schedule, reward redesign, aiming fit, provider call or
commander experiment was started here.

The remaining research objective is unchanged: qualify one autonomous Engage
executor with all runtime teacher overrides disabled, then resume the other
M7b missions and M7c composition. Neither S2's assisted teacher effect nor S3's
parameter movement closes that gate.

## Verification and artifacts

The [archive](../runs/m7b_engage_r1m_s3_v0) contains 104 compressed baseline
episodes, digest-bound snapshot sets, 90 training event files, three final
evaluations and checkpoints at updates 10/20/30 for each RNG. Intermediate
checkpoints were saved for recovery only and were not evaluated or selected.

All 222 manifest-listed files and the manifest digest verified with exact
inventory coverage. All nine checkpoint loaders passed. Every recorded training
episode references an allowed training snapshot. Stored actor likelihoods cover
only the learned window. Across all 720 trajectories, direct discounted rewards
and folded rewards agreed within 1.09e-7. The maximum stored/reevaluated latent
log-probability difference was 1.06e-4, within the declared implementation check
of 1e-3 for differing CPU batch shapes. Source weights and prior archives were
unchanged.

Tests verified tail-return equality, time-input validation, zero-residual
handoff parity, actor/critic gradient separation, unused shot-head gradients,
finite boundary behavior, snapshot partition and tamper rejection, and exact
update-boundary resume. Full gates passed before implementation commit and
again before result commit: 367 TypeScript tests, 51 client tests, 268 training
tests and the build. Existing bundle-size and legacy-Gym warnings remain.
All work was headless and provider-independent.
