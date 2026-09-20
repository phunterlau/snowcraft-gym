# M8-S4 — 3v3 Engage: metric definitions, teacher achievability, and random-init floors

Declared 2026-09-20, before any research collection. This is a collect-and-archive
step (declare → implement → collect → archive), unlike the S1–S3 infrastructure
steps. It trains nothing and makes no R1 qualification claim.
`autonomousQualificationEligible` stays false.

## 0. Why, and what S3 left open

[S3](m8_s3_declaration.md) showed the existing actor, critic, Engage tracker and
collector run at 3v3 unchanged, and that a random-init blue team loses units
mid-option while the option continues. R1n-e's headline metric, **death rate**,
was defined by "the only blue unit died" and has no direct 3v3 reading. Three
different 3v3 quantities (any-unit death, mean units lost, team elimination) would
be three different experiments. **They are fixed here, before the seed band, the
budget, or any data**, so none can be chosen after seeing results.

## 1. Metric definitions (frozen, before data)

Per episode, at the moment the Engage option ends, with `A` = the blue units
assigned to the option (all three at 3v3) and roster `n = |A|`:

- **Success** — the frozen Engage criterion, unchanged: the activated targets'
  summed health is at most 20% of its activation value (`tracker._success`),
  reached before failure and within the 200-decision horizon.
- **Units lost fraction `L`** = (`n` − number of units in `A` alive at option end) / `n`,
  in {0, 1/3, 2/3, 1}. At roster 1 this equals R1n's death indicator, so `L` is the
  strict generalization of death rate rather than a new metric.
- **Team wipe** — `L = 1` (every assigned unit dead). This is the tracker's own
  `living == 0` failure.
- **Timeout** — the 200-decision horizon reached without success.
- **Rejected-action rate** — rejected / total submitted actions.

**Primary loss metric for later learning comparisons (declared now, not to be
changed after S4's data): mean `L`, paired by world.** Success is the co-primary
outcome; wipe rate and timeout rate are co-reported. Killing the last red unit
while losing blue units counts as success with `L > 0`; the two are reported
separately and never merged into one score.

`L` is measured on the option's end state, so it inherits R1n's structure: an
episode that succeeds quickly has fewer opportunities to lose units. It is
therefore always reported next to success and mean episode length (decisions), not
alone.

## 2. Design

Worlds: 400 evaluation worlds, seeds **2100000–2100399**, at 3v3 in the R1n arena
and horizon (`full_authority_train_v1.scenario()` with `blueUnits = redUnits = 3`).
The band was audited against every tracked file in the repository, including all
JSON, and is unused; the preflight test additionally covers the new declaration.
Every condition below uses the **same 400 worlds** (world-paired).

Red arms, each native (Red runs behind the team action, per S3 §2):
`random`, `scripted-easy`, `scripted-normal` (`mixture_imitation.EVAL_ARMS`).

Conditions:

1. **Teacher** — the native plan-conditioned blue controller
   (`run_block(choose=None)`), 400 worlds × 3 arms.
2. **Random-initialized policy floor** — `FullAuthorityPolicyV1(destination="global")`
   at three initialization seeds (98101, 98102, 98103), each in two modes:
   deterministic (argmax and mean actions) and stochastic (sampled, the behavior a
   from-scratch PPO would collect), 100 worlds × 3 arms each. These are the first
   100 seeds of the band (2100000–2100099).

Initialization and sampling are fixed and identical in procedure across cells: each
floor model is built after `torch.manual_seed(init_seed)`, so every arm and mode of
one init seed uses the same weights. Stochastic cells reseed
`torch.manual_seed(init_seed * 10 + arm_index)` (arm order random, easy, normal)
before collecting, so the sampling stream differs across arms but is reproducible.
Deterministic cells use no sampling.

No checkpoint is trained, loaded, or selected.

Budget bound: teacher ≤ 400 × 3 × 200 = 240,000 decisions; floors ≤ 3 × 2 × 100 × 3
× 200 = 360,000; total bound 600,000, hard cap 700,000 (the run aborts if exceeded).
A run past the bound but under the cap is reported as such. The peek below suggests
usage far below this.

## 3. Pre-declared gates and their meaning

| Question | Rule (fixed now) | If it fails |
| --- | --- | --- |
| Teacher achievable at 3v3 | Teacher success ≥ 0.90 in each of the three Red arms | The plain teacher is not a usable imitation target at 3v3; repair or change it before any learner |
| Floors show no free signal | Random-init deterministic success ≤ 0.10 against both scripted arms, for all three init seeds | A from-scratch 3v3 PPO is not obviously hopeless; reconsider whether an imitation initializer is required |
| Timeout/wipe structure | Reported, no threshold | Descriptive |

**Failure policy (decided before the data):** a failed gate does not stop the run. S4
always completes, seals and reports; a failed gate is recorded as a finding, and what
to do about it (repair the teacher, or reconsider from-scratch PPO) is decided in a
new declaration, not inside S4. A run that exceeds its declared bound but stays under
the hard cap is sealed and reported as such; a run that hits the cap aborts unsealed.

Intervals: Wilson 95% for success and wipe proportions; percentile bootstrap over
worlds (10,000 resamples, seed 960004) for mean `L`. The three init seeds are three
draws of one initialization procedure, not independent training runs; they are
reported per seed and never pooled into a single interval.

## 4. Feasibility peek (disclosed, not independent evidence)

Before declaring, a 20-world peek on throwaway seeds 9001–9020 (outside every
research band, results discarded) checked that the run is cheap and non-degenerate.
It showed teacher 20/20 at 3v3 on all three arms and random-init 0/20 success, with
scripted Red wiping blue in nearly every random-init episode. **The gates were set
with knowledge of that direction**: they are not independent of the peek, and were
not tuned to its exact numbers (0.90 sits just under R1n-c's 93/91 teacher
success rates; 0.10 is a round near-zero ceiling chosen here, not an inherited convention). They test whether the pattern holds at 400
worlds and against frozen thresholds, not whether it exists.

## 5. What this does and does not decide

It provides the 3v3 teacher ceiling, the random-init floor, and frozen metric
definitions, all archived. It does not show that 3v3 Engage is learnable, does not
choose an initializer, and says nothing about command-following or roles. The next
step (S5) will follow R1n's recipe: teacher imitation as the initializer, then a
predeclared comparison. Before S5's critic warm-start gate is reused, it must be
re-checked at 3v3: `assignedLivingFraction` in `option_state` is now graded
(1, 2/3, 1/3, 0) instead of binary, and the R1n-d return-predictability ceiling
(R² about 0.16–0.24) was measured at roster 1.

## 6. Verification and archive

New module `options/roster_baseline.py` and tests; no existing module is edited.
Complete gates before the collection: `npm test` (366/367, the one documented
R1n-b exception), `npm run build`, client and training `pytest`. Archive sealed with
`death_rate_ppo.seal`, manifest verified with `verify_sealed`, results in
`reviews/m8_s4_results.md`. The run refuses to overwrite an existing directory.
Wrong if: a gate fails (reported as a finding, not patched around), the teacher's
end-state `L` cannot be read from the sealed episode rows, or an existing module
needs editing.
