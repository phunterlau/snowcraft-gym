# R1n: all six configurations stopped at the critic warm-start gate

Completed 2026-09-13. The [declaration](m7b_r1n_declaration.md) (committed
`4b827e6`) and tested implementation were committed before collection. Used
142,880 of the 12,500,000-decision budget: the uniform-random floor
(100 seeds) plus six critic warm-start attempts (20,480 decisions each, one
per arm/seed). No actor training ran, because every one of the six
arm/seed combinations failed the predeclared critic gate
(held-out return R^2 >= 0.25) on its first and only attempt. Per the
declaration's stopping rule, none was retried with a different warm-start
budget.

## Headline

**The experiment did not reach the question it was designed to answer.**
E3 asked whether PPO can learn full-authority fighter micro from scratch
when the task is well posed. It never got that far: the critic gate --
built specifically to avoid spending the 2,000,000-decision actor budget on
top of an unvalidated value function -- fired for all six configurations,
with held-out R^2 far below the 0.25 threshold (range -510.7 to -61.1, not
a marginal miss). A follow-up diagnostic (below) shows this is not simply
"the critic learned nothing": it is a specific, fixable flaw in how this
warm start measures held-out fit, compounded by genuine task difficulty
recorded in the declaration. Both are reported; neither is corrected here.

## Results

| Arm | Seed | Held-out R^2 | Train rows | Held-out rows |
| --- | ---: | ---: | ---: | ---: |
| local | 96001 | -179.95 | 16,384 | 4,096 |
| local | 96002 | -510.67 | 16,384 | 4,096 |
| local | 96003 | -68.32 | 16,384 | 4,096 |
| global | 96001 | -61.14 | 16,384 | 4,096 |
| global | 96002 | -168.99 | 16,384 | 4,096 |
| global | 96003 | -106.03 | 16,384 | 4,096 |

Uniform-random floor: 0/100 successes on seeds 610000-610099 (unsurprising
given the reward-sparsity finding below: a policy that cannot reliably land
a hit rarely reaches the 20%-health mission threshold within 200 decisions).

## Diagnosis: why R^2 is catastrophically negative, not just below threshold

A standalone re-run of one configuration (local, seed 96001; same seeds and
model initialization, so directly comparable to its archived run) inspected
the intermediate return and prediction distributions that the archived run
does not persist:

| Quantity | Mean | SD | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Training-fold GAE returns (256 decisions/world) | -0.317 | 0.247 | -1.000 | -0.067 |
| Held-out-fold GAE returns (64 decisions/world) | -0.077 | **0.0038** | -0.090 | -0.070 |
| Held-out critic prediction, before warm start | -0.081 | 0.005 | -0.092 | -0.061 |
| Held-out critic prediction, after warm start | -0.151 | 0.053 | -0.404 | -0.061 |

The held-out fold's target has almost no variance (SD 0.0038): at 64
decisions/world against a 200-decision episode horizon, almost no episode
reaches a real terminal outcome inside the window, so its GAE return is
dominated by bootstrapping off the *fresh, untrained* critic's own
near-constant value at the window's end, not by anything the environment
did. The training fold's longer 256-decision window is long enough that
some episodes do time out (reaching the real, informative -1.0), giving it
20-30x more spread. After ten epochs fitting the training fold, the critic
produces genuine cross-state variance (SD 0.053) that reflects what it
learned there -- but the held-out target has essentially no matching
variance to explain, so `R^2 = 1 - Var(pred-target)/Var(target)` divides a
merely-modest numerator by a near-zero denominator and produces a number in
the hundreds of negative units. This reproduces the archived run's order of
magnitude (this standalone check does not claim bit-identical
reproduction of the exact R^2 figure).

**This is a flaw in the warm-start gate's own measurement design, and window
length is not quite the mechanism -- it only controls how much the flaw
bites.** `Collector.rollout()` marks the artificial window edge as
truncated (`cut = truncated | (final & ~terminated)`) and GAE correctly
bootstraps from `next_value` there, exactly as it should for a genuinely
cut-off rollout. The problem is what `next_value` *is* during a cold warm
start: the untrained critic's own near-constant output. Both folds' targets
are therefore partly self-referential -- the critic is partly being fit
toward its own predictions -- and the short held-out fold is *almost
entirely* so, since almost no episode reaches a real terminal inside 64
decisions against a 200-decision horizon. A longer held-out window dilutes
this with more real terminals but does not remove the circularity. This was
not tried: changing the warm-start protocol after seeing this result,
without a fresh declaration, would be exactly the kind of after-the-fact
fix the reviewer handoff and every R1m note warn against. The clean fix,
not a workaround, is to measure warm-start R^2 against Monte Carlo returns
on completed episodes only -- no bootstrap, no circularity -- which is
exactly what this review's own R1m-S11/S9 reanalysis already validated as
informative (section C of
`refs/snowgym_fighter_rl_ppo_review_claude_opus_2026-09-12_reanalysis.py`:
a held-out R^2 of 0.18-0.38 against Monte Carlo returns, versus a noisier
GAE lambda-return target on the same states).

Separately, both folds' informativeness is also limited by a property of
the frozen reward recorded in the declaration: `shaping`/`combat` are
health-based, not distance-based, so a policy that has not yet landed a hit
gets no gradient at all toward closing distance. The training fold's own
returns are dominated by the sparse terminal penalty (mean -0.317, min
exactly -1.0); this run did not persist per-episode outcome counts, so how
many of its collected episodes actually reached contact is not measured
here, only consistent with the reward-sparsity finding.

## Interpretation against the predeclared decision rule

None of the review's L-vs-G falsifiers are reachable, because no arm
produced a trained actor to evaluate. The applicable one is the review's
own: **"Both < 30% with a healthy critic: PPO from scratch is inadequate at
this scale"** -- except this result is upstream of even that: the critic
never became healthy enough (by this gate's measurement) to test the actor
at all. That falsifier's proposed next step applies with the same force:
*reward shaping toward approach/contact, or a BC/DAgger-first curriculum,
not a noise or KL sweep* -- and, specific to what this run adds, a corrected
warm-start held-out design (a window comparable to the episode horizon, not
a fraction of it) before spending the actor budget again.

## What this does and does not decide

- **Does not** show full-authority PPO is impossible on this task; the
  measurement that stopped training is itself the thing shown to be
  ill-conditioned.
- **Does** show that six independent attempts (two arms x three seeds), at
  a budget an order of magnitude larger than R1m-S9's, could not clear a
  predeclared sanity gate (R^2 >= 0.25, a reasonable threshold for the
  well-posed regression it was intended to measure) on the very first try,
  and that the shortfall is large enough (two orders of magnitude below
  threshold) that "collect a little more and retry" is not a credible fix
  on its own -- the held-out target this run actually constructed, not the
  threshold, is what needs to change.
- **Does** reinforce, from a different angle than R1m-S9/S11, that this
  reward is sparse before first contact: any protocol here (assisted or
  autonomous) has to get a fighter into contact range before dense signal
  exists at all.

Per the declaration's stopping rule, no arm/seed was retried with a
different warm-start budget, and no checkpoint was promoted. Recommended
next step, not executed here: a small, separately declared diagnostic that
re-derives the warm-start held-out window (full-episode length, not a
fraction of it) and re-measures R^2 before deciding whether to spend the
actor budget on this task.

## Verification

The critic gate fired exactly as designed: `train_run` returns `model=None`
for every stopped arm/seed, and `report.json`'s `runs` entries record
`stoppedAtCriticGate: true` with no `history` key, matching the tested
contract (`test_train_run_stops_at_the_critic_gate_and_records_it_without_actor_training`).
No `final-state.pt` was written for any of the six. `autonomousQualificationEligible:
false` throughout (unreached, since no actor ever trained). No provider
calls, browser input, or protocol changes.

The implementation gate passed 367 TypeScript, 51 Python client, and 372
Python training tests (including 12 targeted `full_authority_ppo`/
`full_authority_train` tests, one a live end-to-end training run) plus
build, before collection.
