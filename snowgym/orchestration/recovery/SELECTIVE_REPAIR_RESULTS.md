# Selective tactical repair: mechanism audit results

2026-09-05. Source revision: `4df9dd880dd1052b6c5139cbb4c5151aebbc6548`.

Binding-only refresh reproduced full reactivation's physical trajectories in
every tested pair, including the original loss-to-win case. Its benefit did not
generalize across rosters: immediate refresh helped some 6v10 cases and severely
harmed 10v10. Keep the production defaults unchanged.

## Protocol and lineage

The [pre-collection declaration](../../../refs/snowgym_selective_repair_review_and_plan.md)
fixed four arms, three rosters, arena6, normal scripted Red, a single main-role
Engage plan, and a deterministic scripted blue executor at 10 Hz. No provider,
learned fighter, training, browser, or HTTP server participated.

Seed preflight inspected 494 JSON metadata files and 19,966 recorded seed
declarations, including ignored local runs. It found no collision with
630000–630119 and no previously registered run of this experiment. One historical
Python metric file contained non-finite values, handled in memory without edits.
The audit covers recorded metadata, not unrecorded runs or standalone compressed
metadata. The configuration binds 222 source files and the historical manifest.

The 300-decision scan found first casualty opportunities in 37/40 5v5, 40/40
10v10, and 40/40 6v10 episodes. Seeds 630015, 630016 and 630035 had no qualifying
opportunity and were not replaced. All cohorts exceeded the declared minimum of
20 qualifying seeds. Three historical fixtures were included separately.

Each fixture received four interventions at delays of 0/1/2/4/8 seconds, with a
shared 300-decision continuation horizon including delay. The 2,400 branches
were each independently executed twice: 4,800 executions, exact action/hash
agreement, and 1,046,237 recorded action results with zero rejections. The
recorded-action count excludes duplicate executions. Historical keep/reactivation
trajectories also matched all 30 original branch artifacts.

The [sealed report](examples/selective-repair-20260905-v0/report.json) contains
all paired analyses, including net damage and survivors. The
[manifest](examples/selective-repair-20260905-v0/manifest.json) digest is
`sha256:0c6321031cab2af1c18c7984ae606228eaf247cbe3bfc5a58d035a47d77c04d9`.
Artifacts retain full prefixes, actions/hashes, plan state, binding revisions,
frozen original target membership/health denominator, range occupancy and local
shot acceptance. Inventory/digest checks and fixture reconstruction passed.
The final commit gate passed 367 TypeScript tests, 51 Python client tests, 257
Python training tests and the production build. Local documentation links also
passed validation. Existing bundle-size and legacy-Gym deprecation warnings remain.

## Immediate intervention

Entries show blue wins / qualifying seeds; parentheses show horizon-censored
episodes. Censoring is recorded separately from red wins.

| Roster | Keep | Binding refresh | Local fire | Full reactivation |
| --- | --- | --- | --- | --- |
| 5v5 | 21/37 (5) | 23/37 (4) | 25/37 (2) | 23/37 (4) |
| 10v10 | 33/40 (2) | 15/40 (5) | 33/40 (2) | 15/40 (5) |
| 6v10 | 0/40 (6) | 5/40 (1) | 0/40 (6) | 5/40 (1) |

Paired win-rate differences relative to keep, in percentage points:

| Roster | Binding refresh: mean [95% interval] | Local fire: mean [95% interval] |
| --- | --- | --- |
| 5v5 | +5.4 [-13.5, +24.3] | +10.8 [-8.1, +29.7] |
| 10v10 | -45.0 [-62.5, -27.5] | 0.0 [0.0, 0.0] |
| 6v10 | +12.5 [+2.5, +22.5] | 0.0 [0.0, 0.0] |

Intervals use 10,000 paired percentile bootstrap resamples, RNG 730001. They are
exploratory, unadjusted for multiple comparisons, and exclude historical cases.
Repeated arms/delays/reruns do not increase the independent environment sample.
Separate comparisons consume successive RNG draws, so identical treatment
effects can have slightly different Monte Carlo interval endpoints in the JSON.

In 5v5, local fire converted nine non-wins into wins but lost five existing wins.
Its net damage-exchange improvement was +85.4 health points [18.4, 154.1]; its
win-rate interval includes zero. Damage here is continuation team-health loss,
with no per-role hit attribution. Local fire issued 312 opportunistic shots at
zero delay, all accepted; acceptance does not establish a hit.

In 10v10, binding refresh gained one win and lost 19. Mean damage exchange
deteriorated by 396.5 health points [313.0, 480.5 deterioration], and blue survivor
count fell by 2.825 [2.075, 3.575 fewer]. In 6v10 it gained five wins, but damage
and survivor improvement intervals included zero. These remain diagnostic
successes rather than qualification results.

The local-fire condition never triggered in any fresh 10v10 or 6v10 branch, at
any delay. Its unchanged outcomes in those cohorts are inactive-intervention
controls, not evidence that active opportunistic firing is universally safe.

## Mechanism and timing

Binding refresh and full reactivation had identical actions and physical hashes
in all 600 fixture-delay pairs: 585 fresh and 15 historical. Binding-only refresh
preserved assignments, symbolic commands, activation anchors/time and plan
version. Thus target rebinding was sufficient to reproduce the historical win
and the observed regressions in this single-main-role Engage setup. This does
not establish equivalence for multiple roles or anchored region missions.

The original seed-620002 5v5 case still loses under keep, wins with two blue
survivors under binding refresh/reactivation, and loses under local fire despite
dealing 360 damage versus keep's 40. Full trajectories preserve this distinction.

Fresh blue win counts at each delay (same seed denominator throughout):

| Roster / arm | 0 s | 1 s | 2 s | 4 s | 8 s |
| --- | --- | --- | --- | --- | --- |
| 5v5 keep | 21 | 21 | 21 | 21 | 21 |
| 5v5 refresh / reactivate | 23 | 20 | 18 | 23 | 18 |
| 5v5 local fire | 25 | 22 | 19 | 20 | 21 |
| 10v10 keep / local fire | 33 | 33 | 33 | 33 | 33 |
| 10v10 refresh / reactivate | 15 | 19 | 22 | 34 | 35 |
| 6v10 keep / local fire | 0 | 0 | 0 | 0 | 0 |
| 6v10 refresh / reactivate | 5 | 5 | 5 | 3 | 2 |

At eight seconds, intervention was unreachable in 4/37, 9/40 and 21/40 fresh
cases respectively; these cases remain in the paired analysis. Later refresh
acts on a later physical state and consumes the shared horizon. The non-monotonic
curves do not identify an optimal provider latency. Red actions can diverge after
intervention; equal seeds do not hold opponent actions fixed thereafter.

## Review conclusions and proposed next gate

The casualty predicate locates a disturbance, but does not establish that the
current binding should be replaced. Automatic refresh at every casualty is
unsupported. Local firing is a narrower action, yet its active 5v5 win-rate
benefit remains uncertain and it can turn individual wins into losses.

A useful next experiment would predeclare a retain-versus-repair decision rule
using information available at intervention: binding occupancy, readiness,
recent damage exchange and remaining force. Use these exposed diagnostic seeds
for hypothesis development only, then audit and declare new validation seeds.
Keep unconditional keep/refresh/local-fire controls and measure treatment
activation, benefits, regressions and request timing separately. The current
data do not train or validate such a selector.

Multi-role and persistent-anchor scenarios are needed before claiming that
selective repair avoids assignment or anchor disruption. A combined refresh plus
local-fire arm, learned repair selector, changed casualty trigger or provider
comparison requires a new declaration; none was added to this run. R1n/M7c
qualification gates and production behavior remain unchanged.

## Reproduce and inspect

```bash
node --import tsx snowgym/orchestration/examples/selective-repair-audit.ts \
  --verify snowgym/orchestration/recovery/examples/selective-repair-20260905-v0

# Full rerun into a new directory; recorded same-experiment seed reuse is explicit.
node --import tsx snowgym/orchestration/examples/selective-repair-audit.ts \
  --output /tmp/snowgym-selective-repair-rerun

# One diagnostic 6v10 win: seed 630087, three blue survivors.
gzip -dc snowgym/orchestration/recovery/examples/selective-repair-20260905-v0/branch-87-refresh_binding-0.json.gz
```

Other immediate-refresh 6v10 winning seeds are 630088, 630093, 630107 and 630115.
These are inspectable headless traces, not replay-UI JSON files. See the
[runner guide](README.md#selective-repair-audit) for artifact semantics.
