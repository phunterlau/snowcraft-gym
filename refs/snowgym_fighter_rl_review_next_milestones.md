# SnowGym Fighter RL Review and Next Milestones

**Repository:** `https://github.com/phunterlau/snowcraft-gym`

**Reviewed handoff:** `main` at `d3b96ad`

**Review date:** 2026-09-03

**Focus:** Fighter-agent RL/PPO design relative to the research goal of a slow LLM commander controlling fast autonomous learned agents.

---

## 1. Executive summary

SnowGym is already beyond the “RL-compatible environment” stage. It has deterministic headless combat, Gym/batch interfaces, replay/provenance, neural executors, BC/DAgger/PPO, symbolic command plans, host grounding, asynchronous commander scheduling, stale-plan reconciliation, and trajectory-triggered replanning.

The primary blocker is now the **plan-conditioned learned fighter**.

Four issues should be fixed before a serious M7 PPO run:

1. **The low-level reward does not define plan obedience.** Terminal win/loss allows the executor to ignore a plan whenever ignoring it wins more often.
2. **The nominal full-state observation is not fully Markov.** `noop` preserves hidden locomotion/controller state that is not sufficiently represented in the observation.
3. **The symbolic plan is not sufficiently bound to the physical members and progress of each role group.** This especially hurts support, cohesion, flank, and recovery behavior.
4. **The current PPO joint likelihood scales with living roster size.** A single fixed clip radius means a much smaller per-unit policy change in 10v10 than in 1v1.

The recommended conceptual correction is:

> **Train the fighter as a plan-conditioned temporally extended option executor with mission-defined success, then evaluate whether the commander’s choice of options improves canonical battle outcome.**

Recommended milestone order:

```text
M7a  Fighter contract repair
M7b  Fixed-plan option PPO
M7c  Full-fight fixed-plan composition
M8   Local CTDE / MAPPO executor
M8.5 Plan lifecycle learning
M9   Frozen-executor commander comparison
M10  Latency/generalization benchmark
```

---

## 2. Research framing

The intended hierarchy is:

```text
strategic state
     ↓
slow commander
     ↓
symbolic group plan
     ↓
host grounding
     ↓
plan-conditioned fighter
     ↓
fast physical actions
```

The commander should decide **what** the team should accomplish. The fighter should decide **how** to realize that intent under continuous dynamics.

A useful formalization is:

\$
P_k \sim \Pi_{\mathrm{commander}}(z_{t_k})
\$

\$
a_t^i \sim \pi_\theta(a_t^i \mid o_t^i, P_k, h_t^i),
\qquad t_k \le t < t_{k+1}
\$

The high-level plan is therefore an option-like latent/command that persists for many low-level decisions.

---

## 3. Severity-ranked findings

| Severity | Type | Finding | Consequence |
|---|---|---|---|
| **S0** | Objective inconsistency | Win/loss reward does not require plan obedience | Policy may rationally ignore hold/withdraw/support |
| **S0** | Environment contract | Hidden persistent movement/controller state is omitted | Nominal full-state feed-forward PPO is partially observable |
| **S0** | Representation | Symbolic role is weakly bound to physical role members/progress | Support/cohesion/flank semantics are ambiguous |
| **S1** | PPO design | Product likelihood over units is clipped with fixed epsilon | Effective trust region shrinks with roster size |
| **S1** | Credit horizon | `gamma=.99`, `lambda=.95` are short in real time at 10 Hz | Long missions rely heavily on critic quality |
| **S1** | Target policy | Hard nearest-target behavior conflicts with commander fire doctrine | Cannot cleanly learn focus/distributed symbolic targeting |
| **S1** | Critic | Value head lacks explicit role-progress representation | Weak plan-conditioned credit assignment |
| **S1** | Evidence | PPO gates largely retain BC competence | Material reward-driven improvement remains weakly shown |
| **S1** | Research claim | Current actor is centralized over squad/global state | Not yet “local RL agents” executing LLM intent |
| **S2** | Benchmark | Blue 10 Hz vs scripted Red 60 Hz | Useful challenge baseline, not symmetric policy comparison |
| **S2** | Reward scaling | Health potential scales with roster | Different shaping strength in 1v1 vs 10v10 |
| **S2** | Entropy | Entropy summed across active units | Exploration pressure changes with roster |

---

## 4. Root objective problem: command collapse

Current canonical reward is terminal:

\$
r_t=
\begin{cases}
+1 & \text{win}\\
-1 & \text{loss}\\
0 & \text{otherwise}
\end{cases}
\$

The plan enters the observation, but not the task definition. Therefore PPO is allowed to learn:

\$
\pi(a\mid s,P) \approx \pi(a\mid s)
\$

if the command is not needed to maximize win probability.

Example:

```text
Plan A: advance
Plan B: hold

If direct aggression wins more often under both,
PPO is rewarded for ignoring Plan B.
```

This is not an optimizer bug. It follows from the objective.

### Potential shaping is not enough

A plan-dependent potential:

\$
r'_t = r_t + \gamma \Phi_P(s_{t+1}) - \Phi_P(s_t)
\$

can accelerate learning, but if used in a policy-invariant form it should not redefine a fundamentally different mission as optimal.

Therefore define a mission objective explicitly.

Recommended executor reward:

\$
r_t^{\mathrm{executor}}
=
r_t^{\mathrm{mission}}
+
\eta r_t^{\mathrm{combat}}
+
\gamma\Phi_P(s_{t+1})-\Phi_P(s_t)
\$

Then evaluate the full hierarchy on canonical game reward only.

---

## 5. Train mission options before full fights

### Engage

Success:

```text
damage or eliminate designated enemy cluster
```

Progress can include target-health reduction and preferred-range control.

### Advance

Success:

```text
required fraction of assigned group reaches target region
```

Potential:

\$
\Phi_{advance}=-d(c_{group},R_{target})
\$

### Hold

Freeze the activation anchor:

\$
c_0=c_{group}(t_{activation})
\$

Success:

```text
remain within radius r for H seconds
while preserving minimum force
```

Do not redefine “hold current position” continuously.

### Withdraw

Use explicit phases:

```text
phase 1: reach safe/backfield region
phase 2: stop/hold after arrival
```

### Flank

Require useful geometry, not lateral movement alone:

```text
lateral progress
+ target-relative angle
+ contact after flank geometry
```

### Support

Train last. It requires explicit relational representation of:

```text
supported group position
supported group health/threat
threats attacking that group
supporting unit/group effect
```

---

## 6. Observation contract: full-state is not yet fully Markov

Current semantics imply:

```text
move  → persistent movement target
noop  → keep previous movement target
hold  → cancel movement target
```

If the persistent movement target/controller state is hidden, two identical neural observations can have different transitions under the same `noop` action.

That directly confounds hold, withdraw, support, and plan transitions.

### Recommended observation v3 fields

Expose stable public equivalents of:

```text
has_move_target
move_target_relative_x
move_target_relative_y

current_waypoint_relative_x
current_waypoint_relative_y

aim_direction_x
aim_direction_y

stun_remaining
recovery_remaining
immunity_remaining

previous_action_type
remaining_episode_fraction
decision_dt
```

The key rule is:

> **The fighter must be able to predict what `noop` means from the observation.**

A GRU can later be evaluated under intentionally partial observations, but it should not be required to reconstruct avoidable hidden simulator state in the full-state benchmark.

---

## 7. Bind symbolic groups to physical groups

Current symbolic data provides a group row matrix and per-unit role assignment, but the actor needs explicit physical role summaries.

For role `r`:

\$
g_r = \mathrm{Pool}_{i:C_{ir}=1,m_i=1}[e_i,x_i,v_i,h_i]
\$

At minimum include:

```text
role centroid
role velocity
role spread
living count
assigned count
health fraction
cooldown/readiness
objective-relative displacement
```

Per-unit plan-conditioned actor input should include:

\$
[e_i,d_i,g_{r(i)},g_{supported(i)},g_{objective},g_{global}]
\$

This supplies the missing mapping:

\$
\text{symbolic role} \leftrightarrow \text{current physical group}
\$

The existing bounded `[3,38]` symbolic plan format can remain unchanged initially.

---

## 8. Hybrid action likelihood

The current factorization is conceptually sound:

\$
\pi(A_i)=
\pi^{type}(a_i)
[\pi^x(x_i)]^{I_i^x}
[\pi^p(p_i)]^{I_i^p}
\$

Unused target/power dimensions should not enter PPO likelihood.

Keep this, but add exact tests:

```text
unused target dimensions → zero policy gradient
unused power → zero policy gradient
masked action type → zero probability
stored logprob == reevaluated logprob
tanh boundary numerics stable
sigmoid boundary numerics stable
```

### Separate move and throw target distributions

Use distinct heads/distributions:

\$
\pi_x^{move},\qquad \pi_x^{throw}
\$

Movement and aim have different geometry and exploration scales.

---

## 9. Replace hard nearest-target override with a residual prior

A hard nearest-enemy target conflicts with commander directives such as:

```text
focus weakest cluster
focus largest cluster
distributed fire
support threatened group
```

Preserve nearest-enemy competence as a prior rather than an override:

\$
u_i^{throw}
=
\mathrm{atanh}(x_i^{nearest})
+
\Delta_\theta(h_i,d_i,g_{r(i)})
\$

\$
x_i^{throw}=\tanh(u_i^{throw})
\$

Later, consider:

```text
enemy pointer distribution
+
continuous lead offset
```

for clean fire-allocation research.

---

## 10. PPO roster-scaling problem

Current joint log probability:

\$
\log \pi(A_t|s_t)=\sum_{i=1}^{N_t}\ell_{ti}
\$

so:

\$
\rho_t=\prod_i \rho_{ti}
\$

With PPO upper clip 1.2 and ten living units, equal per-unit ratio reaches the joint clip at:

\$
1.2^{1/10}\approx1.0184
\$

Only about **+1.84% per unit**.

The lower side is:

\$
0.8^{1/10}\approx0.9779
\$

about **−2.21% per unit**.

In 1v1, the full ±20% applies.

Thus the effective trust region changes with team size and casualties.

### Recommended primary PPO surrogate

Store per-unit log probability:

\$
\ell_{ti}=\log \pi_\theta(A_{ti}|s_t,P_t)
\$

Define:

\$
\rho_{ti}=\exp(\ell_{ti}-\ell_{ti}^{old})
\$

Then use shared team advantage but clip per unit:

\$
L_t^\pi
=
-\frac{1}{N_t}
\sum_i m_{ti}
\min
\left(
\rho_{ti}\hat A_t,
\mathrm{clip}(\rho_{ti},1-\epsilon,1+\epsilon)\hat A_t
\right)
\$

and:

\$
L^\pi=\mathbb E_t[L_t^\pi]
\$

Important:

> Average active units **inside each decision**, so a 10-unit decision does not automatically get ten times the weight of a 1-unit decision.

Keep the exact joint ratio as a diagnostic/ablation.

---

## 11. PPO diagnostics to add

Record:

```text
mean per-unit approximate KL
max per-unit approximate KL
normalized joint KL
per-unit clipping fraction
clipping by action type
ratio distribution by roster size
ratio distribution before/after casualties
```

Use target-KL early stopping based on mean per-unit KL.

---

## 12. Discounting should be defined in seconds

At 10 Hz:

```text
gamma = .99
```

has reward half-life of about 6.9 seconds.

```text
gamma * lambda = .9405
```

has GAE half-life of about 1.13 seconds.

That is short for flank, support, withdraw, and battle-level credit.

Define a physical return half-life `T_R`:

\$
\gamma(\Delta t)=2^{-\Delta t/T_R}
\$

At `dt=.1 s`:

```text
T_R = 30 s → gamma ≈ .997692
T_R = 60 s → gamma ≈ .998845
```

Similarly choose a GAE trace half-life and solve for lambda.

Predeclare a small sweep in **seconds**, not arbitrary discount constants.

---

## 13. Entropy redesign

Current entropy should be normalized per active unit.

A better approximation is:

\$
H_i
=
H_i^{type}
+
\pi_i(move)\hat H_i^{move}
+
\pi_i(throw)(\hat H_i^{throw}+\hat H_i^{power})
\$

\$
H_t=\frac{1}{N_t}\sum_i m_iH_i
\$

Potentially use separate coefficients for:

```text
action type
move target
throw target
power
```

Exact transformed entropy is secondary; unit normalization is the urgent correction.

---

## 14. Normalize health shaping across roster sizes

Current health potential magnitude grows with team size.

Use initial-roster normalization:

\$
\Phi_H
=
\frac{1}{N_B^0}\sum_i h_i^B
-
\frac{1}{N_R^0}\sum_j h_j^R
\$

Then the scale is approximately roster-independent.

Role potentials should also be bounded and weighted:

\$
\Phi_P=\sum_r w_r\Phi_r,\qquad \sum_r w_r=1
\$

Avoid duplicating the same team-health signal in every role term.

---

## 15. Recommended critic

Use a **separate centralized role-aware critic**, not the actor's target-path pool.

\$
V_\phi(s,P)
=
\mathrm{MLP}
([
 g_B,
 g_R,
 g_{projectile},
 g_{obstacle},
 g_{main},
 g_{maneuver},
 g_{reserve},
 e_P,
 \psi_{progress}
])
\$

`ψ_progress` can include:

```text
objective distance
hold displacement
flank angle
support relation
plan age
mission phase
```

A role-pooled MLP is preferable to a graph critic at this stage because there are only one to three strategic groups and interpretability matters.

---

## 16. Recommended initializer

Primary initializer:

> **Accepted target-only qualification checkpoint + zero-initialized shared plan residuals + fresh role-aware critic.**

Retain:

```text
entity encoders
physical action classifier
target heads
power behavior
existing plan-sensitive target path
```

Add zero/gated residuals to:

```text
action logits
move target
throw target
power
```

Use a **shared plan-conditioned residual**, not five fully separate mission experts.

Reason:

```text
shared residual → compositional reuse
mission experts → data fragmentation and weaker transfer
```

Use directive-expert v3 only as a secondary warm-start ablation.

---

## 17. Staged unfreezing

Recommended schedule:

```text
Stage 1
    freeze physical backbone
    train fresh critic + role pools + plan paths

Stage 2
    unfreeze action/target heads
    lower LR for inherited parameters

Stage 3
    optionally unfreeze final entity layers
    only after plan and physical gates pass
```

During early PPO, retain a decaying imitation/KL anchor:

\$
L
=
L_{PPO}
+
\lambda_{BC}L_{BC}
+
\lambda_{KL}D_{KL}(\pi_\theta\|\pi_{init})
\$

Anneal both anchors so PPO can eventually improve beyond the teacher.

---

## 18. What the failed supervised experiments already show

The retained failures are evidence, not dead ends.

They imply:

```text
plan signal reaches target prediction

but plan signal does not yet reliably control:
    move vs hold vs noop
    movement cancellation
    throw timing
    withdraw phase
    support relation
```

The fact that increasingly specialized supervised variants trade one mission against another suggests the bottleneck is not simply insufficient parameter count.

More plausible limiting factors are:

```text
hidden persistent state
missing physical role binding
objective conflict
closed-loop distribution shift
```

Therefore stop adding more supervised mission-expert variants before fixing those issues.

---

## 19. Centralized squad policy versus local agents

The current fighter is a centralized squad policy with global pooled context.

This is a valid first executor and useful upper bound.

It does **not** yet establish the final research claim:

> slow LLM commander + autonomous local learned fighters.

That requires CTDE:

\$
a_t^i\sim\pi_\theta(a_t^i|o_t^i,d_t^i,g_{r(i)}^{local},h_t^i)
\$

with centralized critic:

\$
V_\phi(s_t,P_t)
\$

Recommended path:

```text
corrected centralized PPO
    ↓
qualify plan execution
    ↓
parameter-shared MAPPO
    ↓
local observations
```

Do not begin MAPPO before the centralized formulation passes the mission suite.

---

## 20. Opponent cadence

Keep two benchmark conditions:

```text
Challenge:
    Blue 10 Hz learned executor
    Red 60 Hz scripted controller

Symmetric:
    Blue and Red both through 10 Hz action interfaces
```

The first tests robustness against a fast handcrafted opponent.

The second is needed for clean policy-vs-policy comparisons.

---

## 21. Freeze commander work temporarily

The commander subsystem is already sufficiently mature for the next fighter stage:

```text
bounded symbolic schema
validation
Hamilton allocation
late binding
asynchronous provider calls
fallback
reconciliation
trajectory triggers
trace/replay provenance
```

Do not prioritize more prompt engineering, additional verbs, or model-provider work now.

The research uncertainty is below the commander.

---

## 22. Evaluation claim ladder

### Claim A — Physical competence

The fighter:

```text
beats random
retains/improves BC competence
has near-zero rejected actions
works on held-out seeds
works across roster sizes
```

### Claim B — Plan controllability

From the same state:

\$
\pi(\cdot|s,P)\neq\pi(\cdot|s,P')
\$

Measure resulting trajectories, not only logits or target MSE.

### Claim C — Mission execution

Each mission must independently pass:

```text
engage
advance
hold
withdraw
flank
support
```

Do not hide mission failure inside an aggregate average.

### Claim D — Plan usefulness

With one frozen executor compare:

```text
correct plan
zero plan
shuffled plan
random valid plan
```

### Claim E — Local execution

Repeat plan gains under local actors/CTDE.

### Claim F — Commander value

Freeze executor and compare:

```text
fallback
random valid
rule commander
high-level RL commander
static LLM doctrine
online LLM
distilled commander
```

### Claim G — Latency robustness

Compare exact/high-bandwidth commands with symbolic late-bound group plans under controlled latency.

---

## 23. Recommended closed-loop metrics

| Mission | Metrics |
|---|---|
| Engage | target damage, contact time, preferred-range error, elimination |
| Advance | signed region progress, arrival time, fraction reaching |
| Hold | time inside anchor, max displacement, survival |
| Withdraw | backfield progress, safe-zone arrival, stopping after arrival, health preservation |
| Flank | lateral progress, angular separation, contact after flank geometry |
| Support | distance to supported group, threat coverage, supported-group survival delta |
| Focus | target-damage concentration / Herfindahl index |
| Distributed | target entropy, number of simultaneously engaged enemies |
| All | W/D/L, survivors, health, duration, rejection rate, plan age |

For same-state paired plan forks also record:

```text
action-type TV distance
target displacement
trajectory divergence
mission-outcome difference
```

---

## 24. Statistics

Suggested development level:

```text
30–50 paired seeds per mission condition
```

Frozen headline comparisons:

```text
100+ paired seeds per major condition,
subject to observed variance
```

Map generalization:

```text
10–20 sealed generated maps
multiple paired seeds per map
```

Use paired bootstrap confidence intervals and hierarchical bootstrap over map + seed.

Separate:

```text
training seeds
development seeds
qualification seeds
plan-generation seeds
sealed-map seeds
```

---

# 25. Recommended next milestone list

## M7a — Fighter Contract Repair

**Goal:** Make the nominal full-state fighter a valid plan-conditioned control problem and make PPO updates roster-stable.

Implement:

```text
[ ] observation v3 exposes persistent movement/controller state
[ ] explicit decision_dt
[ ] explicit remaining episode horizon
[ ] role assignments bound to physical ally embeddings
[ ] role-conditioned physical group summaries
[ ] supported-group summary
[ ] separate move/throw target distributions
[ ] hard nearest-target becomes residual prior
[ ] per-unit PPO log probabilities
[ ] per-unit clipped PPO surrogate
[ ] per-decision active-unit normalization
[ ] per-unit entropy normalization
[ ] physical-time gamma/lambda configuration
[ ] roster-normalized health shaping
[ ] separate role-aware centralized critic
[ ] controller dt contract corrected
[ ] symmetric 10 Hz opponent mode
```

**Exit gate:**

```text
full-state action consequences are predictable from observation
PPO update statistics are comparable across 1/3/5/10 units
same seed/action traces remain deterministic
```

---

## M7b — Fixed-Plan Option PPO

**Goal:** One frozen checkpoint can execute each command family independently.

Initializer:

```text
accepted target-only checkpoint
+
zero-init shared plan residuals
+
fresh role-aware critic
```

Curriculum:

```text
1. engage
2. advance
3. hold
4. withdraw
5. left/right flank
6. focus/distributed fire
7. support
```

Then compose:

```text
main only
main + maneuver
main + reserve
main + maneuver + reserve
```

Keep plans fixed for the option episode/segment.

**Exit gate:**

```text
[ ] every mission passes separately
[ ] physical combat competence retained
[ ] same-state plans cause causal trajectory differences
[ ] PPO materially improves beyond initializer
[ ] nontrivial learning rate is used
[ ] one checkpoint works across mission families
```

---

## M7c — Full-Fight Fixed-Plan Composition

**Goal:** Sustain multi-role plans throughout full battles before lifecycle replanning.

Test:

```text
60% main / 30% flank / 10% reserve
70% main / 30% maneuver
70% main / 30% support
focus vs distributed
left vs right flank
```

Include:

```text
casualties
fragmentation
target replacement
terrain
longer horizon
```

**Exit gate:**

```text
[ ] role behavior remains distinct
[ ] plan adherence survives casualties
[ ] no command collapse
[ ] 3v3 → 5v5 → 10v10 scaling measured
[ ] full-fight performance remains competent
```

---

## M8 — Local CTDE / MAPPO Executor

**Goal:** Transition from centralized squad policy to parameter-shared local fighter agents.

Actor receives:

```text
local fighter state
local visible enemies/projectiles/obstacles
own directive
own role summary
supported-role summary
optional recurrent state
```

Critic receives:

```text
global state
all group assignments
active plan
role progress
```

Use MAPPO/CTDE.

**Exit gate:**

```text
[ ] local actor learns above random
[ ] mission suite remains qualified
[ ] plan effects remain measurable
[ ] 3v3 and 5v5 stable
[ ] centralized fighter remains documented upper bound
```

---

## M8.5 — Plan Lifecycle Learning

**Goal:** Introduce plan replacement only after fixed-plan execution works.

Add:

```text
plan age
source-state age
plan version
plan_changed
activation anchor
mission phase
previous directive or recurrent memory
```

Curriculum:

```text
fixed plan
→ scheduled boundary switch
→ stale-state switch
→ event-triggered replan
```

Measure:

```text
adaptation time
old-plan persistence
new-plan adherence
oscillation
mission recovery
```

---

## M9 — Freeze Executor and Compare Commanders

Freeze one qualified learned executor.

Compare:

```text
fixed fallback
random valid plan
rule commander
high-level RL commander
static LLM doctrine
online LLM
distilled commander
```

All arms share:

```text
same CommandPlan schema
same grounder
same fallback
same executor
same request opportunities
same scenario seeds
same latency condition
```

Measure:

```text
W/D/L
mission completion
plan validity
repair rate
plan age/churn
tokens
latency
cost
action rejection
```

---

## M10 — Latency and Generalization Benchmark

Latency sweep:

```text
0 ms
100 ms
250 ms
500 ms
1 s
2 s
4 s
8 s
```

Compare command representations:

```text
exact per-unit assignment
exact target IDs/coordinates
symbolic late-bound group plans
```

Evaluate across:

```text
3v3 / 5v5 / 10v10
seen maps / sealed generated maps
seen doctrine / held-out opponent doctrine
```

Primary hypothesis:

> **Late-bound group plans plus autonomous local execution degrade more gracefully under commander latency and state drift than high-bandwidth individual commands.**

---

# 26. Recommended PR order

```text
PR 1   Observation v3 + role-aware physical summaries
PR 2   Per-unit PPO clipping + unit-normalized entropy
PR 3   Separate role-aware centralized critic
PR 4   Separate move/throw distributions + residual target prior
PR 5   Mission option environments and reward definitions
PR 6   Engage/advance/hold PPO
PR 7   Withdraw/flank/fire-doctrine PPO
PR 8   Support PPO + relational metrics
PR 9   Full-fight fixed-plan composition
PR 10  Unit-local CTDE/MAPPO adapter
PR 11  Plan-lifecycle learning
PR 12  Frozen-executor commander benchmark
```

---

# 27. Experiments to stop/defer

Defer:

```text
more five-way supervised directive-expert variants
online LLM over learned executor
large prompt/model sweeps
plan lifecycle during initial option PPO
graph critic
pixels
self-play league
aggregate-only mission qualification
```

These add variance before the fighter contract and mission semantics are qualified.

---

# 28. Experiments to add now

```text
current observation
vs actuator-state observation
vs GRU
```

```text
joint PPO ratio
vs per-unit clipping
vs geometric-mean surrogate
```

```text
hard nearest target
vs residual prior
vs enemy pointer
```

```text
current critic
vs separate global critic
vs role-aware critic
```

```text
correct plan
vs zero plan
vs shuffled plan
```

```text
same-state paired closed-loop plan forks
```

```text
target-only initializer
vs directive-expert initializer
```

```text
3v3
vs 5v5
vs 10v10
```

```text
60 Hz scripted Red
vs symmetric 10 Hz Red
```

---

# 29. Strongest immediate experiment

Before full M7b, run one deliberately narrow causal test.

Start from the **same physical state** and fork three futures:

```text
A. HOLD
B. WITHDRAW
C. ADVANCE
```

Hold fixed:

```text
opponent behavior
random future
initial state
fighter checkpoint
```

Measure over a fixed horizon:

```text
centroid displacement
anchor retention
backfield progress
survival
action-type distribution
movement-target distribution
```

If one policy cannot produce three cleanly separable trajectories from the same state, stop there.

Do not proceed to support, lifecycle replanning, MAPPO, or online LLM command.

This is the cleanest test that the executor truly understands the command channel.

---

# 30. Blockers for online LLM evaluation

Do not use the learned fighter for the headline online-LLM study while any of these remains true:

```text
[ ] any core mission family fails
[ ] shuffled plan behaves like correct plan
[ ] plan-conditioned fighter materially regresses physical competence
[ ] action rejection exceeds code executor materially
[ ] support remains ungrounded
[ ] only 3v3 is qualified
[ ] PPO only preserves BC rather than improving it
[ ] plan replacement causes oscillation
[ ] qualification only uses development seeds/maps
```

The code-based reactive executor remains the right commander-integration substrate until these gates pass.

---

# 31. Final recommendation

The next development objective should not be simply:

```text
“improve PPO”
```

It should be:

> **Make plan execution a well-defined Markov option-control problem, then train explicit mission skills whose success is independently measurable.**

The four highest-priority corrections are:

\$
\boxed{
\text{mission-defined executor objective}
+
\text{observable persistent controller state}
+
\text{role-to-physical-group binding}
+
\text{roster-stable PPO objective}
}
\$

Then proceed:

```text
fighter contract repair
    ↓
fixed-plan option PPO
    ↓
full-fight plan composition
    ↓
local CTDE/MAPPO
    ↓
plan lifecycle
    ↓
frozen-executor commander comparison
    ↓
latency/generalization benchmark
```

That ordering most directly supports SnowGym's central research claim:

> **A slow semantic commander should reduce the strategic decision burden of fast autonomous RL fighters without taking over their real-time physical control.**
