# SnowGym through R1j: Review and Next RL Experiments

**Review date:** 2026-09-04  
**Primary evidence:** `snowgym_rl_recovery_r1j_handoff_for_gpt_pro_2026-09-04.md`  
**Revision reported by that handoff:** local `6827894`; locally recorded remote `ec1b21f`  
**Scope:** Evidence review and proposed experiments, not a fresh execution audit of unpublished commits.

The handoff says R1i/R1j include unpublished local commits. This review therefore treats the supplied document as the authoritative report of those experiments. It does not claim to have inspected those local changes, rerun training, or verified the stored artifacts independently. Citations `[H §n]` refer to sections of the handoff; numbered external references at the end provide methodological context, not evidence about SnowGym's results.

## Executive decision

The experiments have substantially narrowed the problem. They do **not** currently justify another joint PPO run, another general decoder redesign, or a conclusion that PPO cannot learn SnowGym.

The next best experiment is a **fixed-architecture, learner-action-conditioned supervision audit**, followed—only if supported—by corrective-data training against a matched old-data control.

The important distinction is:

> A target head trained where the teacher selects an action is not necessarily trained where the learner selects that action.

The most promising subsequent RL experiment is **movement-only PPO with corrected shots held fixed**, explicitly labeled teacher-assisted. This follows directly from R1h and creates a narrower optimization problem with observed nonzero success. It is a diagnostic/curriculum task, not autonomous-fighter qualification.

Priorities:

1. Preserve the corrected simulator and option semantics; do not reopen already repaired issues without a failing test.
2. Audit conditional-label coverage, target-loss gradients, and physical consequences on deterministic learner states from training seeds.
3. Run one fixed-architecture corrective-data experiment with explicit causal controls.
4. Introduce a valid stochastic policy contract and resume RL on one control channel at a time.
5. Retain the original autonomous qualification gates, and add a fresh development replication set before major promotion decisions.

---

## 1. What the evidence now supports

| Evidence | Supported interpretation | Unsupported interpretation |
|---|---|---|
| Random Red actions were not applied before simulation v2 | Older random-opponent results do not establish competence against active random Red | All prior experiments are invalid; scripted-Red gates are also affected |
| R1g: corrected throw direction changes success from 0/40 to 7/40 | Target selection/aim geometry is causally useful on the tested trajectories | Power alone is the main issue; direction replacement isolates aiming from target choice |
| R1h: corrected shots plus teacher movement reaches 40/40 | Geometry substitutions are sufficient for this frozen benchmark while retaining learner action choice | The resulting controller is an autonomous learned fighter |
| R1h: teacher action choice changes 10/40 to 11/40 with corrected shots | Little evidence that categorical action choice is the next dominant bottleneck in this condition | Action choice will never matter in another mission or after learning geometry |
| R1i absolute: 100% contact, 100% any-hit, 30.8% mean progress, 0/40 completion | Geometry learning improved intermediate combat substantially | Contact or one hit qualifies Engage |
| R1i relative vs absolute, and R1j decoder factorial | Tested variants did not demonstrate a reliable improvement over the absolute reference | Relative representations and direction policies generally do not work |
| PPO paused after R1e | R1f–R1j mainly diagnose representation, supervision, and deployment behavior | R1i/R1j are negative evidence about a newly trained PPO optimizer |

Source: [H §§3,6–9].

The R1h movement effect is conditional on corrected shots: +75 percentage points relative to that assisted baseline, with the reported interval [60, 87.5]. It is not an unconditional effect of movement under the raw learner. The movement replacement also bundles range, formation, cohesion, and dodge behavior. [H §7]

The current contact/any-hit metrics are close to saturated for R1i. Diagnostics should now emphasize **hit probability per shot, repeated useful damage, survival under sustained engagement, and time from first hit to mission completion**.

### Corrections to earlier advice

Do not propose the successful-teacher reservoir as if it were missing: it was added, weighted, and extended in R1/R1d/R1e. Do not repeat “fix absolute coordinates” as a proven solution: the controlled R1j variants failed to earn promotion. Do not present the six original defects as still open; the handoff reports repairs, with narrower remaining boundaries. [H §§3,6,9]

---

## 2. Leading hypothesis: conditional-head coverage mismatch

Let:

- `s` denote physical, plan, and option state;
- `k` denote an action type;
- `a_E(s)` denote the teacher's selected action type;
- `a_L(s)` denote the learner's selected action type;
- `f_k(s)` denote the learner's conditional parameter head;
- `g_E^k(s)` denote a recommendation for that head, when independently defined.

Teacher-masked supervision optimizes approximately:

\[
L_k^{old}=\mathbb E_{s\sim D_E}
\left[\mathbf 1\{a_E(s)=k\}\,\ell_k(f_k(s),g_E^k(s))\right].
\]

But the head is used under a different distribution:

\[
s\sim d_{\pi_L},\qquad a_L(s)=k.
\]

There are **two separate mismatches**:

1. **State distribution:** successful-teacher trajectories versus learner trajectories.
2. **Action-conditioned distribution:** states where the teacher uses a head versus states where the learner uses it.

A generic DAgger-style collector can address the first and still retain the second if it only labels the parameters of the teacher-selected action. DAgger provides the relevant interactive-learning principle, but the conditional-head extension here is a SnowGym-specific proposal. [1]

The reservoir has 1,162 throw labels out of 26,815 living-unit labels, about 4.3%. Reweighting those labels can alter optimization pressure; it cannot supply throw geometry for states excluded by the teacher-action mask. This is a plausible mechanism, not an established explanation of the failure. [H §§5,10]

### Concrete example

```text
At the same state:

teacher action: move
learner action: throw

Old BC:
    supervise movement target
    do not supervise throw direction/power

Deployment:
    execute the unsupervised-at-this-state throw head
```

The correct extra label is not “the agent should throw.” It is:

> If a throw is executed here, this is a valid direction/power recommendation.

Keep categorical-action supervision separate from conditional geometry supervision.

---

## 3. Proposed R1k: no-training audit on actual learner action opportunities

### 3.1 Freeze a reference

Use the R1i absolute final checkpoint, which the R1j absolute arm exactly reproduced. It is a controlled reference, not a newly selected best checkpoint. Preserve its architecture, action-type classifier, decoder, initialization digest, and parent lineage. [H §§8–9]

Collect deterministic trajectories on training seeds only. Do not collect corrective training labels from development or qualification states.

Retain the original reservoir unchanged. Create any genuinely unseen teacher-regression holdout from fresh training-pool seeds: a subset already used to fit the source checkpoint is not an unseen holdout merely because a later fit excludes it.

### 3.2 Record action-opportunity rows

For every living fighter at each sampled decision, save:

```text
physical observation and plan/option state
learner action type and all conditional-head outputs
teacher action type
legal-action mask
independently defined movement recommendation, validity, provenance
independently defined throw direction/power, validity, provenance
selected enemy and target margin for analysis only
range, relative velocity, cooldown/readiness, incoming threats
previous movement target / waypoint
phase: approach, contact, sustained engagement, retreat/recovery
source checkpoint, scenario, seed, tick, state hash
```

Reuse the precisely defined recommendation helpers from R1h where applicable. Do not invent a new teacher during data collection. Their success is evidence of usefulness on the reported trajectories, not proof of optimal recommendations in every learner state. [H §7]

### 3.3 Cross-tabulate teacher and learner decisions

For each pair `(teacher_action, learner_action)`, report:

- sample count and deployment frequency;
- recommendation availability and legality;
- move error, angular error, power error;
- physical effect of replacing that particular head;
- whether the sample would have been excluded by the old mask.

An informative statistic is:

\[
U_k=\frac{\#\{a_L=k\text{ and old teacher mask excludes head }k\}}
{\#\{a_L=k\}}.
\]

`U_k` measures an exact masking mismatch on collected states. It is not itself a statistical proof that similar labeled states never existed in the reservoir.

Compare errors in the intersection `a_E=k, a_L=k` with errors in `a_E!=k, a_L=k`. A large discrepancy concentrated in the second set supports the conditional-coverage hypothesis.

### 3.4 Distinguish coverage from optimization

Use a staged test on a frozen, stratified set of hard training states:

1. Query valid recommendations on those exact states.
2. Substitute the recommendations in short simulator branches and check that physical outcomes improve.
3. Fit the unchanged geometry modules on a small hard-state subset.
4. Verify error reduction on that subset and on disjoint learner-state episodes.
5. Re-run short closed-loop branches from the audited states.

Interpretation:

| Result | Next conclusion/action |
|---|---|
| Recommendation replacement helps; model fits hard states; broad error is concentrated in previously unlabelled opportunities | Corrective-data experiment is justified |
| Replacement helps, but hard-state fit stalls despite valid gradients | Audit representation limits, conditioning, saturation, conflicting labels |
| Small-batch error improves, but physical consequences do not | Loss/label quality is the limiting issue |
| Recommendations themselves do not help on these states | Do not force their imitation; reconsider feasibility or cost-sensitive alternatives |
| Fit works only on the tiny set, not neighboring learner states | Generalization/representation remains a candidate, not simply missing optimizer steps |

R1i/R1j already passed disposable 32-state gates. Repeating an easy gate is not useful; these must be the difficult states and action combinations encountered by the learner. [H §§8–9]

---

## 4. Gradient and representation audit

For movement, ray, endpoint, and power losses separately, measure:

```text
raw and weighted loss
sample count / effective weight
per-module gradient norm before clipping
gradient cosine against every other component
fraction of saturated tanh outputs
predicted ray length and near-degenerate rate
local physical-output Jacobian norms
finite-difference agreement with autograd
```

Verify target losses reach the new entity encoder, not merely the residual output layer. The handoff reports that the inherited target path detaches shared actor features and that R1i supplied a differentiable target path. This new path needs direct reachability tests on the relevant minibatches. [H §§3–4,8]

Nonzero gradients alone do not establish useful conditioning. Inspect the physical changes produced by equal optimizer steps in world units and degrees.

For the bounded displacement arm, report the fraction of labels outside its attainable correction range. If its per-axis correction is limited to 10 world units, the unattainable component can be diagnosed by:

\[
e_{j,min}\geq\max(|g_j^*-x_{0,j}|-10,0)
\]

before considering arena constraints. This is a representation floor, not a training failure. Do not infer a similar bound for another decoder without checking its actual mapping. [H §9]

Do not replace the actor initialization simply because it is old. Change it when data show that the inherited path imposes substantial irreducible bias, inaccessible features, saturated corrections, or poor conditioning that remains after the fixed-architecture audit.

---

## 5. Supervise physical consequences, not just coordinate agreement

The engine turns shooter-to-target displacement into a ray; target distance is not projectile range. Thus two endpoints at different radii may produce the same throw direction, while a numerically close endpoint can produce a bad shot. This matches R1f and R1j: endpoint improvements did not consistently improve angular agreement or completion. [H §§6–9]

### 5.1 Movement error

Convert normalized differences into world units before comparison:

\[
e_m=(50(\hat x_x-x_x^*),\ 40(\hat x_y-x_y^*)).
\]

A dimensionless candidate loss is:

\[
L_m=\operatorname{Huber}(\|e_m\|/d_{tol}).
\]

Choose `d_tol` from a declared movement tolerance or measured sensitivity experiment, not simply a number that gives the desired scalar loss magnitude.

Endpoint agreement remains a surrogate. Also measure range error, immediate steering direction, dodge clearance, cohesion, and short-horizon damage exposure. Two endpoints can induce similar useful paths; forcing exact equality can penalize good alternatives.

### 5.2 Throw error

Use world-coordinate rays:

\[
\hat d=\frac{\hat x-p}{\|\hat x-p\|},\qquad
 d^*=\frac{x^*-p}{\|x^*-p\|}.
\]

Track angular error and a task-scale miss estimate:

\[
e_\perp=R\sin(\theta),\qquad
\theta=\operatorname{atan2}(|\hat d\times d^*|,\hat d\cdot d^*).
\]

For illustration only, a 20-degree directional error at range 20 corresponds to roughly 6.8 units of lateral miss in a planar straight-line calculation. This is not a claim about SnowGym collision radii, projectile height, or hit tolerance. Measure those from the engine.

A candidate scaled surrogate is:

\[
L_{aim}=\operatorname{Huber}\!\left(
\frac{R\,2\sin(\theta/2)}{b_{tol}}
\right).
\]

The chord form avoids falsely assigning zero loss to a directly backward ray, which `sin(theta)` alone would do. Clip extreme weights and validate gradients; this is a proposal, not a demonstrated improvement over R1i's cosine loss.

Separate wrong-target selection from lead/aim error. The teacher picks a nearest target with deterministic ties; a prediction between two valid targets can be bad even when averaged endpoint error is small. Conversely, hitting another valid opponent need not be inferior merely because it disagrees with the teacher. Audit target-identity margins and actual hit feasibility before changing architecture.

### 5.3 Power

Use a declared scale:

\[
L_p=\operatorname{Huber}((\hat p-p^*)/p_{tol}).
\]

Better, calibrate power error by simulated hit/damage consequences. R1g did not establish a separate power-only benefit; prioritize movement and direction. [H §7]

### 5.4 Per-head normalization

Normalize each loss by its own valid sample count, then average across episodes or controlled strata. Do not let thousands of movement labels numerically erase a small number of important throw labels. Do not call a 0.9 loss weight “90% of samples”; R1d/e used equal learner/reservoir sample counts with unequal loss weights. [H §5]

---

## 6. Use simulator consequences as a cost-sensitive diagnostic

The decisive quantity is not necessarily distance to the teacher's command. It is the harm caused by the difference.

For a training state and a particular control channel, estimate:

\[
\Delta J_k(s)=J_H(s,a_k^{reference};\pi_c)
-J_H(s,a_k^{learner};\pi_c),
\]

where all other channels and a predeclared continuation policy `pi_c` are held fixed. `J_H` can report target damage, incoming damage, preferred-range error, or mission progress separately.

Use one-decision substitutions followed by the common continuation first. If an action persists, its future effects are naturally part of the branch. Do not silently turn this into an H-step forced expert intervention.

AggreVaTe supplies methodological precedent for learning from action consequences/cost-to-go rather than treating every imitation error equally. The bounded local branch protocol above is our proposed application, not a replication of that algorithm. [2]

Branching requires complete reproducible state: RNG, controller state, projectiles, plan assignments, option tracker, and clocks. A public hash is a verification aid, not a restorable simulator snapshot. If full snapshots are unavailable, deterministically replay the action prefix from reset. Pair exogenous randomness where supported; the same seed does not imply identical state-dependent opponent actions after branches diverge.

Do not declare the teacher optimal. Use its candidate as a reference that must itself pass a usefulness check.

---

## 7. Proposed R1l: conditional corrective-data experiment

Only run this if R1k supports it.

Hold fixed:

```text
R1i absolute architecture and decoder
reference starting weights
frozen action-type classifier
trainable geometry modules
loss definitions and coefficients
optimizer, gradient clipping, and update count
final-checkpoint selection rule
simulation v2, task, scenario, and opponent
```

Use a 2x2 design if resources permit:

| Arm | State support | Geometry labels |
|---|---|---|
| A | original teacher reservoir | teacher-selected-action masks |
| B | reservoir plus learner training states | teacher-selected-action masks |
| C | original teacher reservoir | independently valid conditional-head recommendations |
| D | reservoir plus learner training states | independently valid conditional-head recommendations |

This separates visiting different states from exposing the right heads on those states.

For a minimum-cost pilot, compare A and D, but call any improvement a combined corrective-data effect. It would not isolate which factor caused it.

Use equal optimizer steps, not equal epochs over differently sized datasets. Record distinct states, labels per head, effective weights, and oracle query counts. Predeclare sampling proportions and retain old-data rehearsal. Match head exposures where possible; new label availability is an intentional intervention and must remain visible in the report.

A DAgger-style extension should aggregate the states from each new learner iteration rather than replacing earlier data. Keep PPO completely out of this first corrective-data fit. [1]

### Training labels versus deployment

At training time, conditional recommendations may be queried for all legal and meaningful action opportunities, including unselected heads. At deployment, only the chosen action's head is executed. Later PPO log probabilities still include **only parameters actually sampled for the chosen action**. Additional supervised labels do not justify including unused action dimensions in PPO likelihoods.

No valid shot recommendation is a distinct outcome: do not invent a target, train an illegal throw, or silently use zero coordinates. Record validity/feasibility and keep the action-type label separate.

### Exit decision

Promote only if physical closed-loop behavior improves, not merely teacher-state MSE. Preserve the existing R1 bootstrap gate. If D improves hard-state fitting but not sustained combat, inspect local consequence errors before adding more data again.

---

## 8. The most promising actual RL experiment after the audit

### Movement-only PPO with corrected shots

R1h offers a useful starting point: the corrected-shot controller with learner movement already achieves 10/40 Engage successes. Changing only movement recommendations takes it to 40/40. [H §7]

Use this to define a **teacher-assisted movement-learning task**:

```text
frozen learner action-type choice
fixed corrected throw direction and power
learned movement distribution only
unchanged Engage completion criterion
```

Reproduce the exact frozen R1h baseline first; do not assume its success rate transfers unchanged to another checkpoint.

This removes simultaneously learning aim, movement, firing choice, and power from the first RL experiment. Nonzero starting success offers more useful reward contrast than the original all-failure run. Whether PPO improves it remains an open empirical question.

Only the sampled movement parameters enter the trainable actor objective. Teacher-set geometry is a fixed part of this assisted environment/controller, not a sampled action attributed to the learner. Persist assist type and version in every manifest.

Then use the complementary experiment:

```text
fixed movement helper
fixed power initially
learned throw direction
```

Finally combine learned movement and aiming and remove assistance. A fresh autonomous 5v5 evaluation is mandatory: skills learned under assistance may fail when composed.

**These assisted results do not close M7b.** They measure learnability of specific motor channels and can support a curriculum, not the final all-learned-executor claim.

### Optional alternative research scope

A code-based nominal local controller plus a learned residual is also a legitimate *different* baseline. Residual RL explicitly studies combining a conventional controller with learned corrections. [3]

That baseline may be well matched to the commander research question: must every aiming calculation be learned to test strategic orchestration? But it changes the claim and must not be mislabeled as the existing autonomous neural-fighter qualification.

Because the full teacher already achieves 40/40 on this benchmark, there may be no measurable success-rate headroom for a teacher-plus-residual controller here. Any improvement study would need predeclared additional difficulty/generalization conditions and a matched nominal-controller baseline, without weakening the existing qualification task.

---

## 9. Probability contract before returning to PPO

The R1i/R1j policies are deterministic inference probes, not completed stochastic PPO policies. This is explicit in the handoff. [H §9]

### 9.1 A valid movement distribution

A state-bounded, invertible transform is one option:

\[
z_m\sim\mathcal N(\mu_m,\operatorname{diag}(\sigma_m^2)),
\qquad x_m=c(s)+b(s)\odot\tanh z_m.
\]

`c(s)` and positive `b(s)` define a valid rectangle in world coordinates using a frozen, state-dependent mapping. Include the affine and tanh Jacobians when logging density in executed-target space. An alternative is to define the latent action explicitly and store its exact latent log probability; keep that contract consistent across collection and update.

Calibrate exploration by **executed world-space displacement and path perturbation**, not a generic normalized log standard deviation. Movement and aiming need separate scales. Verify zero-noise rollout parity before stochastic training.

### 9.2 A valid shot distribution

Since physical throw geometry is a direction, a circular distribution is natural:

\[
\theta\sim\operatorname{VonMises}(\mu_\theta,\kappa),\qquad
\log p(\theta)=\kappa\cos(\theta-\mu_\theta)-\log(2\pi I_0(\kappa)).
\]

Map the sampled angle to a positive-length endpoint on the same ray inside arena bounds. Do not clip x and y independently if that would rotate the sampled direction. Handle boundary/degenerate rays explicitly.

Power can retain its sigmoid-Normal contract or use a separately tested Beta distribution. PyTorch documents von Mises, Beta, and transformed distributions. Von Mises does not need reparameterized sampling for PPO's score-function objective. [4]

This is **not** a recommendation to repeat R1j's failed deterministic direction decoder. It specifies the probability law required if direction-based actions are later trained with PPO.

### 9.3 Many-to-one decoders

Normalizing a 2-D sample into a direction is many-to-one. Two valid approaches are:

1. define a correct directional density and log that action;
2. define the original vector as the policy's latent action, retain it, and evaluate its exact density under a fixed environment mapping.

The second is mathematically possible but contains a redundant radial dimension. Do not normalize or clip samples, discard their source, then pretend the result has the original Gaussian density. Also do not promote high latent entropy as evidence of useful physical exploration.

### 9.4 Exploration continuity

Start with modest independently sampled noise and a fully tested likelihood. Later compare a verified state-dependent exploration method or a carefully defined macro-action policy. gSDE is relevant precedent for smoother exploration. [5]

Do not reuse an unlogged random waypoint offset for several decisions while evaluating independent per-step Gaussian probabilities. Persistent noise requires its actual conditional behavior distribution or latent history to be represented. Macro-actions require duration-aware rewards and discounting.

### 9.5 Optimizer rules

Retain the current per-unit PPO surrogate as an explicitly chosen surrogate, not an exact joint importance ratio. Keep critic and actor clipping separate, audit BC and initializer-anchor effects on the **total** update, and perform KL stopping using actually collected behavior-policy data. PPO clipping does not constrain extra auxiliary gradients. [H §4; 6]

Teacher data stays in supervised auxiliaries. No teacher-executed transition enters an on-policy PPO ratio or advantage as if sampled from the learner.

---

## 10. Repairs required before further option RL

The handoff distinguishes repaired historical defects from remaining ones. Keep that distinction. [H §§3,10]

### Before another option-PPO run

- Expose the remaining **option** budget, not only remaining full-battle time. For this fixed reset-time Engage task it may be inferable, but an explicit contract prevents ambiguity under later starts/curricula.
- Freeze activated Engage target membership and its health denominator for mission scoring. A late-bound tactical aim target may change; the scored mission identity must not silently change.
- Test success before resolver fallback can replace an eliminated target.
- Preserve option-timeout terminal semantics and zero terminal potential.
- Enforce simulation v2, hash version, opponent implementation, and checkpoint/initializer lineage.
- Retain a random-Red application regression test; distinguish its target-aware structure from uniform-coordinate random actions.

The reported multi-cluster identity bug does not explain the current one-cluster benchmark. Repair/test it for correctness without claiming it caused R1j's failure.

### Before temporal/multi-role options

Expose sufficient option state: occupancy/settling counters, flank phase and timing, accumulated damage statistics, activation anchors, and supported-role progress. Test width/height versus diagonal coordinate reconstruction. These issues need not block a frozen-policy Engage label audit, but should block later mission claims.

### Potential shaping caution

For a fixed initial state and absorbing terminal potential:

\[
\sum_{t=0}^{T-1}\gamma^t
[\gamma\Phi(s_{t+1})-\Phi(s_t)]
=-\Phi(s_0).
\]

Therefore approach shaping can improve temporal credit propagation, but it is not additional net preference among otherwise identical terminal failures. Do not promise that shaping alone fixes an all-failure regime. In R1i, contact and first hits already occur: repeated useful damage and survival are now more informative bottlenecks than first contact. [H §8]

---

## 11. Evaluation: what to retain and what to strengthen

Retain the existing R1 gate:

```text
Engage success >= 50%
correct-plan success >= wrong-plan control + 20 percentage points
correct-plan success >= initializer + 20 percentage points
contact >= 80%
rejections essentially zero
```

R2 and full autonomous qualification remain unchanged. A HOLD-input preview is one wrong-plan control; it does not establish successful HOLD execution or broad plan understanding. [H §§5,11]

Add:

```text
hits / legal throws
damage / ready-to-act opportunity
damage received while engaging
useful range occupancy
time from first hit to completion (including censored failures)
per-unit and per-episode tail geometry error
correct-plan versus HOLD-input behavior under fresh seeds
```

The reused 40 development seeds remain useful for historical comparability and debugging. They are not a fresh confirmation set after many adaptive choices. Keep qualification sealed, and predeclare a new replication-development set before the next major promotion decision. Use multiple training RNG seeds—preferably at least three for the selected recipe—because 40 evaluation environments do not measure optimizer-seed variability.

If the source model has already seen all teacher-reservoir episodes, a later partition of those episodes is a regression set, not a clean unseen test.

Report confidence intervals as exploratory on the reused set. Preserve final-checkpoint selection and avoid retrospective best-update selection. Zero successes in 40 seeds does not prove population equivalence between decoders. [H §§9–10]

---

## 12. Milestone and decision list

| Proposed milestone | Deliverable | Decision criterion |
|---|---|---|
| **R1k — Conditional coverage and consequence audit** | Teacher/learner action cross-tabs, independent head labels, gradient reachability, hard-state fits, paired physical substitutions | Identify whether missing labels, optimization, or loss/representation is dominant |
| **R1l — Matched corrective-data fit** | Fixed architecture; old-data control; preferably state-support x label-support factorial | Improve actual learner-state geometry and autonomous completion, not just reservoir MSE |
| **R1m — Scoped PPO mechanism test** | Valid stochastic movement policy with shots fixed; then complementary aim task | Reward-driven improvement over matched assisted initialization, with assists explicitly labeled |
| **R1n — Remove assistance and requalify Engage** | One neural checkpoint, autonomous active-random benchmark, original gates plus fresh replication | Close R1 only on autonomous evidence |
| **Later M7b/M7c** | Universal missions, composition, repaired temporal state | No isolated checkpoint per mission; no hidden teacher overrides |
| **Later M8/M9** | Local actors/centralized critic, frozen-executor commander study | Separate executor value from commander, grounder, and information advantages |

These are suggested new labels, not claims about already implemented milestones.

### Avoid next

Do not prioritize another broad PPO sweep, a larger transformer, mission experts, a fifth decoder variant, online LLM calls, or new commander vocabulary. Do not declare coverage solved merely because an offline loss falls.

### Escalation rule

If valid on-learner-state labels cannot be fitted by the fixed architecture, stop extending the dataset and test accessibility/conditioning or a fresh geometry head. If labels fit but do not improve short-rollout outcomes, change the training target or use cost-sensitive consequences. If consequences improve locally but not globally, investigate sequential distribution shift and coordinated engagement, then aggregate new learner states.

---

## 13. Research-level conclusion

There are two nested tasks:

1. Learn reliable motor geometry and sustained combat behavior.
2. Condition that capability on a strategic group instruction.

Current R1i/R1j results do not yet cleanly test the second task because the first remains unreliable. Neither the LLM nor delayed-command orchestration is involved in these recovery probes. [H §§2,8–10]

A code-based local-controller baseline is legitimate for studying the hierarchy, provided its role is explicit. It must not replace the all-learned claim without changing the research scope. Conversely, insisting that a neural model first rediscover every deterministic geometric rule can consume effort unrelated to the distinctive commander question.

**Immediate recommendation:** follow the handoff's fixed-architecture audit, sharpen it around learner-selected action opportunities, and make the next improvement test causal. Then use the successful channel interventions to build a narrow, reward-informative RL task rather than asking PPO to repair all motor channels simultaneously.

---

## References and evidence boundary

**H — User-supplied primary experimental report:** [SnowGym fighter recovery through R1j](snowgym_rl_recovery_r1j_handoff_for_gpt_pro_2026-09-04.md), dated 2026-09-04. All experiment numbers, revisions, implementation status, and current gate definitions above come from this handoff.

External methodological sources consulted:

1. Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (AISTATS 2011). https://proceedings.mlr.press/v15/ross11a.html
2. Ross, Bagnell. *Reinforcement and Imitation Learning via Interactive No-Regret Learning* (2014). https://arxiv.org/abs/1406.5979
3. Johannink et al. *Residual Reinforcement Learning for Robot Control* (2018 preprint; ICRA 2019). https://arxiv.org/abs/1812.03201
4. PyTorch official documentation. *Probability distributions — torch.distributions*. https://docs.pytorch.org/docs/stable/distributions.html
5. Raffin, Kober, Stulp. *Smooth Exploration for Robotic Reinforcement Learning*. https://arxiv.org/abs/2005.05719
6. Schulman et al. *Proximal Policy Optimization Algorithms* (2017). https://arxiv.org/abs/1707.06347

The external papers motivate methods; they do not validate the proposed SnowGym fixes. No new experiment, code change, repository write, provider call, or benchmark result was produced in this review.
