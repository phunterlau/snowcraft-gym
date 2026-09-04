# SnowGym RL and LLM commander review handoff

Prepared: 2026-09-03

Repository: `https://github.com/phunterlau/snowcraft-gym`

Reviewed source snapshot: `main` at `d3b96ad`

## Review request

Please review the architecture, mathematics, experimental design, and next
milestones for a hierarchical game-control agent:

- a deterministic 60 Hz combat simulator;
- a fast neural blue-team executor acting at 10 Hz;
- a slow event-driven LLM commander that emits a bounded symbolic plan;
- host-owned grounding, safety checks, lifecycle handling, and monitoring.

Please return:

1. mathematical or implementation inconsistencies, ranked by severity;
2. concerns about PPO credit assignment, hybrid-action likelihoods, masking,
   reward shaping, and roster scaling;
3. concerns about the symbolic commander interface and asynchronous scheduler;
4. an evaluation design that can support a credible hierarchy claim;
5. a prioritized three-milestone implementation plan;
6. experiments or ablations that should be removed, added, or reordered;
7. a recommendation for the first plan-conditioned PPO initializer and critic
   design.

Distinguish code defects, design risks, missing evidence, and optional
improvements. Treat retained failed experiments as evidence.

## 1. Project objective

SnowGym studies hierarchical control under a large cadence difference:

$$
f_{\mathrm{simulation}}=60\ \mathrm{Hz},\qquad
f_{\mathrm{executor}}=10\ \mathrm{Hz},\qquad
f_{\mathrm{commander}}\approx 0.1\text{--}0.5\ \mathrm{Hz}.
$$

The central research question is:

> Can a slow LLM improve team-level objectives through a small symbolic command
> language while a learned executor handles continuous physical control,
> latency, and local reactions?

The intended decomposition is:

$$
\text{strategic summary}
\xrightarrow{\text{LLM}}
\text{symbolic command plan}
\xrightarrow{\text{host grounding}}
\text{plan tensors}
\xrightarrow{\text{neural executor}}
\text{physical squad action}.
$$

The production commander never selects entity IDs, coordinates, movement
targets, throw targets, timing, or individual actions.

## 2. Current implementation status

| Component | Status | Evidence boundary |
| --- | --- | --- |
| Renderer-free deterministic simulation | Implemented | Same-seed action traces and public-state hashes reproduce |
| Gymnasium environments | Implemented | Fixed-capacity v0/v1/v2 environments and checker coverage |
| Configurable fights | Implemented | 1–10 units per team, open and map-backed scenarios |
| Persistent batch simulation | Implemented | Exact HTTP/batch parity; 1/8/32/64-world benchmark |
| Replay and commander overlay | Implemented | Existing Three.js UI consumes detached replay and trace files |
| Code-based blue controller | Implemented | Deterministic autonomous wins and M-vs-N examples |
| Repository-owned neural executor | Implemented | PyTorch source, checkpoints, BC, DAgger, PPO, evaluation |
| Centralized PPO curriculum | Seven gates accepted | Results are primarily BC-initialized policy retention |
| Plan-conditioned supervised executor | Offline gate accepted | Plan target sensitivity passed a frozen paired evaluation |
| Plan-conditioned closed-loop executor | Open | Hold, withdraw, maneuver, and support competence do not coexist in one checkpoint |
| Asynchronous symbolic commander | Implemented | Mock latency, fallback, reconciliation, bounded live Luna runs |
| LLM over learned executor | Open | Existing Luna battles use the code-based reactive executor |
| Latency/generalization benchmark | Foundation implemented | Full baseline matrix and frozen evaluation remain open |

The primary blocker is the plan-conditioned learned executor. Commander
infrastructure can already drive the code-based executor.

## 3. Environment and control contract

### 3.1 Scenario

A scenario defines:

- blue and red roster sizes;
- seed and maximum simulation ticks;
- decision frequency;
- arena dimensions or a bundled map;
- deterministic spawns;
- obstacles and collision geometry;
- red controller and difficulty.

Supported fixed Gym capacities are eight or ten team slots, with a fixed 3v3
compatibility environment. Smaller rosters use presence masks. Map capacity and
spawn validity are checked before an episode starts.

### 3.2 Observation

The blue observation contains fixed-shape tensors for:

- allies: 10 features per slot;
- enemies: 10 features per slot;
- projectiles: 8 features per slot;
- obstacles: 9 features per slot;
- presence masks;
- living counts for both teams;
- simulation tick;
- legal action types for every ally slot.

The state is detached from engine entities. Correctness and training use server
state, with no pixel or browser input.

### 3.3 Physical action

For each ally slot $i$, the executor emits

$$
A_i=(a_i,x_i,p_i),
$$

where

$$
a_i\in\{\mathrm{noop},\mathrm{move},\mathrm{throw},\mathrm{hold}\},
\quad
x_i\in[-1,1]^2,
\quad
p_i\in[0,1].
$$

`hold` cancels stale movement. `noop` leaves any previous movement order
unchanged. The action adapter checks masks, finite values, targets, power,
cooldown, life state, and scenario bounds.

### 3.4 Reward and episode boundary

Canonical evaluation reward is terminal:

$$
r_t=
\begin{cases}
+1,&\text{blue wins},\\
-1,&\text{blue loses},\\
0,&\text{otherwise}.
\end{cases}
$$

Termination and time-limit truncation are separate fields. Diagnostics include
survivors, health, episode length, action rejection, public-state hashes, and
replay provenance.

## 4. Fast neural executor

The project contains a custom PyTorch `EntityPolicy`. It is trained locally and
does not embed an OpenAI model.

Representative committed models:

| Checkpoint | Parameters | Interpretation |
| --- | ---: | --- |
| `runs/plan_bc_ablation_qual_v1/plan-conditioned` | 47,649 | Accepted offline plan-target model |
| `runs/plan_directive_experts_v3_dev` | 145,269 | Latest supervised mission-expert development model; failed closed-loop suite |
| `checkpoints/bc_10v10_terrain_relational_v0` | 23,495 | Successful 10v10 BC initializer |

The canonical implementation is
`snowgym/training/src/snowgym_training/executor/model.py`.

### 4.1 Entity encoder

Each entity type $k$ has a separate two-layer encoder:

$$
e_{k,j}=
\mathrm{ReLU}\!\left(
W_{k,2}\mathrm{ReLU}(W_{k,1}z_{k,j}+b_{k,1})+b_{k,2}
\right).
$$

For presence mask $q_{k,j}$, the global summary uses masked mean and maximum:

$$
\bar e_k=
\frac{\sum_j q_{k,j}e_{k,j}}
{\max\left(1,\sum_j q_{k,j}\right)},
\qquad
e_k^{\max}=\max_{j:q_{k,j}=1}e_{k,j}.
$$

The global context concatenates mean/max summaries for allies, enemies,
projectiles, and obstacles with normalized living counts and transformed tick.
A shared per-ally actor consumes the ally embedding and global context.

Optional model variants add nearest-living-enemy geometry, pairwise masked
enemy attention, separate move/throw target heads, and deterministic relational
target priors.

### 4.2 Hybrid policy distribution

For legal-action mask $\ell_{i,a}$, the categorical probability is

$$
\pi_i^{\mathrm{type}}(a\mid s)=
\frac{\ell_{i,a}\exp z_{i,a}}
{\sum_b \ell_{i,b}\exp z_{i,b}}.
$$

The normalized target uses a squashed diagonal Gaussian:

$$
u_i^x\sim\mathcal{N}\!\left(
\mu_i^x,\mathrm{diag}((\sigma^x)^2)
\right),
\qquad
x_i=\tanh(u_i^x).
$$

For $u_i^x=\mathrm{atanh}(x_i)$,

$$
\log\pi_i^x=
\sum_{d=1}^{2}
\left[
\log\mathcal{N}(u_{i,d}^x;\mu_{i,d}^x,\sigma_d^x)
-\log(1-x_{i,d}^2+\varepsilon)
\right].
$$

Throw power uses

$$
u_i^p\sim\mathcal{N}(\mu_i^p,(\sigma^p)^2),
\qquad
p_i=\mathrm{sigmoid}(u_i^p),
$$

with transformed log density

$$
\log\pi_i^p=
\log\mathcal{N}(\mathrm{logit}(p_i);\mu_i^p,\sigma^p)
-\log(p_i(1-p_i)+\varepsilon).
$$

Let $m_i$ mark a present ally,
$I_i^x=\mathbf{1}[a_i\in\{\mathrm{move},\mathrm{throw}\}]$, and
$I_i^p=\mathbf{1}[a_i=\mathrm{throw}]$. The PPO squad log probability is

$$
\log\pi_\theta(A\mid s)=
\sum_i m_i
\left[
\log\pi_i^{\mathrm{type}}
+I_i^x\log\pi_i^x
+I_i^p\log\pi_i^p
\right].
$$

The entropy bonus uses masked categorical entropy and base Gaussian entropies.
The tanh and sigmoid Jacobians are included in log probability and excluded
from the entropy approximation.

Review concern: this sum grows with living roster size. A fixed PPO clip radius
therefore represents a different per-unit policy change in 1v1 and 10v10.

### 4.3 Centralized critic

The critic pools per-ally target-path features:

$$
\bar h=
\frac{\sum_i m_i h_i^{\mathrm{target}}}
{\max(1,\sum_i m_i)},
\qquad
V_\phi(s)=w_V^\top\bar h+b_V.
$$

In a plan-conditioned target-path model, the critic receives plan-conditioned
hidden features. The actor emits per-slot actions from a shared network with
global observation context.

## 5. PPO implementation

### 5.1 Generalized advantage estimation

Define

$$
b_t=1-\mathbf{1}[\mathrm{terminated}_t],
\qquad
c_t=1-\mathbf{1}[
\mathrm{terminated}_t\lor\mathrm{truncated}_t].
$$

The implementation computes

$$
\delta_t=r_t+\gamma b_tV_\phi(s_{t+1})-V_\phi(s_t),
$$

$$
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1},
\qquad
\hat R_t=\hat A_t+V_\phi(s_t).
$$

A truncation uses its next-state value in $\delta_t$ and stops backward
recursion. A terminal transition removes both bootstrap and recursion.

Advantages are normalized across all time-by-world samples in the rollout:

$$
\tilde A_t=
\frac{\hat A_t-\mathrm{mean}(\hat A)}
{\max(10^{-8},\mathrm{std}(\hat A))}.
$$

### 5.2 Clipped objective

The joint importance ratio is

$$
\rho_t(\theta)=
\exp\left(
\log\pi_\theta(A_t\mid s_t)
-\log\pi_{\mathrm{old}}(A_t\mid s_t)
\right).
$$

The policy loss is

$$
L_{\mathrm{policy}}=
-\mathbb{E}_t
\left[
\min\left(
\rho_t\tilde A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\tilde A_t
\right)
\right].
$$

The complete loss is

$$
L_{\mathrm{PPO}}=
L_{\mathrm{policy}}
+\frac{c_V}{2}\mathbb{E}_t[
(V_\phi(s_t)-\hat R_t)^2]
-c_H\mathbb{E}_t[H_t].
$$

Defaults:

$$
\gamma=0.99,\quad
\lambda=0.95,\quad
\epsilon=0.2,\quad
c_V=0.5,\quad
c_H=0.01.
$$

The default global gradient-norm limit is $0.5$. The optimizer loop records
approximate KL, clipping fraction, entropy, losses, and gradient norms.
Minibatch order is derived from the training seed and update index.

### 5.3 Potential shaping

The implemented optional health potential is

$$
\Phi_H(s)=
\sum_i m_i^B H_i^B-\sum_j m_j^R H_j^R,
$$

$$
r_t^{\Phi}=r_t+\gamma\Phi_H(s_{t+1})-\Phi_H(s_t).
$$

At termination, $\Phi_H(s_{t+1})=0$. Run manifests store shaped and
canonical reward sums separately.

Mission-aware plan potentials have not been implemented.

## 6. Supervised and plan-conditioned training

### 6.1 Behavior cloning

For teacher action $y_i$, class/role weight $\omega_i$, and presence mask
$m_i$,

$$
L_{\mathrm{action}}=
\frac{
\sum_i m_i\omega_i[-\log\pi_i^{\mathrm{type}}(y_i\mid s)]
}{
\sum_i m_i\omega_i
}.
$$

Target loss applies to move and throw labels. Power loss applies to throw
labels. With the respective masks $J_i^x,J_i^p$,

$$
L_{\mathrm{target}}=
\frac{
\sum_i m_iJ_i^x\omega_i^r
\lVert\hat x_i-x_i^*\rVert_2^2
}{
2\sum_i m_iJ_i^x\omega_i^r
},
$$

$$
L_{\mathrm{power}}=
\frac{
\sum_i m_iJ_i^p\omega_i^r
(\hat p_i-p_i^*)^2
}{
\sum_i m_iJ_i^p\omega_i^r
}.
$$

The complete objective is

$$
L_{\mathrm{BC}}=
\alpha_aL_{\mathrm{action}}
+\alpha_xL_{\mathrm{target}}
+\alpha_pL_{\mathrm{power}}.
$$

### 6.2 Plan tensor

The host converts a validated, grounded command plan into at most three group
rows:

$$
G\in\mathbb{R}^{3\times38},
\qquad
q\in\{0,1\}^{3}.
$$

The plan encoder receives

$$
e_P=f_P\left(
\mathrm{concat}(\mathrm{vec}(G\odot q),q)
\right).
$$

The host also provides a one-hot role assignment $C_i\in\{0,1\}^3$ for each
living unit. The unit's resolved directive is

$$
d_i=C_i^\top G.
$$

Plan features include role, mission, approach, posture, fire policy, preferred
range, cohesion, objective type and relative geometry, assigned fraction,
living fraction, support relation, and normalized plan age.

### 6.3 Plan adapters

The accepted offline checkpoint uses a separate target actor. Plan features
affect targets and power while the physical action classifier remains
plan-invariant.

Later development variants add a zero-initialized action residual:

$$
z_i=
z_i^{\mathrm{base}}
+\Delta z_i(h_i,e_P,C_i,d_i).
$$

The directive-expert model selects one of five residual functions through the
unit mission:

$$
\Delta z_i=
\sum_{k=1}^{5}
d_{i,k}^{\mathrm{mission}}
f_k(h_i,e_P,C_i,d_i).
$$

The routes correspond to engage, advance, hold, withdraw, and support.

### 6.4 Same-state counterfactual supervision

The host can preview an alternate symbolic plan at the current physical state.
It returns the alternate grounded tensor and teacher action without advancing
the simulator or replacing the active plan.

For primary plan $P$, alternate plan $P^{\prime}$, and paired teacher
actions $A^*,A^{\prime *}$,

$$
L_{\mathrm{pair}}=
L_{\mathrm{BC}}(s,P,A^*)
+\beta L_{\mathrm{BC}}(s,P^{\prime},A^{\prime *})
+\eta L_{\mathrm{changed}}.
$$

Let $D$ contain unit-state pairs where the two teacher action types differ:

$$
L_{\mathrm{changed}}=
\frac{1}{2}
\left[
\mathrm{CE}_{D}(\pi_\theta(\cdot\mid s,P),a^*)
+\mathrm{CE}_{D}(
\pi_\theta(\cdot\mid s,P^{\prime}),a^{\prime *})
\right].
$$

Mission-uniform sampling and inverse-frequency role weights address dataset
imbalance. The current multi-group corpus contains main, maneuver, and reserve
assignments.

## 7. Current learning evidence

### 7.1 Centralized physical PPO

Seven curriculum gates have accepted checkpoint-series artifacts:

1. 1v1 random;
2. 1v1 easy scripted;
3. 3v3 random;
4. 3v3 scripted;
5. 3v3 terrain;
6. 5v5 terrain;
7. 10v10 terrain.

Every series retains all predeclared updates and evaluates disjoint seeds
against masked-random and scripted baselines. The later results mainly preserve
strong BC or DAgger initializers under PPO. Several configurations use learning
rate $10^{-8}$. These establish pipeline stability and closed-loop
competence. They provide limited evidence for reward-driven improvement.

### 7.2 Offline plan qualification

The frozen target-only qualification passed all predeclared checks on new
seeds:

- conditioned target MSE: $0.044480$;
- no-plan target MSE: $0.264987$;
- MSE increase after plan swapping: $0.353375$;
- mean predicted target change: $0.372476$;
- identical action accuracy for conditioned and no-plan models:
  $0.963021$;
- zero no-plan counterfactual sensitivity.

This supports plan-dependent destination prediction under matched physical
states.

### 7.3 Closed-loop plan failures

The qualified target-only model suppresses movement for defensive plans. It
does not separate hold, withdraw, and support reliably.

Successive development approaches include:

- target-only learner-state correction;
- residual action adapter;
- paired same-state counterfactual labels;
- changed-action weighting;
- role conditioning;
- per-unit full directive conditioning;
- mission-uniform sampling and role-balanced losses;
- explicit main/maneuver/reserve data;
- five directive-specific residual experts.

No checkpoint passes the joint direct, flank, hold, withdraw, and support
closed-loop criteria. Selected findings:

- one residual-adapter checkpoint produced a 6–0 blue support win and strong
  flank survival, while direct and withdraw checks failed;
- higher changed-action weighting improved paired-label recall and regressed
  several closed-loop missions;
- mission and role balancing improved direct and withdraw behavior while flank
  and support regressed;
- the latest directive-expert checkpoint improved offline accuracy and still
  failed the closed-loop suite.

All failed runs remain versioned. The evaluation split for the latest
multi-group corpus remains sealed.

### 7.4 Inference from the failures

Supervised action imitation has reached a mission tradeoff across several
architectures and sampling strategies. The planned next method is
plan-conditioned closed-loop RL fine-tuning from retained checkpoints.

## 8. Symbolic LLM commander

### 8.1 Command language

`snowgym.command-plan.v0` contains one to three groups. Roles are:

~~~text
main | maneuver | reserve
~~~

Each group specifies:

- integer allocation weight from 1 to 10;
- deterministic selection rule;
- mission and compatible symbolic objective;
- approach;
- engagement policy.

Bounded values:

~~~text
selection:
  balanced | frontline | rearline | healthiest |
  nearest_objective | nearest_left_lane | nearest_right_lane

mission:
  engage | advance | hold | withdraw | support

enemy cluster:
  nearest | largest | weakest | leftmost | rightmost

region:
  left_lane | center_lane | right_lane |
  own_backfield | enemy_backfield

approach:
  direct | left_flank | right_flank | avoid_center

posture:
  aggressive | balanced | conservative

fire:
  focus | distributed | opportunistic

preferred range:
  close | medium | long

cohesion:
  tight | normal | loose
~~~

Mission/objective constraints:

- `engage` targets an enemy cluster;
- `advance` targets a region;
- `hold` targets a region or current position and uses direct approach;
- `withdraw` targets own backfield and uses direct or avoid-center approach;
- `support` targets another declared ally-group role;
- support relations must be acyclic.

Example model output:

~~~json
{
  "schemaVersion": "snowgym.command-plan.v0",
  "intentSummary": "Pin the largest cluster while preserving a reserve.",
  "groups": [
    {
      "role": "main",
      "allocationWeight": 7,
      "selection": "balanced",
      "order": {
        "mission": "engage",
        "objective": {
          "kind": "enemy_cluster",
          "select": "largest"
        },
        "approach": "direct",
        "engagement": {
          "posture": "balanced",
          "fire": "focus",
          "preferredRange": "medium",
          "cohesion": "normal"
        }
      }
    },
    {
      "role": "reserve",
      "allocationWeight": 3,
      "selection": "rearline",
      "order": {
        "mission": "support",
        "objective": {
          "kind": "ally_group",
          "role": "main"
        },
        "approach": "direct",
        "engagement": {
          "posture": "conservative",
          "fire": "opportunistic",
          "preferredRange": "long",
          "cohesion": "normal"
        }
      }
    }
  ]
}
~~~

`intentSummary` is trace-only.

### 8.2 Host-owned grounding

The host:

1. validates the exact schema and mission compatibility;
2. converts allocation weights to deterministic group sizes using Hamilton
   apportionment;
3. assigns each living blue unit exactly once;
4. resolves enemy clusters and team-relative regions from current state;
5. adds plan ID, source tick, state hash, and provenance;
6. freezes the grounded plan in an atomic `PlanStore`.

Objectives are re-resolved from the latest observation while group membership
remains stable for the active plan.

### 8.3 Strategic input to the LLM

The LLM receives an ID-free payload:

- request ID and lifecycle triggers;
- source tick and public-state hash;
- arena dimensions and obstacle count;
- blue/red alive count, health fraction, centroid, and spread;
- hostile projectile count;
- group role, mission, assigned count, living count, and objective kind;
- current symbolic plan;
- optional bounded group trajectory digest;
- optional preceding-plan outcome.

The trajectory digest includes group-level objective progress, health trend,
cohesion trend, action counts, rejection counts, and host-computed stuck
fraction. Raw unit trajectories and engine IDs stay inside the host.

### 8.4 Provider adapter

The current adapter uses:

~~~text
model: gpt-5.6-luna
API: OpenAI Responses API
structured output: strict JSON Schema
reasoning effort: configurable, medium by default
store: false
credentials: OPENAI_API_KEY from the server environment
~~~

Provider output passes through the normal validator and reconciler. Provider
metadata records latency, input/output/reasoning tokens, response ID, and
request ID when supplied.

### 8.5 Asynchronous scheduler

The physical controller runs synchronously. After each environment step, the
scheduler performs a non-blocking poll:

~~~text
observe
  -> execute current plan
  -> step simulator
  -> update trajectory digest
  -> detect lifecycle signals
  -> poll/start commander request
  -> continue current or fallback plan
~~~

Scheduler properties:

- one provider request in flight;
- deterministic trigger ordering;
- coalescing while busy or cooling down;
- configurable request interval;
- simulation-tick timeout;
- optional minimum simulated response latency for reproducible tests;
- per-episode provider-attempt limit;
- abortion on episode close;
- ignored late or superseded responses.

Hard lifecycle triggers are plan expiry, major own-force loss, assigned-group
elimination, and objective completion. They install a deterministic fallback
immediately and schedule replanning.

Soft trajectory triggers are plan stall and repeated action rejection. They
retain the active plan while replanning.

When a candidate arrives, reconciliation uses the newest living roster and
observation. Bounded repairs can remove infeasible optional groups, repair a
support target, or replace a defeated enemy objective. Rejection preserves the
active plan or fallback.

## 9. Existing commander evidence

### 9.1 Deterministic mock commander

Mock commander tests cover:

- simulated latency;
- timeout;
- trigger coalescing;
- late response;
- invalid response;
- provider failure;
- stale-state reconciliation;
- exact replay of actions, state hashes, plans, signals, and scheduler trace.

A configurable 6-blue versus 10-red seed-14 mock run ends in a 1–0 blue win
with zero rejected physical actions. This is a single deterministic example.

### 9.2 Live GPT-5.6 Luna

Two bounded headless acceptances are retained:

1. A single-request 10v10 battle continued at 10 Hz during 3.51 seconds of
   inference, activated the reconciled plan at tick 204, and ended 9–0 for blue.
2. A trajectory-aware 10v10 battle detected plan stalls, used two of three
   permitted requests, accepted a split-force plan after 7.13 seconds, and
   ended 4–0 for blue with zero rejected actions.

Both live runs used the code-based reactive executor. They validate provider
integration, latency isolation, and plan activation. They do not establish an
LLM advantage over commander baselines.

## 10. Proposed M7 plan-conditioned PPO

### 10.1 Missing bridge

The centralized PPO collector currently resets physical scenarios and stores
physical observations. The batch API already supports plan activation,
authoritative plan tensors, per-unit assignments, and read-only plan-teacher
actions.

The new collector should use

$$
\tilde s_t=(s_t,G_t,q_t,C_t),
$$

where plan tensors and assignments are refreshed before each policy decision.

Required behavior:

1. select a frozen plan template for each batch world;
2. activate it after reset;
3. retrieve fresh host-resolved plan observations;
4. sample and execute the neural joint action;
5. store physical and plan fields in the rollout;
6. re-activate the correct plan after selective world reset;
7. restore plan schedule, seed cursor, optimizer, and random state on resume;
8. record plan and reward provenance in every checkpoint.

### 10.2 Candidate initializers

| Initializer | Advantage | Risk |
| --- | --- | --- |
| Accepted target-only qualification checkpoint | Clean frozen provenance and strong target sensitivity | Action type is plan-invariant at initialization |
| Residual-adapter v1 checkpoint | Partial action sensitivity, 6–0 support result, strong flank survival | Failed direct, withdraw, and paired-action thresholds |
| Directive-expert v3 checkpoint | Separate mission residual paths and full role coverage | Latest closed-loop suite failed |
| 10v10 relational physical checkpoint plus zero-init plan paths | Strong physical targeting behavior | Requires a new matched plan-conditioned initialization contract |

One reasonable first comparison is:

- control: accepted target-only checkpoint plus zero-initialized action adapter;
- treatment: directive-expert v3 checkpoint;
- shared plan curriculum, rollout budget, critic, rewards, and evaluation suite.

### 10.3 Mission potential

A generic host-computed plan potential can be written as

$$
\Phi_P(\tilde s)=
\sum_{r\in\{\mathrm{main},\mathrm{maneuver},\mathrm{reserve}\}}
q_r
\left[
w_{r,d}f_{r,\mathrm{distance}}
+w_{r,h}f_{r,\mathrm{health}}
+w_{r,c}f_{r,\mathrm{cohesion}}
+w_{r,e}f_{r,\mathrm{engagement}}
\right].
$$

The shaped reward is

$$
r_t^P=
r_t+\gamma\Phi_P(\tilde s_{t+1})-\Phi_P(\tilde s_t).
$$

The feature definitions need mission-specific signs and anchors:

- engage: enemy health reduction and range control;
- advance: progress toward resolved region;
- hold: bounded displacement from activation anchor and survival;
- withdraw: progress toward own backfield and survival;
- support: distance and combat relation to the supported group;
- flank: lateral approach progress and final engagement geometry.

Potential-based policy invariance requires the augmented state to contain the
active plan, activation anchors, and any variables used by the potential.
Plan changes during an episode require explicit boundary handling.

### 10.4 Evaluation proposal

Development and qualification should use disjoint environment seeds and plan
seeds. Retain every predeclared checkpoint.

For each case, compare:

1. no-plan neural executor;
2. accepted plan-conditioned BC model;
3. plan-conditioned PPO;
4. production code-based plan-aware executor;
5. masked-random baseline where informative.

Cases:

- direct versus left/right flank from the same state;
- focus versus distributed fire;
- hold versus advance;
- hold versus withdraw;
- main plus reserve support;
- main plus maneuver flank;
- main plus maneuver plus reserve;
- unseen valid directive combinations;
- 3v3, 5v5, and 10v10 transfer;
- sealed generated maps.

Metrics:

- canonical win/draw/loss rate;
- survivor and health distributions;
- episode length;
- per-role objective progress;
- hold displacement;
- withdraw backfield progress;
- support distance and supported-group combat effect;
- flank geometry;
- focused-fire concentration;
- action rejection;
- counterfactual action and target sensitivity;
- seed-paired difference from each baseline;
- confidence intervals across frozen seeds.

The qualification gate should require competence across mission families. An
aggregate mean can hide a mission regression.

## 11. Proposed M9 commander comparison

Freeze one learned executor before comparing commanders. Every commander
should emit the same `CommandPlan` schema.

Suggested arms:

1. fixed fallback plan;
2. random valid plan;
3. hand-written rule commander;
4. high-level RL commander;
5. online GPT-5.6 Luna;
6. static Luna-generated doctrine;
7. distilled commander.

Use paired scenario seeds and fixed request triggers. Measure:

- canonical outcome;
- objective completion;
- plan validity;
- reconciliation and repair rate;
- source-state age at activation;
- plan churn;
- token use and provider latency;
- executor action rejection;
- trajectory quality;
- cost per completed episode.

Run deterministic simulated-latency sweeps before live provider comparisons:

$$
0,\ 100,\ 250,\ 500\ \mathrm{ms},
\quad
1,\ 2,\ 4,\ 8\ \mathrm{s}.
$$

The production symbolic pathway can be compared with an experimental
exact-assignment control to test whether late-bound group plans degrade more
gracefully under latency.

## 12. Main review concerns

### Policy and optimization

1. The joint log probability sums over living units, so effective PPO clipping
   changes with roster size.
2. Unit actions are conditionally factorized after shared global context;
   coordinated multimodal actions may be difficult to represent.
3. The entropy bonus uses base Gaussian entropy for squashed actions.
4. The centralized critic pools target-path features and may need explicit
   plan progress, role structure, or a separate encoder.
5. Current accepted PPO gates emphasize retention from BC initializers.
6. Health shaping supplies combat feedback and no direct mission credit.

### Plan learning

7. Supervised labels come from the code-based plan-aware teacher, which can
   transfer its biases into the learned executor.
8. Plan target prediction is strong offline and mission-dependent action timing
   remains weak.
9. Mission labels, unit roles, and physical states are correlated; observed
   counterfactual pairs cover only a bounded slice of the state distribution.
10. Support quality requires relational multi-group metrics that are difficult
    to capture with one scalar potential.

### Commander evaluation

11. Existing live LLM successes are integration acceptances with no paired
    commander baseline.
12. Strategic summaries omit detailed terrain topology and uncertainty.
13. Late binding combines commander quality with host grounder quality.
14. Lifecycle triggers and fallback can dominate the measured effect of the
    commander.
15. Token, latency, and plan-repair costs require comparison at equal request
    opportunities.

### Generalization and evidence

16. Most held-out PPO gates use eight seeds; statistical power is limited.
17. Generated maps support stress testing and have not yet produced sealed
    policy-generalization evidence.
18. Several strong results are possibility proofs from one seed.
19. Cold-start PPO and material improvement over BC remain unproven.
20. The repository has no top-level license file, which affects external reuse
    of code and weights.

## 13. Questions for the reviewer

1. Should the PPO ratio use the summed squad log probability, mean active-unit
   log probability, or a structured per-unit clipping objective?
2. Does the hybrid target/power factorization correctly handle unused action
   dimensions for PPO?
3. Should entropy use transformed-distribution estimates?
4. Which critic input best supports plan-conditioned credit assignment:
   pooled executor features, explicit group-role pooling, or a graph/attention
   critic?
5. Which initializer in Section 10.2 provides the cleanest first experiment?
6. How should mission potentials be defined to preserve Markov state and avoid
   reward conflict across roles?
7. Should plan templates remain fixed for a PPO episode before introducing
   lifecycle replanning?
8. Is the `[3,38]` plan representation sufficient for 10v10 transfer?
9. Which closed-loop metrics provide strong evidence for hold, withdraw,
   flank, support, and fire-allocation obedience?
10. How many seeds and maps are needed for a defensible commander comparison?
11. Which baselines are essential for attributing improvement to online LLM
    reasoning?
12. Should the strategic summary expose a compact terrain graph or learned map
    embedding while preserving the symbolic boundary?
13. How should plan age and source-state staleness enter the learned executor
    and critic?
14. What failure thresholds should block an online LLM evaluation?
15. Which three implementation milestones should be completed next?

## 14. Reproduction commands

Install training dependencies:

~~~bash
cd snowgym/training
uv sync --extra dev --extra learn
~~~

Audit a centralized PPO series:

~~~bash
.venv/bin/snowgym-audit-ppo-series \
  runs/ppo_10v10_terrain_relational_bc_v0 \
  --json
~~~

Run the current M7 closed-loop behavior suite:

~~~bash
.venv/bin/snowgym-evaluate-plan-closed-loop \
  --ablation runs/plan_bc_ablation_qual_v1 \
  --conditioned-checkpoint runs/plan_directive_experts_v3_dev \
  --suite src/snowgym_training/configs/plan_closed_loop_behaviors_v1.json \
  --output /tmp/snowgym-m7-review.json \
  --json
~~~

Run deterministic commander latency tests:

~~~bash
cd ../..
npm run snowgym:commander:benchmark -- \
  --seeds 11,12,13,14,15 \
  --latency-ticks 0,6,15,30,60,120,240,480 \
  --blue-units 10 \
  --red-units 10 \
  --map arena6.json \
  --output /tmp/snowgym-commander-latency.json \
  --json
~~~

Run the full acceptance gate:

~~~bash
npm test
npm run build
cd snowgym/python
.venv/bin/python -m pytest -q
cd ../training
.venv/bin/python -m pytest -q
~~~

Live Luna commands require explicit authorization and `OPENAI_API_KEY`.
Deterministic mock tests and all model audits require no provider request.

## 15. Relevant source map

| Area | Repository path |
| --- | --- |
| Authoritative headless simulator | `snowgym/core/SnowEnvironment.ts` |
| JSON service contract | `snowgym/server/SnowGymService.ts` |
| Python Gym adapter | `snowgym/python/src/snowgym_client/` |
| Persistent batch host/client | `snowgym/batch/`, `snowgym/python/src/snowgym_client/batch.py` |
| Neural executor | `snowgym/training/src/snowgym_training/executor/` |
| Training mathematics | `snowgym/training/math/README.md` |
| PPO implementation | `snowgym/training/src/snowgym_training/ppo.py` |
| PPO collector | `snowgym/training/src/snowgym_training/ppo_collect.py` |
| BC and plan losses | `snowgym/training/src/snowgym_training/loss.py`, `trainer.py` |
| Command schema | `snowgym/orchestration/command/` |
| Host grounding | `snowgym/orchestration/grounding/` |
| Lifecycle and reconciliation | `snowgym/orchestration/lifecycle/` |
| Async scheduler | `snowgym/orchestration/scheduler/CommanderScheduler.ts` |
| Trajectory monitoring | `snowgym/orchestration/trajectory/` |
| OpenAI adapter | `snowgym/orchestration/providers/OpenAICommanderClient.ts` |
| Commander trace/replay overlay | `snowgym/orchestration/trace/`, `snowgym/replay/` |
| Current roadmap and evidence log | `snowgym/PLAN.md` |

## 16. Claim boundary

Supported claims:

- SnowGym provides deterministic, renderer-free, configurable squad combat with
  fixed-shape Gym and persistent batch interfaces.
- A repository-owned neural policy controls blue teams in closed loop.
- BC-initialized PPO checkpoints pass the frozen physical-control curriculum.
- Plan tensors influence offline target prediction under same-state
  counterfactual evaluation.
- A slow Luna commander can return valid symbolic plans without blocking the
  code-based 10 Hz executor.
- Trajectory signals, fallback, stale-state reconciliation, request limits,
  provider metadata, and replay-aligned traces are implemented.

Open claims:

- reward-driven PPO materially improves over the strongest BC initializer;
- one learned executor reliably follows all command missions in closed loop;
- online LLM command improves outcomes over rule, random, static, RL, or
  distilled commanders;
- the hierarchy generalizes across roster sizes and sealed generated maps;
- the measured LLM benefit justifies latency, tokens, and cost.
