# Mathematics of SnowGym training

This document defines the equations implemented by the SnowGym neural training
package. The corresponding code is in
[`executor/model.py`](../src/snowgym_training/executor/model.py),
[`loss.py`](../src/snowgym_training/loss.py), and
[`ppo.py`](../src/snowgym_training/ppo.py).

## 1. Notation

One policy decision controls a blue squad with a fixed capacity of $U$ ally
slots. Masks distinguish populated slots from padding.

| Symbol | Meaning |
| --- | --- |
| $s_t$ | Detached environment observation at decision $t$ |
| $m_i$ | Presence mask for ally slot $i$ |
| $a_i$ | Discrete action type for ally slot $i$ |
| $x_i$ | Two-dimensional normalized move or throw target |
| $p_i$ | Throw power in $[0,1]$ |
| $A_t$ | Joint squad action $(a_i,x_i,p_i)_{i=1}^U$ |
| $r_t$ | Canonical team reward |
| $V_\phi(s_t)$ | Centralized value estimate |
| $\pi_\theta(A_t\mid s_t)$ | Squad policy |
| $\gamma$ | Discount factor |
| $\lambda$ | GAE trace parameter |

The discrete action set is

$$
\mathcal{A}=\{\text{noop},\text{move},\text{throw},\text{hold}\}.
$$

Each present unit has a legal-action mask $\ell_{i,a}\in\{0,1\}$. Absent
slots are mapped to the compatible no-op behavior.

## 2. Entity encoding

Allies, enemies, projectiles, and obstacles use separate two-layer encoders.
For entity type $k$ and row $z_{k,j}$,

$$
e_{k,j}=\operatorname{ReLU}\!\left(
W_{k,2}\operatorname{ReLU}(W_{k,1}z_{k,j}+b_{k,1})+b_{k,2}
\right).
$$

Let $q_{k,j}$ be the presence mask for row $j$. The masked mean and maximum
summaries are

$$
\bar e_k=
\frac{\sum_j q_{k,j}e_{k,j}}
     {\max\!\left(1,\sum_j q_{k,j}\right)},
\qquad
e_k^{\max}=\max_{j:q_{k,j}=1} e_{k,j}.
$$

An empty entity set receives a zero maximum. The global physical context is

$$
g_t=\operatorname{concat}\left(
\bar e_{\mathrm{ally}},e_{\mathrm{ally}}^{\max},
\bar e_{\mathrm{enemy}},e_{\mathrm{enemy}}^{\max},
\bar e_{\mathrm{projectile}},e_{\mathrm{projectile}}^{\max},
\bar e_{\mathrm{obstacle}},e_{\mathrm{obstacle}}^{\max},
c_t,\frac{\log(1+t)}{10}
\right),
$$

where $c_t$ contains normalized living-team counts. A shared actor processes
each ally embedding together with $g_t$:

$$
h_i=f_{\mathrm{actor}}\!\left(\operatorname{concat}(e_{\mathrm{ally},i},g_t)\right).
$$

Optional nearest-enemy geometry and pairwise enemy attention add ally-specific
relational features before the shared actor.

## 3. Plan conditioning

A symbolic command plan is encoded as at most $R=3$ group rows with $F=38$
features. Let $G\in\mathbb{R}^{R\times F}$ be the group tensor and
$q\in\{0,1\}^R$ its row mask. The plan embedding is

$$
e_P=f_P\!\left(\operatorname{concat}
\left(\operatorname{vec}(G\odot q),q\right)\right).
$$

The host supplies a one-hot assignment $C_i\in\{0,1\}^R$ for each living ally.
The resolved directive for unit $i$ is

$$
d_i=C_i^\top G.
$$

Architecture flags determine where $e_P$, $C_i$, and $d_i$ enter the network.
The target-only design sends plan information to a separate target actor. The
residual action design updates base action logits with

$$
z_i=z_i^{\mathrm{base}}+\Delta z_i(h_i,e_P,C_i,d_i).
$$

The final layer of $\Delta z_i$ starts with zero weights and zero bias. Initial
inference therefore matches the inherited checkpoint exactly. The
directive-expert variant has five residual functions and selects one through
the mission one-hot encoded in $d_i$:

$$
\Delta z_i=\sum_{k=1}^{5} d_{i,k}^{\mathrm{mission}}
f_k(h_i,e_P,C_i,d_i).
$$

The five routes correspond to engage, advance, hold, withdraw, and support.

## 4. Hybrid squad policy

### 4.1 Discrete action type

Illegal logits are assigned the smallest representable value before the
categorical distribution is constructed. For a legal action,

$$
\pi_i^{\mathrm{type}}(a\mid s_t)=
\frac{\ell_{i,a}\exp z_{i,a}}
     {\sum_b \ell_{i,b}\exp z_{i,b}}.
$$

### 4.2 Target distribution

The model predicts an unconstrained mean $\mu_i^x$. PPO samples

$$
u_i^x\sim\mathcal{N}(\mu_i^x,\operatorname{diag}((\sigma^x)^2)),
\qquad
x_i=\tanh(u_i^x).
$$

Action-conditioned models use separate means for movement and throwing. Target
probability contributes only when $a_i$ is move or throw. With
$u_i^x=\operatorname{atanh}(x_i)$, the change-of-variables log density is

$$
\log \pi_i^x(x_i\mid a_i,s_t)=
\sum_{d=1}^{2}\left[
\log\mathcal{N}(u_{i,d}^x;\mu_{i,d}^x,\sigma_d^x)
-\log(1-x_{i,d}^2+\varepsilon)
\right].
$$

### 4.3 Throw-power distribution

The unconstrained power sample and bounded action are

$$
u_i^p\sim\mathcal{N}(\mu_i^p,(\sigma^p)^2),
\qquad
p_i=\operatorname{sigmoid}(u_i^p).
$$

Power probability contributes only when $a_i$ is throw. Its transformed log
density is

$$
\log \pi_i^p(p_i\mid s_t)=
\log\mathcal{N}(\operatorname{logit}(p_i);\mu_i^p,\sigma^p)
-\log(p_i(1-p_i)+\varepsilon).
$$

### 4.4 Joint log probability

Define

$$
I_i^x=\mathbf{1}[a_i\in\{\text{move},\text{throw}\}],
\qquad
I_i^p=\mathbf{1}[a_i=\text{throw}].
$$

The squad log probability used by PPO is

$$
\log\pi_\theta(A_t\mid s_t)=
\sum_{i=1}^{U}m_i\left[
\log\pi_i^{\mathrm{type}}(a_i\mid s_t)
+I_i^x\log\pi_i^x(x_i\mid a_i,s_t)
+I_i^p\log\pi_i^p(p_i\mid s_t)
\right].
$$

The entropy bonus sums categorical entropy and the base Gaussian entropies
under the same presence and conditional-action masks. The implementation uses
Gaussian entropy before the tanh and sigmoid transformations.

## 5. Centralized value function

The value head pools the per-ally target-path hidden states:

$$
\bar h_t=
\frac{\sum_i m_i h_i^{\mathrm{target}}}
     {\max(1,\sum_i m_i)},
\qquad
V_\phi(s_t)=w_V^\top\bar h_t+b_V.
$$

The actor produces per-unit actions. The critic produces one scalar for the
whole blue squad.

## 6. Rewards

Canonical evaluation reward is terminal:

$$
r_t=
\begin{cases}
+1,&\text{blue wins},\\
-1,&\text{blue loses},\\
0,&\text{otherwise}.
\end{cases}
$$

Training can enable health-potential shaping. With normalized unit health
$H_i$ and masks $m_i^{B},m_j^{R}$,

$$
\Phi(s)=\sum_i m_i^{B}H_i^{B}-\sum_j m_j^{R}H_j^{R}.
$$

The shaped reward is

$$
r_t^{\Phi}=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

At a terminal transition, the implementation sets
$\Phi(s_{t+1})=0$. Run manifests retain canonical and training reward sums
separately.

## 7. Generalized advantage estimation

SnowGym tracks environment termination and time-limit truncation separately.
Define

$$
b_t=1-\mathbf{1}[\text{terminated}_t],
\qquad
c_t=1-\mathbf{1}[\text{terminated}_t\lor\text{truncated}_t].
$$

The temporal-difference residual is

$$
\delta_t=r_t+\gamma b_t V_\phi(s_{t+1})-V_\phi(s_t).
$$

Backward GAE recursion is

$$
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1},
\qquad
\hat R_t=\hat A_t+V_\phi(s_t).
$$

A truncation bootstraps its own next-state value through $b_t$ and stops the
advantage recursion through $c_t$. A true terminal transition sets both
effects to zero.

Advantages are normalized over the complete time-by-world rollout:

$$
\tilde A_t=
\frac{\hat A_t-\operatorname{mean}(\hat A)}
     {\max(10^{-8},\operatorname{std}(\hat A))}.
$$

## 8. PPO objective

For stored behavior-policy log probability $\log\pi_{\mathrm{old}}$ and current
log probability $\log\pi_\theta$, the importance ratio is

$$
\rho_t(\theta)=
\exp\left(\log\pi_\theta(A_t\mid s_t)
-\log\pi_{\mathrm{old}}(A_t\mid s_t)\right).
$$

With clip radius $\epsilon$, the implemented policy loss is

$$
L_{\mathrm{policy}}(\theta)=
-\mathbb{E}_t\left[
\min\left(
\rho_t\tilde A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\tilde A_t
\right)
\right].
$$

The value loss and mean entropy are

$$
L_V(\phi)=\frac{1}{2}\mathbb{E}_t
\left[(V_\phi(s_t)-\hat R_t)^2\right],
\qquad
\mathcal{H}=\mathbb{E}_t[H(\pi_\theta(\cdot\mid s_t))].
$$

The optimized loss is

$$
L_{\mathrm{PPO}}=
L_{\mathrm{policy}}+c_VL_V-c_H\mathcal{H}.
$$

Default values are $\gamma=0.99$, $\lambda=0.95$, $\epsilon=0.2$,
$c_V=0.5$, and $c_H=0.01$. Gradients are clipped to global norm $0.5$ by
default. Diagnostics include

$$
\widehat{D}_{\mathrm{KL}}=
\mathbb{E}_t[\log\pi_{\mathrm{old}}-\log\pi_\theta]
$$

and the fraction of samples satisfying $|\rho_t-1|>\epsilon$.

## 9. Behavior cloning

Let $y_i$ be the teacher action type. The action loss over present units is

$$
L_{\mathrm{action}}=
\frac{\sum_i m_i\omega_i[-\log\pi_i^{\mathrm{type}}(y_i\mid s)]}
     {\sum_i m_i\omega_i},
$$

where $\omega_i$ combines the configured throw-class weight and an optional
role-balancing weight.

Let $J_i^x$ select teacher move and throw actions, and let $J_i^p$ select
teacher throw actions. The masked regression losses are

$$
L_{\mathrm{target}}=
\frac{\sum_i m_iJ_i^x\omega_i^r
\lVert\hat x_i-x_i^*\rVert_2^2}
     {2\sum_i m_iJ_i^x\omega_i^r},
$$

$$
L_{\mathrm{power}}=
\frac{\sum_i m_iJ_i^p\omega_i^r
(\hat p_i-p_i^*)^2}
     {\sum_i m_iJ_i^p\omega_i^r}.
$$

$\omega_i^r$ is one without role balancing. Empty target or power selections
produce an exact finite zero. The complete supervised objective is

$$
L_{\mathrm{BC}}=
\alpha_aL_{\mathrm{action}}
+\alpha_xL_{\mathrm{target}}
+\alpha_pL_{\mathrm{power}}.
$$

## 10. Plan-counterfactual supervision

Each paired example can provide the same physical state with a primary plan
$P$, alternate plan $P'$, and corresponding host-teacher labels. The full
paired objective is

$$
L_{\mathrm{pair}}=
L_{\mathrm{BC}}(s,P,A^*)
+\beta L_{\mathrm{BC}}(s,P',A'^*)
+\eta L_{\mathrm{changed}}.
$$

Let $D$ contain present unit positions where the primary and alternate teacher
action types differ. The changed-action term is

$$
L_{\mathrm{changed}}=
\frac{1}{2}\left[
\operatorname{CE}_{D}(\pi_\theta(\cdot\mid s,P),a^*)
+\operatorname{CE}_{D}(\pi_\theta(\cdot\mid s,P'),a'^*)
\right].
$$

The configuration bounds are $\beta\in[0,10]$ and $\eta\in[0,100]$.

## 11. Mission and role balancing

Transition-uniform sampling draws from one deterministic permutation per epoch.
The seed is derived from the configured training seed and epoch index.

Mission-uniform sampling cycles through sorted plan names and samples one
transition from each selected mission pool. A deterministic random generator is
initialized from the training seed and optimizer step.

For observed role $r$ with count $n_r$, role balancing assigns

$$
w_r=
\frac{1/n_r}
     {\frac{1}{|\mathcal{R}_{\mathrm{obs}}|}
      \sum_{q\in\mathcal{R}_{\mathrm{obs}}}1/n_q}.
$$

These weights have mean one across observed roles. An absent role receives
weight zero. The per-unit value is selected through its one-hot host assignment
and applied to action, target, power, and changed-action losses.

## 12. Optimization and reproducibility

Behavior cloning uses Adam with configurable learning rate, betas
$(0.9,0.999)$, $\epsilon_{\mathrm{Adam}}=10^{-8}$, and zero weight decay.
PPO uses Adam and the hyperparameters stored in each run configuration.

CPU acceptance runs enable deterministic Torch algorithms, use one Torch
thread, and derive data order from explicit seeds. Rollout tensors are detached
and cloned before storage. Checkpoints bind the model state, optimizer state,
architecture, dataset or curriculum digest, source revision, seed, and update
index. Exact resume restores the optimizer and Torch random state.

## 13. Training paths

| Path | Data source | Objective |
| --- | --- | --- |
| Behavior cloning | Scripted teacher states | $L_{\mathrm{BC}}$ |
| DAgger | Learner-visited states labeled by the host teacher | $L_{\mathrm{BC}}$ |
| Plan ablation | Matched physical states with and without plan tensors | $L_{\mathrm{BC}}$ plus frozen counterfactual metrics |
| Counterfactual DAgger | Same physical state under two host-resolved plans | $L_{\mathrm{pair}}$ |
| Centralized PPO | Persistent vector worlds | $L_{\mathrm{PPO}}$ |

## 14. M7 plan-conditioned PPO extension

The current PPO collector stores physical observations. M7 will extend its
state input to

$$
\tilde s_t=(s_t,G_t,q_t,C_t),
$$

with fresh host-resolved plan tensors at every decision. The policy and value
equations above then use $\tilde s_t$ directly.

Mission-aware potential shaping can use a plan-dependent scalar
$\Phi_P(\tilde s_t)$ while retaining the implemented form

$$
r_t^{P}=r_t+\gamma\Phi_P(\tilde s_{t+1})-\Phi_P(\tilde s_t).
$$

Each mission potential, coefficient, seed range, and acceptance threshold must
be fixed in a versioned configuration before a qualifying run. Canonical
terminal returns remain the evaluation measure.
