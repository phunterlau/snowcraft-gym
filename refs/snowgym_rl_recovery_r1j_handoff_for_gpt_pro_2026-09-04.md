# SnowGym fighter recovery: review handoff through R1j

Date: 2026-09-04. Repository: `phunterlau/snowcraft-gym`.

This note supplies the context, results, mathematical contract, limitations, and proposed next experiment needed for an independent review. Repository links provide supporting evidence; the conclusions below do not require opening them.

## 1. Review baseline and current decision

The previous GPT Pro review was dated 2026-09-03 and reviewed revision `c747fce`. Its instructions are preserved in [the recovery review](snowgym_m7b_engage_review_and_recovery_instructions.md). It recommended repairing the fighter contract, qualifying fixed-plan mission execution, and postponing further commander experiments.

Current local revision is `6827894`. Local `main` is four commits ahead of the locally recorded `origin/main`, `ec1b21f`. The four unpublished commits implement and archive R1i and R1j. Remote GitHub therefore may lack the latest results until these commits are pushed. This note itself is a new working-tree document.

**M7b-R1, the first Engage bootstrap gate, remains open. No learned executor is qualified.** The production teacher succeeds on the frozen benchmark. Replacing selected learner control channels with teacher-style geometry can recover success, but supervised geometry and decoder fits have not reproduced that success reliably. PPO has been paused since R1e; R1f–R1j are diagnostic or supervised-only experiments.

The proposed next step is a learner-state supervision and gradient audit with the architecture held fixed. A corrective-data fit should follow only if that audit supports it. Data coverage is a hypothesis, not an established cause.

## 2. Research objective and execution hierarchy

The research question is whether a relatively slow LLM commander can improve tactical behavior by generating symbolic plans for a fast learned executor, including under delayed observations and delayed commands.

- The commander emits a bounded `CommandPlan`, represented by a fixed `[3,38]` symbolic tensor for main, maneuver, and reserve roles.
- The host grounds objectives, owns stable unit-to-role assignments, resolves physical targets, and handles plan lifecycle and fallback.
- The executor selects per-unit motor actions at 10 Hz from detached server state. Rendering is optional replay output and never policy input.
- Later comparisons must freeze one qualified executor and vary only the commander under matched schemas, seeds, and request opportunities.

Current recovery experiments use one main role and one fixed Engage plan. There are no online LLM calls, command switches, local-visibility restrictions, or latency injections in these experiments. Historical LLM-driven scripted-policy demos do not establish LLM-over-learned-fighter competence.

The simulator is renderer-free TypeScript, importing the original game engine. Python owns training and option evaluation. Actions are `noop`, `move`, `throw`, or `hold`, with normalized coordinate targets and throw power. `noop` preserves a previous movement order; `hold` cancels movement. Move and throw use separate target heads.

## 3. Contract repairs and a major simulator correction

M7a supplied v3 full-state observations while preserving v0–v2 compatibility: persistent movement targets, steering waypoint, aim, controller timers, projectile age, and decision timing. Unit and projectile feature widths are 21 and 9. State-hash v2 includes the added transition-relevant state; legacy hash/replay fixtures remain available. Symmetric joint stepping exists at 10 Hz.

The physical full-state Markov claim is restricted to symmetric joint-action play. Scripted opponents can have private controller state, and the option wrapper has additional history-dependent state discussed below.

The six principal defects identified by the previous review were repaired in the recovery path:

| Review finding | Implemented response | Important boundary |
|---|---|---|
| Zero-migrated v3 columns remained frozen | Trainable, initially zero-output actor extension paths | The inherited target path still detaches shared features; action-path repair alone does not give target losses access to those features |
| Option timeout bootstrapped a reset-state value | Option-horizon failure treated as terminal | External collection truncation must remain a separate concept |
| Temporal counters advanced during multiple queries | Single transition update; pure progress/success queries | Counters are still missing from policy observations |
| Terminal shaping retained a next-state potential | Absorbing terminal potential set to zero | Mission-state semantics still require further repair for other scenarios |
| Fresh critic dominated a shared gradient clip | Separate actor/critic clipping in plan PPO | Generic legacy PPO code is a separate path |
| Discount labels conflated lambda and GAE decay | Manifest records both lambda and gamma-times-lambda half-lives | Historical metadata is retained rather than rewritten |

A paired-seed audit also discovered that the semantic random Red controller generated actions which the host never applied. The host now applies them once per team decision; scripted Red retains its physics-tick path. This change is versioned as `snowgym.sim.v2`.

**Earlier results against `redController: random` used an inert opponent.** They cannot qualify a fighter against active random Red. Physical gates 1, 3, 5, 6, and 7 were reopened; scripted-Red gates 2 and 4 were unaffected by this specific defect. Old artifacts remain historical evidence, with legacy verification supported.

The random opponent also contains useful structure: it selects living enemies and aims at them. Its coordinate actions are not uniformly random motor outputs. A learned absolute-coordinate policy can therefore lose to this controller without an engine contradiction.

## 4. Current PPO and model contract

The main PPO surrogate uses per-unit likelihood ratios, a shared squad advantage, and normalization by active roster size within each decision:

$$
\rho_{ti}=\exp(\ell_{ti}^{\mathrm{new}}-\ell_{ti}^{\mathrm{old}}),\qquad
L_\pi=-\frac{1}{T}\sum_t\frac{1}{N_t}\sum_i m_{ti}
\min\left(\rho_{ti}\hat A_t,
\mathrm{clip}(\rho_{ti},1-\epsilon,1+\epsilon)\hat A_t\right).
$$

Here $m_{ti}$ is the active-unit mask and $N_t=\sum_i m_{ti}$. Joint squad ratios are retained as diagnostics and a legacy ablation. Hybrid-action likelihoods include only the selected action's meaningful continuous dimensions. Unused dimensions are tested for zero gradient.

Conditional entropy is assembled per unit:

$$
H_i=H_i^{\mathrm{type}}+\pi_i(\mathrm{move})H_i^{\mathrm{move}}
+\pi_i(\mathrm{throw})(H_i^{\mathrm{throw}}+H_i^{\mathrm{power}}).
$$

The continuous entropy terms currently use latent Normal entropy. Log likelihoods include tanh/sigmoid Jacobians, so this entropy diagnostic is not the exact entropy of executed, transformed actions.

Defaults remain $\gamma=0.9976921765$ and $\lambda=0.9885140204$. At 10 Hz, these imply approximately 30 seconds return half-life, 6 seconds lambda half-life, and 5 seconds effective GAE half-life:

$$
T_R=\frac{\log(1/2)}{10\log\gamma},\qquad
T_\lambda=\frac{\log(1/2)}{10\log\lambda},\qquad
T_{\mathrm{GAE}}=\frac{\log(1/2)}{10\log(\gamma\lambda)}.
$$

The option objective separates mission, combat, shaping, and canonical battle reward:

$$
r_t=r_t^{\mathrm{mission}}+0.1r_t^{\mathrm{combat}}
+\gamma\Phi_P(s_{t+1})-\Phi_P(s_t).
$$

Mission success gives $+1$, assigned-group elimination or horizon failure gives $-1$. Combat damage is clipped and normalized by initial roster. Terminal next potential is zero. Canonical battle results are reported separately.

The actor inherits entity encoders, an action classifier, separate move/throw heads, and power behavior from the target-only initializer. Shared plan residuals start at zero. A separate centralized critic consumes global entity pools, plan encoding, all three `[20]` physical role rows, and mission progress. Each fighter receives its owning and supported role rows.

Stage 1 freezes inherited actor parameters and trains new paths plus the critic. Stage 2 opens inherited heads at one-tenth the new-module learning rate. Final encoder unfreezing requires physical and plan gates. Current recovery has not earned those gates.

The inherited target path uses detached actor features. This matters: a trainable v3 action adapter does not automatically receive movement/throw losses. R1i added an explicitly differentiable encoder-to-target path to test that limitation.

Additional repairs before R1f persisted expanded-initializer identity, reported actual optimizer-group learning rates, and added active per-unit plan-PPO metrics and optional KL stopping. Historical runs retain their original reporting defects; they were not silently regenerated. PPO clipping alone does not bound the combined PPO, BC, and initializer-anchor update.

Exploration remains an unresolved future PPO design choice: latent target/power standard deviations are frozen in the staged optimizer. A target standard deviation of $e^{-1}$ near tanh's origin corresponds to roughly 18.4 x-world-units and 14.7 y-world-units in this arena. Deterministic evaluation omits that sampling noise.

## 5. Frozen benchmark, data, and metric meaning

- Open 100 × 80 arena, 5 blue versus 5 active random red, simulation v2 and state-hash v2.
- Fixed main-role Engage; the five initial red units form one connected selected cluster.
- Engage succeeds when the selected cluster is eliminated or reduced below 20% health within 200 decisions, or 20 seconds. The underlying battle limit is 1,800 physics ticks, or 30 seconds.
- Mean Engage progress is mean clipped selected-cluster health reduction, $\mathrm{clip}(1-h_{\mathrm{objective}},0,1)$.
- Contact and at-least-one-hit are intermediate diagnostics. Neither implies mission success. Option success also differs from winning a complete battle.
- Teacher-reservoir and probe training-evaluation seeds: 100000–100039. Prior PPO exposure is broader: R1e's final continuation records 401 distinct episode seeds through 101600, and its seed schedule spans 100000–109999. An ancestry audit is required before calling other training-pool episodes unseen. Development seeds: 200000–200039. These same 40 development seeds have been reused throughout recovery. Qualification seeds were not used by these probes.
- The later “shuffled” control is specifically a HOLD input preview, evaluated in closed loop. It is one wrong-plan control, not a distribution of randomly shuffled plans or a test of successful HOLD execution.

The successful-teacher reservoir has 5,367 states from 40 training episodes, 26,815 living-unit labels, 1,162 throw labels, and 503 hold labels. Throw labels are approximately 4.3% of living-unit labels. Teacher data supervises only the BC auxiliary objective; it never enters PPO likelihood ratios, returns, or advantages.

R1d/e use equal counts of learner-state and reservoir samples per minibatch, with loss weights 0.1 and 0.9 respectively. Describing this as “90% reservoir samples” would be inaccurate.

## 6. Recovery sequence through R1f

All entries below use the final predeclared checkpoint, rather than selecting the best intermediate result. Percentages are over 40 development seeds.

| Experiment | Change | Contact | Hit | Engage success |
|---|---|---:|---:|---:|
| R1 Stage-1 checkpoint | Successful-teacher BC reservoir | 70% | 25% | 0% |
| R1 final | Continue into Stage 2 | 0% | 0% | 0% |
| R1b | Keep Stage 1 frozen through update 100 | 0% | 0% | 0% |
| R1c | Preserve BC coefficient floor of 0.05 | 65% | 2.5% | 0% |
| R1d | Increase reservoir BC loss weight to 0.9 | 85% | 52.5% | 0% |
| R1e | Exact continuation from update 100 to 200 | 95% | 70% | 0% |
| R1f | 20 epochs supervised-only, Stage-1 actor paths | 87.5% | 52.5% | 0% |

R1c reproduced the original update-50 model, optimizer, and RNG state before changing later behavior. R1e preserved the R1d continuation state. R1e progress reached 14%; longer training recovered contact and some damage but no completion.

R1f removed PPO to test reservoir fitting with the existing trainable Stage-1 modules. It used action/target/power loss weights 1/5/0.5 and throw-class weight 5. On teacher states, move endpoint RMSE improved from 10.06 to 6.83 world units and throw endpoint RMSE from 18.71 to 9.74. However, mean throw-ray error worsened from 31.81° to 45.65°, and development progress fell from 14% to 6.6%.

Coordinate regression agreement was therefore insufficient to predict useful physical execution. This result does not establish a capacity lower bound or isolate PPO as the cause.

## 7. R1g and R1h: frozen-policy channel interventions

R1g froze the R1f epoch-20 checkpoint and changed only selected learner throw fields. Teacher-style aim chooses the nearest living enemy with deterministic tie-breaking and a 0.18-second velocity lead. Power follows the production medium-range rule. Recommendations are defined even when the teacher would not choose to throw.

| R1g intervention | Engage success | Hit | Mean progress |
|---|---:|---:|---:|
| Unchanged learner | 0/40 | 52.5% | 6.6% |
| Replace throw direction | 7/40 | 95% | 46.7% |
| Replace power only | 0/40 | 42.5% | 6.2% |
| Replace direction and power | 10/40 | 95% | 52.7% |
| Full production teacher | 40/40 | 100% | 84.9% |

Direction replacement includes enemy selection as well as aiming. Its success improvement over the learner was 17.5 percentage points, paired-bootstrap 95% interval [7.5, 30]. Adding power to direction gave another 7.5 points with interval [-5, 20], so an independent power benefit was not established.

The physics explains why ray error matters: the engine normalizes target minus shooter position into a direction. Endpoint distance does not set projectile range; power governs projectile behavior. A nearer regressed endpoint can still point in the wrong direction.

R1h then held corrected direction and power fixed and crossed teacher action choice with teacher movement recommendations:

| R1h intervention | Engage success | Mean progress |
|---|---:|---:|
| Corrected shots only | 10/40 | 52.7% |
| Plus teacher movement | 40/40 | 84.7% |
| Plus teacher action choice | 11/40 | 53.7% |
| Plus both | 40/40 | 84.9% |
| Full teacher | 40/40 | 84.9% |

The movement effect was +75 percentage points, interval [60, 87.5]. The action-choice effect was +2.5 points, interval [-12.5, 17.5]. Movement recommendations bundle desired range, formation, cohesion, and immediate dodge; these components were not separately isolated.

R1h also corrected an intervention confound: after changing action type, it reselects the corresponding learned conditional target head. An old move target cannot be interpreted as a throw target, or vice versa. Older R0 channel estimates should consequently be treated cautiously; the later controlled matrix is the stronger evidence.

The corrected-shot baseline reproduced R1g action digests and state hashes on all 40 seeds. Combined teacher controls reproduced full-teacher state trajectories. All intervention arms had zero rejected actions. These are teacher-assisted diagnostic successes, not learned checkpoint qualifications.

## 8. R1i: trainable geometry paths

R1i compared absolute and shooter-relative entity features with identical 33,669-parameter new modules. The inherited R1f actor and critic remained frozen; zero-output initialization exactly preserved the source policy. New entity encoders fed movement and throw/power residual heads directly, allowing target losses to train them.

The relative arm subtracts shooter position from selected entity positions and controller targets. Own-state and role inputs retain absolute fields; this is not a fully translation-invariant policy.

Both arms used the same absolute decoder and supervised objective:

$$
L=L_{\mathrm{move}}+L_{\mathrm{ray}}
+0.1L_{\mathrm{throw\ endpoint}}+0.5L_{\mathrm{power}},
\qquad
L_{\mathrm{ray}}=1-\frac{v\cdot v^*}{\max(\|v\|,\varepsilon)\max(\|v^*\|,\varepsilon)}.
$$

Endpoint losses use normalized coordinates; ray geometry uses world coordinates with x/y scale 50/40. Losses are masked by teacher action. Degenerate teacher rays are excluded; a zero predicted ray incurs a direction penalty. No runtime teacher targeting prior is supplied.

Disposable 32-state fitting gates passed before fresh full fits. Both full fits used 20 epochs, batch size 256, Adam learning rate $3\times10^{-4}$, gradient clip 0.5, identical initialization and shuffle, and final-epoch-only evaluation.

| R1i policy | Engage success | Contact | Hit | Progress | Teacher-state ray error |
|---|---:|---:|---:|---:|---:|
| R1f source | 0/40 | 87.5% | 52.5% | 6.6% | 45.65° |
| Absolute features | 0/40 | 100% | 100% | 30.8% | 19.73° |
| Relative features | 1/40 | 100% | 97.5% | 26.3% | 18.81° |

Absolute-minus-source progress improved by 24.2 points, exploratory paired interval [18.8, 29.7]. Relative-minus-absolute progress was -4.5 points, interval [-10.7, 1.8]. There was no clear relative-feature advantage. Improvements over the source combine additional capacity, a new gradient path, and a different loss; they cannot be assigned to one of those changes alone.

## 9. R1j: matched decoder factorial

R1j retained R1i absolute input features, parameter count, data, loss, initialization, shuffle, and fitting budget. It crossed two decoder changes:

- Movement: inherited world target plus a bounded $10\tanh(\Delta)$ correction per coordinate, clipped to the arena.
- Throw: normalize inherited shot direction plus a learned residual; retain inherited ray length, with finite degenerate handling and clipping along the ray.

The four arms were absolute, movement displacement, throw direction, and both. Zero residuals preserve the source outputs exactly. These are constrained corrections around an inherited policy, not unrestricted fresh movement or direction policies.

All four small-batch gates passed; four full fits and 400 development episodes completed.

| R1j decoder | Engage success | Progress | HOLD-input progress | Teacher-state ray error |
|---|---:|---:|---:|---:|
| Absolute control | 0/40 | 30.8% | 0% | 19.73° |
| Displacement | 0/40 | 25.3% | 3.9% | 19.72° |
| Direction | 0/40 | 25.5% | 0% | 26.51° |
| Both | 0/40 | 22.0% | 3.9% | 26.50° |

Every arm had zero full-battle wins and rejected actions. Direction decoding improved endpoint RMSE to about 6.84 world units, while worsening angular agreement relative to the absolute control. Displacement weakened separation from HOLD input.

Predeclared paired progress effects versus the absolute control were -5.5 points for displacement, interval [-11.3, 0.5], and -5.3 for direction, interval [-12.4, 1.6]. The interaction interval also included zero. No decoder was promoted. Zero observed successes in every arm does not establish population equivalence.

The R1j absolute arm exactly reproduced R1i's final model state and evaluation records. This strengthens implementation comparability. Equal parameter count and learning rate still do not imply equal physical gradient scale or conditioning across decoders.

R1i/R1j checkpoints expose deterministic inference only. They are not drop-in stochastic PPO checkpoints; direction normalization requires a deliberate probability-density contract before PPO integration.

## 10. Remaining defects, limitations, and interpretation

1. **Fixed target identity:** the general resolver can replace an eliminated enemy cluster before the option tracker measures success. Fixed Engage should retain activated target IDs and initial health. The current single-cluster benchmark does not exercise the known multi-cluster failure.
2. **Option observability:** settlement counts, occupancy history, flank timing, accumulated target damage, and remaining option time are not all actor/critic inputs. Feedforward physical-state observations alone are insufficient for history-dependent options.
3. **Role geometry:** an earlier review identified width/height versus diagonal reconstruction inconsistencies in hold/support phase summaries. These need repair and dedicated tests before those missions.
4. **Qualification lineage:** enforce seed/protocol/checkpoint identity and requalify reopened physical gates before promotion. Recent deterministic fitting probes do not close these broader requirements.
5. **Conditional labels:** teacher throw labels are sparse and movement labels arise where the teacher chooses movement. A learner can invoke a head on states poorly represented by that head's teacher-conditional labels. This mechanism is plausible but has not yet been measured adequately on learner-visited training states.
6. **Development reuse:** many experiments reuse the same 40 seeds. Paired intervals describe those comparisons; they do not remove adaptive development-set selection. Untouched qualification remains essential.
7. **Restricted scenario:** open terrain, one role, one connected target cluster, one opponent, and one wrong-plan control provide limited generalization evidence.
8. **Inherited reference:** the R1f source already has substantial geometry error. Bounded corrections and inherited shot radius may constrain recovery. The decoder results do not establish that direction-based policies generally fail.

The supported conclusion is that physical geometry and its learnable representation deserve priority. The results do not show that PPO is inherently unsuitable, that a larger model would solve the problem, or that an LLM commander causes the current failure.

## 11. Proposed next bounded milestone for review

Before further PPO, keep the R1i absolute reference architecture fixed:

1. Collect deterministic learner trajectories on training seeds only. Preserve the original teacher reservoir unchanged.
2. Audit movement, aim, and power errors by phase, distance, readiness, teacher-selected action, and learner-selected action. Separate teacher labels defined only on teacher-selected actions from independently defined conditional recommendations.
3. Measure per-head loss/gradient magnitudes and encoder reachability on both reservoir and learner states. Check scaling and saturation before attributing failure to coverage.
4. Report where the learner visits unsupported or contradictory label regions. Check whether closer teacher agreement actually improves physical error on these states.
5. If justified, predeclare one fixed-architecture corrective-data fit against an equal-budget old-data-only control. Retain a teacher-distribution holdout, HOLD-input control, exact initialization, and frozen final-checkpoint selection. Do not train on development states.
6. Reapply the existing 40-seed bootstrap gate. Preserve failures and avoid choosing a checkpoint retrospectively.

R1 requires at least 50% Engage success, at least 20 percentage points improvement over both shuffled input and initializer, contact on at least 80% of seeds, and essentially zero rejected actions. R2 tightens Engage success to 70%, with paired progress and improvement gates. Subsequent single-role missions, universal eight-mission qualification on untouched seeds, and M7c full-battle composition remain blocked.

Only after executor qualification should work return to local actors/centralized training, temporal switches and stale plans, frozen-executor commander comparisons, and latency/generalization experiments. Provider calls are unnecessary for the immediate recovery work.

### Questions for GPT Pro

1. Is conditional-label coverage the right next diagnostic after these matched negative decoder results? Which measurement would distinguish coverage from optimization or loss-conditioning failure most decisively?
2. Should movement/throw supervision be defined for every learner action opportunity, or should corrective collection retain teacher-selected-action masks? How should off-teacher-action recommendations be validated?
3. Is a fixed-architecture corrective-data experiment preferable to replacing the inherited target initializer? What evidence should trigger that architectural change?
4. How should world displacement, angular error, and power losses be scaled so that local fitting predicts closed-loop physical competence?
5. Which stochastic target distribution and decoder would provide valid PPO likelihoods while controlling high-frequency world-space exploration?
6. Which option-state and target-identity repairs must precede another Engage run, and which can safely remain gated before multi-cluster or temporal missions?
7. Are the bootstrap gate and repeatedly reused development set sufficient for selecting the next research direction, with untouched qualification reserved for promotion?

## 12. Evidence and reproducibility index

Authoritative roadmap: [snowgym/PLAN.md](../snowgym/PLAN.md). Detailed reports: [R1f](../snowgym/training/reviews/m7b_r1f_results.md), [R1h](../snowgym/training/reviews/m7b_r1h_results.md), [R1i](../snowgym/training/reviews/m7b_r1i_results.md), [R1j](../snowgym/training/reviews/m7b_r1j_results.md).

Implementation: [PPO](../snowgym/training/src/snowgym_training/ppo.py), [plan PPO](../snowgym/training/src/snowgym_training/plan_ppo.py), [option tracker](../snowgym/training/src/snowgym_training/options/tracker.py), [geometry probe design](../snowgym/training/src/snowgym_training/executor/GEOMETRY_PROBE.md), [decoder probe design and entry point](../snowgym/training/src/snowgym_training/executor/DECODER_PROBE.md).

Artifact directories under `snowgym/training/runs/`:

- `m7b_engage_teacher_reservoir_v0`: successful-teacher dataset.
- `m7b_engage_teacher_reservoir_r1e_continue200_v0`: last PPO continuation.
- `m7b_engage_r1f_supervised_probe_v0`: supervised source checkpoint and diagnostics.
- `m7b_engage_r1g_throw_channels_v0`: frozen throw interventions.
- `m7b_engage_r1h_control_channels_v0`: action/movement factorial.
- `m7b_engage_r1i_geometry_probe_v0`: matched feature fits.
- `m7b_engage_r1j_decoder_probe_v0`: four-arm decoder fits, manifests, checkpoints, paired report, and episode records.

The reservoir digest is `sha256:a1410c32a718c53664b91878852a2203247454ae0bf2dcb4caeb904b0ac334a6`. The frozen R1f epoch-20 checkpoint digest is `sha256:10d924ecdfbc554a8e0324387d8d049b9ffe719e8d8f2768123e4886c265a697`. The R1j predeclaration binds both identities, seeds, optimizer settings, four arms, and 10,000 paired-bootstrap samples.

Source milestones: `e352e0e` measurement/initializer repair; `8f7d4f9` R1f; `3dd8e8f` R1g; `bb2dec0` R1h; `160bea9` R1i; `3e499c8` R1j; `6827894` latest results/decision.

Implementation milestones used targeted tests plus the TypeScript suite, production build, Python client suite, and Python training suite. This documentation update did not rerun training or those full gates. Stored trajectory hashes and episode metrics are not uniformly browser replay JSON; consult each probe's artifact contract before attempting visual replay.

No provider calls or browser sessions were needed for R1f–R1j. No new training, provider request, commit, or push was performed to prepare this note.
