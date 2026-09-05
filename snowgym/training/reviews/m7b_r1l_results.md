# R1l corrective-data factorial: negative autonomous result

Completed 2026-09-05, headless, with no provider calls or runtime assistance.
Configuration, final checkpoints, sample/label exposure, paired evaluations,
fresh holdout tensors, and digests are in
[`m7b_engage_r1l_corrective_v0`](../runs/m7b_engage_r1l_corrective_v0).
The [runner documentation](../src/snowgym_training/executor/CORRECTIVE_DATA.md)
describes reproducible entry points and immutable outputs.

## Lineage and protocol

The first preflight failed because the original root BC dataset was absent.
Its byte-identical native manifest and tensor shards were recovered from a
previous local run; the dataset digest and full ancestor chain then verified.
Both preflights are retained. Prior PPO exposure includes seeds through 101600;
the original 40-episode reservoir does not describe total training exposure.
Reserved 108000–108039, 108100–108139, and 210000–210039 had no ancestry collision.

Each arm started from the identical R1i absolute epoch-20 checkpoint, froze
inherited parameters, and ran 420 Adam steps at 3e-4 with batch 256 and clip 0.5.
R1i loss coefficients were unchanged; each component used its own label count.
Paired arms shared episode-balanced sampling streams. Mixed arms drew 128
reservoir and 128 learner states per batch. Fresh training-pool episodes were
used only for regression evaluation. Only final-step models were evaluated.

## Results

Each entry below is mean Engage progress on 40 paired environment seeds.
Every arm had **0/40 mission successes and physical wins on both sets**.

| Policy | Historical 200000–200039 | Fresh 210000–210039 |
|---|---:|---:|
| R1i reference | 0.308 | 0.326 |
| A: reservoir, teacher masks | 0.303 | 0.278 |
| B: mixed states, teacher masks | 0.312 | 0.316 |
| C: reservoir, conditional labels | 0.336 | 0.305 |
| D: mixed states, conditional labels | 0.340 | 0.326 |

All evaluated policies had zero rejected actions. Correct-plan contact rates
were 97.5–100%. HOLD-input controls had zero success; some mixed-state arms made
contact under HOLD. The classifier was frozen, but changed geometry can alter
future observations and therefore future action choices.

D minus A progress was 0.037, paired 95% interval [-0.028, 0.102], on historical
development and 0.048 [0.001, 0.097] on fresh development. D minus the immediate
reference was 0.032 [-0.033, 0.095] and approximately zero [-0.054, 0.053].
The four-arm interaction interval included zero on both sets. These are
episode-paired bootstrap intervals, not optimizer-seed variability estimates.
Only training RNG 93001 ran: D's success gain over A was zero, so the declared
93002/93003 replication was skipped. Qualification seeds remain untouched.

## Interpretation and stopping decision

R1k established useful local aim/movement substitutions and fit capacity.
R1l did not establish that correcting conditional labels and adding the fixed
learner-state set is sufficient for sustained autonomous Engage. The data do
not identify a unique remaining cause: action timing, trajectory distribution
shift, and the relationship between geometric loss and battle completion
remain plausible contributors. Improved loss or progress does not pass R1.

Stop this supervised branch at its declared budget. Do not select a different
arm or enlarge the dataset after seeing these results. R1m is a separately
declared reward-only movement mechanism test with corrected shots; any result
there remains explicitly teacher-assisted and cannot close autonomous R1.

## Verification

Targeted corrective-data tests passed, including ancestry failure/collision,
immutable outputs, paired sampling, masks, frozen source, and checkpoint tamper
rejection. Full gates passed: 325 TypeScript tests, production build, 50 Python
client tests, and 244 Python training tests. No live observation contract was
changed by this milestone. Source and artifact digests are recorded alongside
the results; the recovered dataset is preserved without regeneration.
