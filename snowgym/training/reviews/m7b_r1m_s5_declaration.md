# R1m-S5: destination-to-motion boundary probe

Declare and commit before experimental measurement. Headless, provider-independent,
no training, decoder changes or checkpoint promotion. All episodes retain R1h
corrected shots and are autonomous-qualification ineligible.

Use the original R1h source checkpoint and digest-verified S3 training first-hit
snapshot inventory (57 snapshots in 100000–100063). Sort seeds, restore prefixes,
and take the first 24 eligible snapshots. Eligibility requires a living fighter
whose frozen classifier selects MOVE, a valid conditional movement recommendation,
and a nondegenerate ray to its engine-clipped original destination. Choose the
smallest eligible unit ID. Record exclusions; do not replace from other seed
blocks. Fewer than 24 fails preflight. Selection uses no branch outcomes.

For each snapshot run ten arms: keep; radial +/-1; lateral +/-1; radial +/-5;
lateral +/-5 world units; and the conditional teacher destination. Axes are fixed
from the selected fighter's position to the clipped baseline destination at the
branch start. The larger amplitude is a predeclared fivefold contrast, not a
Gaussian standard deviation or tuned optimum. Clip destinations to the existing
arena margin and record requested, serialized and effective displacement.

Only the selected fighter's first MOVE destination changes. All other first-step
action fields remain identical. Subsequently recompute the unchanged deterministic
source policy with corrected shots until the original 200-decision option ends.
Persistent orders retain their existing semantics; no forced holds, extra command
commitment or budget extension is introduced. Equal prefixes do not imply equal
opponent actions after trajectories diverge.

Each branch is independently reconstructed and executed twice; store one full
trace and both semantic digests. Exact duplicates are mandatory and are not extra
statistical samples. Keep must match every archived baseline suffix hash and its
action digest. Verify physical, plan, option and observation identity before each
branch. Verify source weights and all S3 source files/digests before and after.
Maximum simulator budget: 57*200 selection-prefix decisions plus
24*10*2*200 prefix-and-continuation decisions = 107,400. No new seeds are collected.

## Measurements fixed before execution

- Requested and effective target change; arena clipping; prior movement target;
  controller state, throw readiness, action mask and recommendation availability.
- Analytic unobstructed desired-velocity change using speed 6 and slow radius 2.1;
  distinguish this approximation from simulated motion including separation,
  acceleration, collisions and interruptions. Fail source preflight on engine drift.
- Selected-unit position, velocity, health, life state and nearest-enemy range
  after decisions 1, 5 and 30 when reached; earlier termination is explicitly
  censored for later fixed-time measures. Report displacement relative to Keep,
  velocity difference and displacement/effective-target-change gain.
- Team damage dealt/received, mission progress, success and remaining budget;
  full discounted return with separate mission, combat, shaping and canonical
  components; action rejection counts. Keep reward definitions unchanged.
- For signed radial/lateral arms, report paired seed-level differences and
  bootstrap 95% intervals (10,000 resamples, RNG 980001). Intervals are descriptive
  and unadjusted for multiple comparisons. No best-arm selection or promotion.
- For each scale, average physical displacement over its four signed directions
  within a seed before comparing scales. Report the fraction of seeds with any
  return change above 1e-6 and signed mean return change. Larger variance of return
  or an oracle-best direction alone is not evidence that random exploration helps.
- Report the 1-step near-zero motion fraction (position difference <=1e-6),
  including the unclipped radial subset whose start/end rays remain outside the
  slow radius. No causal reward claim is inferred from teacher-target distance.

Stop after this diagnostic. Useful reward variation would motivate a separately
declared exploration intervention; low physical sensitivity would motivate a
separate interface test. Negative or mixed outcomes remain archived. A one-decision
probe does not settle sustained-action exploration or the S2 teacher bundle.

Before implementation and results commits run targeted geometry, mask, prefix,
single-fighter substitution, repeatability, censoring, seed and tamper tests plus
`npm test`, `npm run build`, client Python tests and training Python tests.
