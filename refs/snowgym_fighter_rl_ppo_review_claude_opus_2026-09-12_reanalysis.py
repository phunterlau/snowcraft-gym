"""Reviewer read-only reanalysis of archived R1m S5/S9/S10 artifacts. No simulator, provider, or policy update.
Run from the repository root:
  snowgym/training/.venv/bin/python refs/snowgym_fighter_rl_ppo_review_claude_opus_2026-09-12_reanalysis.py
"""
import copy, gzip, json
from pathlib import Path
import numpy as np

RUNS = Path('snowgym/training/runs')
S9 = RUNS / 'm7b_engage_r1m_s9_v0'
GAMMA = 0.9976921765
rows = lambda p: [json.loads(line) for line in gzip.open(p, 'rt')]


def ols(X, y):  # einsum normal equations (avoids spurious BLAS matmul warnings on macOS)
    b = np.linalg.solve(np.einsum('ni,nj->ij', X, X), np.einsum('ni,n->i', X, y))
    return b, np.einsum('ni,i->n', X, b)


print('A. Is the final-minus-behavior destination change one constant world vector? (S10 opportunities)')
for arm in ('short', 'full'):
    for u in (1, 11, 21):
        o = [r for r in rows(RUNS / f'm7b_engage_r1m_s10_v0/opportunities-{arm}-{u:03d}.jsonl.gz')
             if r.get('recommendationAvailable') and r['recommendationWorldGap'] > 4]
        D = np.array([r['worldDirection'] for r in o]); y = np.array([r['finalGapReduction'] for r in o])
        s = np.array([r['finalWorldShift'] for r in o])
        v, pred = ols(D, y)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f'  {arm:5s} u{u:02d} n={len(o):4d} v=({v[0]:+.2f},{v[1]:+.2f}) |v|={np.hypot(*v):.2f} '
              f'shift mean/sd={s.mean():.2f}/{s.std():.2f} R2={r2:.3f}')

print('B. Where does outcome variance come from? (S9 training episodes; S3 deterministic baselines)')
snap = json.load(open(RUNS / 'm7b_engage_r1m_s3_v0/snapshots.json'))
det = {f['seed']: f['baseline']['success'] for f in snap['datasets']['training']}
for arm in ('short', 'full'):
    eps = [(u, t['summary']['seed'], float(t['summary']['success']))
           for u in range(1, 31) for t in rows(S9 / f'99301/{arm}/events-{u:03d}.jsonl.gz')]
    by = {}
    for u, seed, ok in eps:
        by.setdefault(seed, []).append(ok)
    allv = np.array([e[2] for e in eps])
    within = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in by.values()) / len(allv) / allv.var()
    up = sum(ok and not det[s] for _, s, ok in eps); down = sum((not ok) and det[s] for _, s, ok in eps)
    fixed = sum(min(v) == max(v) for v in by.values())
    e1 = [e for e in eps if e[0] <= 10]; e3 = [e for e in eps if e[0] > 20]
    print(f'  {arm:5s} within-frame success-variance share={within:.2f}; flips vs deterministic: +{up}/-{down} of {len(eps)}; '
          f'frames never flipping={fixed}/{len(by)}; stochastic success u1-10={sum(e[2] for e in e1):.0f}/80 '
          f'(deterministic {sum(det[e[1]] for e in e1)}), u21-30={sum(e[2] for e in e3):.0f}/80 (deterministic {sum(det[e[1]] for e in e3)})')

print('C. Can 3 option-state scalars predict Monte Carlo return? (frame-grouped 5-fold CV vs archived critic)')
def value_rows(arm, target):
    X, G, V, seed, late = [], [], [], [], []
    for u in range(1, 31):
        for t in rows(S9 / f'99301/{arm}/events-{u:03d}.jsonl.gz'):
            r = np.array(t['rewards']); mc = np.zeros(len(r)); acc = 0.
            for j in range(len(r) - 1, -1, -1):
                acc = r[j] + GAMMA * acc; mc[j] = acc
            for k in range(1, t['summary']['learnedDecisions']):
                opt = t['events'][k - 1]['info']['option']  # state before action k
                phi = opt['progress'] if target == 'mc+phi' else 0.  # telescoped -Phi(s_k) is action-independent
                y = t['returns'][k] if target == 'lambda' else mc[k] + phi  # archived GAE target or Monte Carlo return
                X.append((1 - opt['decision'] / 200, 1 - opt['progress'], opt['metrics']['assignedLivingFraction']))
                G.append(y); V.append(t['values'][k] + phi)
                seed.append(t['summary']['seed']); late.append(k >= 30)
    return np.array(X), np.array(G), np.array(V), np.array(seed), np.array(late)
def design(X, poly):
    rem, h, liv = X.T; need = np.minimum(np.clip(h - .2, 0, None) / np.maximum(rem, 1e-3), 5)
    cols = [rem, h, liv] + ([need, need ** 2, rem * h, h * liv, rem * liv, rem ** 2, h ** 2, (h <= .3) * 1., (h <= .4) * rem] if poly else [])
    return np.column_stack(cols)
for arm, target in [(a, t) for a in ('short', 'full') for t in ('lambda', 'mc', 'mc+phi')]:
    X, G, V, seed, late = value_rows(arm, target)
    folds = np.array_split(np.random.default_rng(1).permutation(np.unique(seed)), 5)
    out = []
    for poly in (False, True):
        A = design(X, poly); pred = np.zeros_like(G)
        for f in folds:
            te = np.isin(seed, f); mu, sd = A[~te].mean(0), A[~te].std(0) + 1e-9
            Z = np.column_stack([np.ones(len(A)), (A - mu) / sd])
            b = np.linalg.solve(np.einsum('ni,nj->ij', Z[~te], Z[~te]) + 1e-3 * np.eye(Z.shape[1]), np.einsum('ni,n->i', Z[~te], G[~te]))
            pred[te] = np.einsum('ni,i->n', Z[te], b)
        for name, m in (('all', np.ones(len(G), bool)), ('late', late)):
            if m.sum():
                out.append(f"{'poly' if poly else 'linear'}-{name} R2={1 - np.mean((G[m] - pred[m]) ** 2) / G[m].var():.2f}")
    ev = [f"{name} {1 - np.var(G[m] - V[m]) / np.var(G[m]):+.3f}" for name, m in (('all', np.ones(len(G), bool)), ('late', late)) if m.sum()]
    label = {'lambda': 'archived GAE lambda-return target', 'mc': 'shaped Monte Carlo return',
             'mc+phi': 'Monte Carlo return + Phi(s)'}[target]
    print(f'  {arm:5s} {label:34s} rows={len(G)} archived critic EV: {", ".join(ev)} | CV ' + '; '.join(out))

print('D. One-decision, one-fighter perturbations: mission outcome flips vs Keep (S5)')
s5 = {}
for p in sorted((RUNS / 'm7b_engage_r1m_s5_v0').glob('branch-*.jsonl.gz')):
    seed, arm = p.name[len('branch-'):-len('.jsonl.gz')].split('-', 1)
    s5.setdefault(seed, {})[arm] = json.loads(gzip.open(p, 'rt').readline())
for arm in ('radial+1', 'radial-1', 'lateral+1', 'lateral-1', 'lateral+5', 'lateral-5', 'teacher'):
    flips = sum(v[arm]['final']['success'] != v['keep']['final']['success'] for v in s5.values())
    dr = np.mean([abs(v[arm]['discounted']['executor'] - v['keep']['discounted']['executor']) for v in s5.values()])
    print(f'  {arm:9s} flips={flips}/{len(s5)} mean|dReturn|={dr:.3f}')

print('E. Which parameters moved in S9? (exact initializer rebuilt and digest-checked)')
try:
    import torch
    from snowgym_training.checkpoint import semantic_state_digest
    from snowgym_training.executor.recovery_ppo import RecoveryPolicy
    from snowgym_training.options.identity import checkpoint_model
    from snowgym_training.options.movement_train import REFERENCE
    from snowgym_training.ppo_checkpoint import load_ppo_checkpoint
    meta, state = load_ppo_checkpoint(REFERENCE)
    source = checkpoint_model(meta); source.load_state_dict(state['model']); source.eval().requires_grad_(False)
    torch.manual_seed(99301); init = RecoveryPolicy(copy.deepcopy(source), standard_deviation=.02)
    expected = json.load(open(S9 / '99301/full/training.json'))['initialStateDigest']
    assert semantic_state_digest(init.state_dict()) == expected
    params = {n: p.detach() for n, p in init.named_parameters() if p.requires_grad and not n.startswith('critic.')}
    groups = ('geometry.encoders', 'geometry.move.0', 'geometry.move.2', 'option_move', 'budget_move')
    for arm in ('short', 'full'):
        final = torch.load(S9 / f'99301/{arm}/final/state.pt', map_location='cpu', weights_only=True)['model']
        steps = sum(h['optimizerSteps'] for h in json.load(open(S9 / f'99301/{arm}/training.json'))['history'])
        parts = {g: [0., 0] for g in groups}
        for n, p in params.items():
            g = next(g for g in groups if n.startswith(g)); parts[g][0] += float((final[n] - p).square().sum()); parts[g][1] += p.numel()
        total = sum(v[0] for v in parts.values()) ** .5
        print(f'  {arm:5s} L2={total:.4f} Adam steps={steps} lr*sqrt(steps)={3e-4 * steps ** .5:.4f} lr*steps={3e-4 * steps:.4f}')
        for g, (sq, n) in parts.items():
            print(f'        {g:18s} params={n:6d} L2={sq ** .5:.4f} share={sq / total ** 2:6.1%} rms/param={(sq / n) ** .5:.4f}')
except ImportError as error:
    print('  skipped (run with snowgym/training/.venv/bin/python):', error)

print('F. Power of the S9 initializer gate on both splits for a TRUE paired success gain (Monte Carlo)')
rng = np.random.default_rng(7)
def split_pass(n, p0, gain, chaos):
    base = rng.random(n) < p0; u = rng.random(n)
    new = np.where(base, u >= chaos / p0, u < (chaos + gain) / (1 - p0))
    d = new.astype(float) - base
    lower = np.quantile(d[rng.integers(0, n, (2000, n))].mean(1), .025)
    return new.mean() >= .5 and d.mean() >= .2 - 1e-12 and lower > 0
for chaos in (0., .1):
    print('  symmetric chaotic flip rate', chaos, {g: round(np.mean([split_pass(38, 13 / 38, g, chaos) and split_pass(36, 16 / 36, g, chaos)
                                                         for _ in range(2000)]), 2) for g in (.2, .3, .4)})
