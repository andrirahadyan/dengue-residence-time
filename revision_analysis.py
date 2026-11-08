"""
Revision analyses for the manuscript:
  A. Global stability bound R0_hat (comparison argument) and numerical checks
  B. Endemic equilibrium: uniqueness and local stability (numerical)
  C. Intervention scenarios and scenario sweep (Table 5, Figure 5)
Reuses the model definitions of model_waktu_tinggal_en.py.
Run: python revision_analysis.py
"""
import os, json, numpy as np
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_waktu_tinggal_en.py')).read()
exec(src.split('res = {}')[0])          # model definitions only (make, R0, rhs, simulate, ...)
from scipy.optimize import fsolve

OUT2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'en'); os.makedirs(OUT2, exist_ok=True)
rev = {}

# ------------------------------------------------------------------ A. R0_hat
def Nmin(p):
    """lower bound of the effective population: p_jj N_hj + q_ij N_hi (i != j)."""
    P, Q, Nh = p['P'], p['Q'], p['Nh']
    return np.array([P[0, 0] * Nh[0] + Q[1, 0] * Nh[1], P[1, 1] * Nh[1] + Q[0, 1] * Nh[0]])

def R0_hat(p):
    K = mfac(p) * cfac(p) * np.diag(p['Nh']) @ p['P'] @ np.diag(p['Nv'] / Nmin(p) ** 2) @ p['Q'].T
    return float(np.sqrt(max(abs(np.linalg.eigvals(K)))))

def rand_par(rng):
    return make(ratio=rng.uniform(.5, 3), rv=rng.uniform(1, 3), rh=rng.uniform(1, 3), p12=rng.uniform(0, .5),
                p21=rng.uniform(0, .5), theta=rng.uniform(0, 1), b=rng.uniform(.2, .5), ph=rng.uniform(.1, .75),
                pv=rng.uniform(.1, .75), beta=rng.uniform(.1, 1/3), omega=rng.uniform(.1, .25),
                rho=rng.uniform(1/15, .2), muv=rng.uniform(.07, .2))

rng = np.random.default_rng(21)
# (i) theta = 0 -> R0_hat == R0 ; (ii) R0_hat >= R0 always
e0 = 0; viol = 0; ratios = []; gap = []
for _ in range(2000):
    p = rand_par(rng)
    rh_, r_ = R0_hat(p), R0(p)
    if rh_ < r_ - 1e-12: viol += 1
    ratios.append(rh_ / r_)
    if r_ < 1 <= rh_: gap.append(p)
    q = make(ratio=p['Nv'].sum() / NH, rv=p['Nv'][0] / p['Nv'][1], rh=p['Nh'][0] / p['Nh'][1],
             p12=p['P'][0, 1], p21=p['P'][1, 0], theta=0.0)
    e0 = max(e0, abs(R0_hat(q) - R0(q)) / R0(q))
ratios = np.array(ratios)
rev['R0hat'] = dict(max_rel_theta0=e0, violations_R0hat_lt_R0=viol, ratio_median=float(np.median(ratios)),
                    ratio_p95=float(np.percentile(ratios, 95)), ratio_max=float(ratios.max()),
                    n_samples=2000, frac_gap=len(gap) / 2000)
b0 = make(rv=2); rev['R0hat']['baseline_2_1'] = dict(R0=R0(b0), R0hat=R0_hat(b0))
b1 = make(rv=2, p12=0.05); rev['R0hat']['2_1_p005'] = dict(R0=R0(b1), R0hat=R0_hat(b1))

# (iii) numerical check of global convergence in the gap (R0 < 1 <= R0_hat), random initial conditions
def run_long(p, y0, T):
    s = solve_ivp(rhs, (0, T), y0, method='LSODA', rtol=1e-8, atol=1e-10, args=(p,))
    return s.y[:, -1]
def rand_ic(p, rng):
    y = np.zeros(14)
    for i in range(2):
        w = rng.dirichlet(np.ones(4)); y[[i, 2 + i, 4 + i, 6 + i]] = w * p['Nh'][i]
        w = rng.dirichlet(np.ones(3)); y[[8 + i, 10 + i, 12 + i]] = w * p['Nv'][i]
    return y
ok = 0; worst = 0; tested = 0
for p in gap[:40]:
    for _ in range(5):
        y0 = rand_ic(p, rng); yT = run_long(p, y0, 3 * 365)
        inf = (yT[2:6].sum() + yT[10:14].sum()) / (y0[2:6].sum() + y0[10:14].sum())
        tested += 1; worst = max(worst, inf); ok += inf < 1e-3
rev['gap_check'] = dict(param_sets=min(40, len(gap)), runs=tested, decayed_below_1e3=ok, worst_ratio=float(worst))

import time; _t=time.time(); print('A done', flush=True)
# ------------------------------------------------------------------ B. endemic equilibrium
def find_ee(p, rng, n_guess=8):
    """roots of the right-hand side from random interior guesses (scaled by total populations)."""
    sc = np.concatenate([np.repeat(p['Nh'], 4), np.repeat(p['Nv'], 3)])
    f = lambda z: rhs(0, z * sc, p) / sc
    sols = []
    for _ in range(n_guess):
        z0 = rand_ic(p, rng) / sc
        z, info, ier, _ = fsolve(f, z0, full_output=True, xtol=1e-13)
        if ier == 1 and np.abs(f(z)).max() < 1e-10 and (z > -1e-9).all() and z[[4, 5, 12, 13]].min() > 1e-9:
            sols.append(z * sc)
    return sols
def jac_at(p, x):
    J = np.zeros((14, 14))
    for i in range(14):
        h = 1e-6 * max(1.0, abs(x[i])); e = np.zeros(14); e[i] = h
        J[:, i] = (rhs(0, x + e, p) - rhs(0, x - e, p)) / (2 * h)
    return J
rng = np.random.default_rng(33); ee = []; tries = 0
while len(ee) < 200 and tries < 3000:
    tries += 1; p = rand_par(rng)
    if R0(p) <= 1.02: continue
    sols = find_ee(p, rng)
    if not sols: continue
    spread = max(np.abs(s - sols[0]).max() / np.abs(sols[0]).max() for s in sols)
    lam = max(np.linalg.eigvals(jac_at(p, sols[0])).real)
    ee.append(dict(R0=R0(p), spread=spread, nfound=len(sols), maxre=float(lam)))
rev['endemic'] = dict(n=len(ee), tries=tries, max_spread=float(max(e['spread'] for e in ee)),
                      min_found=int(min(e['nfound'] for e in ee)),
                      all_stable=all(e['maxre'] < 0 for e in ee), max_real_eig=float(max(e['maxre'] for e in ee)),
                      R0_range=[float(min(e['R0'] for e in ee)), float(max(e['R0'] for e in ee))])
# endemic level for simulation scenario
p = make(ratio=2, rv=2, p12=0.1, theta=0.5); x = find_ee(p, rng)[0]
rev['endemic']['scenario_sim'] = dict(Ih_per1000=(x[4:6] / p['Nh'] * 1000).tolist(),
                                      maxre=float(max(np.linalg.eigvals(jac_at(p, x)).real)))

print('B done', time.time()-_t, flush=True)
# ------------------------------------------------------------------ C. interventions
def attack(p, T=730):
    t, y = simulate(p, T)
    It = (y[4] + y[5]) * 1000 / p['Nh'].sum()
    att = [(y[6 + k][-1] + y[4 + k][-1] + y[2 + k][-1]) * 1000 / p['Nh'][k] for k in range(2)]
    tot = (y[2:8, -1].sum()) * 1000 / p['Nh'].sum()
    return dict(R0=R0(p), attack=float(tot), att1=float(att[0]), att2=float(att[1]),
                peak=float(It.max()), tpeak=float(t[It.argmax()]))
def scen(dense=1.0, sparse=1.0, **kw):
    q = make(ratio=2, rv=2, p12=kw.pop('p12', 0.1), theta=kw.pop('theta', 0.5))
    q['Nv'] = q['Nv'] * np.array([dense, sparse]); return q
# equal effort: remove 20% of all mosquitoes (= 30% of the dense cluster or 60% of the sparse one)
S = [('Baseline ($p=0.1$, $\\theta=0.5$)', scen()),
     ('Mobility halved ($p=0.05$)', scen(p12=0.05)),
     ('No mobility ($p=0$)', scen(p12=0.0)),
     ('Infected stay home ($\\theta=1$)', scen(theta=1.0)),
     ('Vector control, dense cluster ($-30\\%$ of $N_{v1}$)', scen(dense=0.7)),
     ('Vector control, both clusters ($-20\\%$ each)', scen(dense=0.8, sparse=0.8)),
     ('Vector control, sparse cluster ($-60\\%$ of $N_{v2}$)', scen(sparse=0.4)),
     ('Dense-cluster control and $\\theta=1$', scen(dense=0.7, theta=1.0))]
rev['interventions'] = [dict(name=n, **attack(q)) for n, q in S]
# stronger dense-cluster control needed to push R0 < 1
rev['dense_control_R0eq1'] = 1 - brentq(lambda f: R0(scen(dense=f)) - 1, 0.01, 1.0)
rev['uniform_control_R0eq1'] = 1 - brentq(lambda f: R0(scen(dense=f, sparse=f)) - 1, 0.01, 1.0)

# sweep: attack size vs p for theta = 0, 0.5, 1 (mosquitoes 2:1, Nv/Nh = 2)
pg = np.r_[0.0, 0.01, 0.02, 0.04, np.linspace(0.06, 0.5, 23)]
sweep = {}
for th in [0.0, 0.5, 1.0]:
    sweep[th] = [attack(make(ratio=2, rv=2, p12=x, theta=th)) for x in pg]
rev['sweep'] = {str(th): dict(p=pg.tolist(), attack=[s['attack'] for s in v], R0=[s['R0'] for s in v])
                for th, v in sweep.items()}
for th in [0.0, 0.5, 1.0]:
    a = np.array(rev['sweep'][str(th)]['attack']); i = a.argmax()
    rev['sweep'][str(th)].update(max_attack=float(a.max()), p_at_max=float(pg[i]))
# sweep over mosquito ratio (theta = 0.5): attack at p=0.02 vs p=1/3
rev['sweep_ratio'] = {str(r): [attack(make(ratio=2, rv=r, p12=x, theta=0.5))['attack'] for x in [0.02, 1/3]]
                      for r in [1.5, 2, 3, 5]}

# Figure 5
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
for th, c, ls in [(0.0, '#c0392b', '-'), (0.5, '#7d3c98', '--'), (1.0, '#2471a3', ':')]:
    ax[0].plot(pg, rev['sweep'][str(th)]['attack'], color=c, ls=ls, lw=2, label=r'$\theta=%g$' % th)
ax[0].set_xlabel(r'$p$ (fraction of time in the other cluster)')
ax[0].set_ylabel('infected over 730 days per 1,000 residents')
ax[0].set_title(r'(a) Epidemic size vs $p$ (mosquitoes 2:1, $N_v/N_h=2$)', fontsize=10)
ax[0].legend(frameon=False)
rows = rev['interventions']; base_att = rows[0]['attack']
lab = [r['name'].replace('$-30\\%$ of $N_{v1}$', '$-30\\%$ $N_{v1}$').replace('$-60\\%$ of $N_{v2}$', '$-60\\%$ $N_{v2}$') for r in rows][::-1]
val = [r['attack'] for r in rows][::-1]
col = ['#7f8c8d' if i == len(rows) - 1 else ('#1e8449' if 'Vector' in rows[::-1][i]['name'] or 'Dense' in rows[::-1][i]['name'] else '#2471a3')
       for i in range(len(rows))]
ax[1].barh(lab, val, color=col); ax[1].axvline(base_att, color='k', lw=.8, ls=':')
for i, (v, r) in enumerate(zip(val, rows[::-1])):
    ax[1].text(v + 8, i, r'$R_0=%.2f$' % r['R0'], va='center', fontsize=8)
ax[1].set_xlim(0, max(val) * 1.25); ax[1].set_xlabel('infected over 730 days per 1,000 residents')
ax[1].set_title('(b) Interventions (same scenario, $p=0.1$)', fontsize=10)
for a in ax:
    a.xaxis.set_major_formatter(koma); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
ax[0].yaxis.set_major_formatter(koma)
fig.tight_layout(); fig.savefig(os.path.join(OUT2, 'gambar5_interventions.png'), dpi=200); plt.close(fig)

json.dump(rev, open(os.path.join(OUT2, 'results_revision.json'), 'w'), indent=1, default=float)
print(json.dumps(rev, indent=1, default=float))
