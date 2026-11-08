"""
Illustrative calibration of the two-cluster residence-time model to the 2024 dengue
outbreak in DKI Jakarta (cluster 1) and Banten (cluster 2).
Data: OpenDengue V1.3, monthly Admin1 counts (data/opendengue_v1_3_dki_banten_monthly.csv).
Populations: Dukcapil 2024 (DKI 11.03 million, Banten 12.88 million).
Residence fractions: rough estimate from BPS Jabodetabek Commuter Survey 2023 (see manuscript).
Run: python calibration.py
"""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'model_waktu_tinggal_en.py')).read().split('res = {}')[0])
from scipy.optimize import least_squares

OUT3 = os.path.join(HERE, 'en'); os.makedirs(OUT3, exist_ok=True)
D = pd.read_csv(os.path.join(HERE, 'data', 'opendengue_v1_3_dki_banten_monthly.csv'), parse_dates=['calendar_start_date'])
D = D.set_index('calendar_start_date')[['DKI JAKARTA', 'BANTEN']]
FIT = D.loc['2023-11-01':'2024-07-01']                       # 9 months
BG = D.loc['2023-08-01':'2023-10-01'].mean().values          # inter-epidemic background (Aug-Oct 2023)
NH2 = np.array([11.03e6, 12.88e6])
edges = np.r_[0, np.cumsum(pd.to_datetime(FIT.index).days_in_month.values)].astype(float)   # month boundaries (days)
obs = FIT.values                                               # shape (9, 2)

def params(m1, m2, p12, p21, theta=0.5):
    p = dict(LIT); p['Nh'] = NH2.copy(); p['Nv'] = np.array([m1, m2]) * NH2
    p['P'] = np.array([[1 - p12, p12], [p21, 1 - p21]]); p['theta'] = theta
    q12, q21 = (1 - theta) * p12, (1 - theta) * p21
    p['Q'] = np.array([[1 - q12, q12], [q21, 1 - q21]]); return p

def rhs_c(t, y, p):                      # model + cumulative onsets (beta * E_h) per cluster
    return np.concatenate([rhs(t, y[:14], p), p['beta'] * y[2:4]])

def monthly(p, seeds, rep):
    y0 = np.concatenate([dfe(p), [0, 0]])
    for i in range(2):                    # seeds split between E and I
        y0[i] -= seeds[i]; y0[2 + i] += seeds[i] / 2; y0[4 + i] += seeds[i] / 2
    s = solve_ivp(rhs_c, (0, edges[-1]), y0, t_eval=edges, method='LSODA', rtol=1e-7, atol=1e-6, args=(p,))
    return BG + rep * np.diff(s.y[14:16], axis=1).T            # (9, 2) predicted reported cases

def unpack(z):
    return np.exp(z[0]), np.exp(z[1]), 1 / (1 + np.exp(-z[2])), np.exp(z[3:5])
def resid(z, P12, P21, y=obs):
    m1, m2, rep, seeds = unpack(z)
    pr = monthly(params(m1, m2, P12, P21), seeds, rep)
    return (np.sqrt(y) - np.sqrt(np.maximum(pr, 1e-9))).ravel()

LB = np.r_[np.log(0.3), np.log(0.3), -14.0, 0.0, 0.0]          # mosquitoes per person 0.3-6; reporting logit; seeds 1-1e6
UB = np.r_[np.log(6.0), np.log(6.0), 0.0, np.log(1e6), np.log(1e6)]
def fit(P12, P21, y=obs, starts=None, seed=0):
    rng = np.random.default_rng(seed); best = None
    starts = starts or [np.r_[np.log(rng.uniform(.8, 2.5, 2)), rng.uniform(-8, -4), np.log(rng.uniform(50, 3000, 2))] for _ in range(8)]
    for z0 in starts:
        try:
            r = least_squares(resid, np.clip(z0, LB + 1e-6, UB - 1e-6), args=(P12, P21, y), bounds=(LB, UB), method='trf', x_scale='jac', max_nfev=200)
        except Exception:
            continue
        if best is None or r.cost < best.cost: best = r
    return best

def summary(z, P12, P21):
    m1, m2, rep, seeds = unpack(z); p = params(m1, m2, P12, P21)
    return dict(m1=m1, m2=m2, rep=rep, seed1=seeds[0], seed2=seeds[1], R0=R0(p),
                R0_local=R0_local(p).tolist(), pred=monthly(p, seeds, rep).tolist())

cal = dict(window=[str(FIT.index[0].date()), str(FIT.index[-1].date())], background=BG.tolist(), obs=obs.tolist())
P12_0, P21_0 = 0.0048, 0.0120   # BPS Jabodetabek Commuter Survey 2023, Tables 2, 22, 25
best = fit(P12_0, P21_0, seed=1); zb = best.x
cal['baseline'] = summary(zb, P12_0, P21_0)
res_ = resid(zb, P12_0, P21_0); pred = np.array(cal['baseline']['pred'])
ss_res = ((obs - pred) ** 2).sum(); ss_tot = ((obs - obs.mean(0)) ** 2).sum()
cal['baseline'].update(R2=float(1 - ss_res / ss_tot), cost=float(best.cost),
                       peak_obs=[str(FIT.index[obs[:, k].argmax()].date()) for k in range(2)],
                       peak_pred=[str(FIT.index[pred[:, k].argmax()].date()) for k in range(2)])
print('baseline', json.dumps({k: v for k, v in cal['baseline'].items() if k != 'pred'}), flush=True)

# residual bootstrap on the sqrt scale -> uncertainty of estimates and R0
rng = np.random.default_rng(5); B = []
r0 = res_.reshape(obs.shape); fitted_sqrt = np.sqrt(pred)
for b in range(40):
    ystar = np.maximum(fitted_sqrt + rng.choice(r0.ravel(), size=r0.shape, replace=True), 0) ** 2
    rb = fit(P12_0, P21_0, y=ystar, starts=[zb, zb + rng.normal(0, .2, 5)])
    if rb is not None: B.append(summary(rb.x, P12_0, P21_0))
q = lambda k, i=None: np.percentile([b[k] if i is None else b[k][i] for b in B], [2.5, 97.5]).tolist()
cal['bootstrap'] = dict(n=len(B), m1=q('m1'), m2=q('m2'), rep=q('rep'), R0=q('R0'),
                        R0_local1=q('R0_local', 0), R0_local2=q('R0_local', 1))
print('bootstrap', json.dumps(cal['bootstrap']), flush=True)

# sensitivity to the residence fractions
cal['p_sensitivity'] = []
for fac in [0.0, 0.5, 2.0, 4.0]:
    b_ = fit(P12_0 * fac, P21_0 * fac, starts=[zb])
    s_ = summary(b_.x, P12_0 * fac, P21_0 * fac); pr_ = np.array(s_['pred'])
    cal['p_sensitivity'].append(dict(p12=P12_0 * fac, p21=P21_0 * fac, R0=s_['R0'], m1=s_['m1'], m2=s_['m2'],
                                     rep=s_['rep'], R2=float(1 - ((obs - pr_) ** 2).sum() / ss_tot)))
print('p_sens', json.dumps(cal['p_sensitivity']), flush=True)

# counterfactuals with the calibrated parameters (excess reported cases over the window)
m1, m2, rep, seeds = unpack(zb)
def excess(m1_, m2_, p12, p21, theta=0.5, T=1825):
    """reported excess cases over the whole epidemic (5 years from November 2023), as in Table 5."""
    p = params(m1_, m2_, p12, p21, theta); y0 = np.concatenate([dfe(p), [0, 0]])
    for i in range(2):
        y0[i] -= seeds[i]; y0[2 + i] += seeds[i] / 2; y0[4 + i] += seeds[i] / 2
    s = solve_ivp(rhs_c, (0, T), y0, t_eval=[T], method='LSODA', rtol=1e-7, atol=1e-6, args=(p,))
    return float(rep * s.y[14:16, -1].sum())
def excess_by(m1_, m2_, p12, p21, theta=0.5, T=1825):
    p = params(m1_, m2_, p12, p21, theta); y0 = np.concatenate([dfe(p), [0, 0]])
    for i in range(2):
        y0[i] -= seeds[i]; y0[2 + i] += seeds[i] / 2; y0[4 + i] += seeds[i] / 2
    s = solve_ivp(rhs_c, (0, T), y0, t_eval=[T], method='LSODA', rtol=1e-7, atol=1e-6, args=(p,))
    return (rep * s.y[14:16, -1]).tolist()
base_exc = excess(m1, m2, P12_0, P21_0)
dense = 0 if m1 * NH2[0] / NH2[0] >= m2 else 1                 # cluster with more mosquitoes per person
tot_v = m1 * NH2[0] + m2 * NH2[1]; cut = 0.2 * tot_v
def ctrl(where):
    v = np.array([m1 * NH2[0], m2 * NH2[1]])
    if where == 'dense': v[dense] -= cut
    elif where == 'sparse': v[1 - dense] -= cut
    else: v *= 0.8
    return v / NH2
cf = [('Calibrated baseline', excess(m1, m2, P12_0, P21_0), R0(params(m1, m2, P12_0, P21_0))),
      ('No commuting ($p_{12}=p_{21}=0$)', excess(m1, m2, 0, 0), R0(params(m1, m2, 0, 0))),
      ('Commuting doubled', excess(m1, m2, 2 * P12_0, 2 * P21_0), R0(params(m1, m2, 2 * P12_0, 2 * P21_0))),
      ('Infected stay home ($\\theta=1$)', excess(m1, m2, P12_0, P21_0, 1.0), R0(params(m1, m2, P12_0, P21_0, 1.0)))]
for w, lab in [('dense', 'Vector control, denser cluster'), ('uniform', 'Vector control, both clusters'),
               ('sparse', 'Vector control, other cluster')]:
    if w == 'sparse' and cut >= (m1 * NH2[0] if dense == 1 else m2 * NH2[1]): continue
    v = ctrl(w); cf.append((lab, excess(v[0], v[1], P12_0, P21_0), R0(params(v[0], v[1], P12_0, P21_0))))
cal['dense_cluster'] = ['DKI Jakarta', 'Banten'][dense]
cal['counterfactuals'] = [dict(name=n, excess=e, change=e / base_exc - 1, R0=r) for n, e, r in cf]
cal['cut_fraction'] = dict(dense=float(cut / (m1 * NH2[0] if dense == 0 else m2 * NH2[1])),
                           other=float(cut / (m2 * NH2[1] if dense == 0 else m1 * NH2[0])))
vv = {'dense': ctrl('dense'), 'uniform': ctrl('uniform'), 'sparse': ctrl('sparse')}
cal['cf_local_R0'] = {k: R0_local(params(v[0], v[1], P12_0, P21_0)).tolist() for k, v in vv.items()}
cal['cf_by_cluster'] = {k: excess_by(v[0], v[1], P12_0, P21_0) for k, v in vv.items()}
cal['cf_by_cluster']['baseline'] = excess_by(m1, m2, P12_0, P21_0)
print('cf', json.dumps(cal['counterfactuals']), flush=True)

# figure: observed vs fitted
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=False)
months = pd.to_datetime(FIT.index); lab = [m.strftime('%b\n%Y') if m.month in (1, 11) else m.strftime('%b') for m in months]
pb = np.array([b['pred'] for b in B])
for k, (nm, c) in enumerate([('DKI Jakarta (cluster 1)', '#c0392b'), ('Banten (cluster 2)', '#2471a3')]):
    x = np.arange(len(months))
    ax[k].fill_between(x, np.percentile(pb[:, :, k], 2.5, 0), np.percentile(pb[:, :, k], 97.5, 0), color=c, alpha=.15, lw=0)
    ax[k].plot(x, pred[:, k], color=c, lw=2, label='model')
    ax[k].plot(x, obs[:, k], 'o', color='k', ms=4, label='reported (OpenDengue)')
    ax[k].axhline(BG[k], color='gray', ls=':', lw=1)
    ax[k].set_xticks(x); ax[k].set_xticklabels(lab, fontsize=8); ax[k].set_title(nm, fontsize=10)
    ax[k].yaxis.set_major_formatter(FuncFormatter(lambda v, pos: '{:,.0f}'.format(v)))
    ax[k].spines['top'].set_visible(False); ax[k].spines['right'].set_visible(False)
ax[0].set_ylabel('reported cases per month'); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT3, 'gambar6_calibration.png'), dpi=200); plt.close(fig)
json.dump(cal, open(os.path.join(OUT3, 'results_calibration.json'), 'w'), indent=1, default=float)
print('done', flush=True)
