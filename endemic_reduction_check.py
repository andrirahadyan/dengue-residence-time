"""
Check of the fixed-point reduction of the endemic equilibrium (Section 3.4, Propositions 3.4-3.5).
For theta = 0, every positive equilibrium of the full 14-dimensional model should equal the unique
fixed point of the two-dimensional map H in equation (11), and no positive equilibrium should exist
when R0 <= 1. Reuses the model definitions of model_waktu_tinggal_en.py.
Run: python endemic_reduction_check.py
"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'model_waktu_tinggal_en.py')).read().split('res = {}')[0])
from scipy.optimize import fsolve


def H(y, p):
    """map H of equation (11) for theta = 0 (effective population constant)."""
    Nh, Nv, P, Q = p['Nh'], p['Nv'], p['P'], p['Q']
    Nt = P.T @ Nh
    Lam = p['b'] * p['pv'] * (Q.T @ y) / Nt
    Iv = p['rho'] * Lam * Nv / ((p['rho'] + p['muv']) * (Lam + p['muv']))
    Ch = p['beta'] / ((p['beta'] + p['muh']) * (p['omega'] + p['muh']))
    Phi = Ch * p['b'] * p['ph'] * (P @ (Iv / Nt))
    kap = (p['omega'] + p['muh']) / p['beta'] + 1 + p['omega'] / p['muh']
    return Nh * Phi / (1 + kap * Phi)


rng = np.random.default_rng(7)
n_below = n_below_none = n_above = n_roots = n_equal = 0
max_dev = max_jac = 0.0
for _ in range(400):
    p = make(ratio=rng.uniform(.5, 3), rv=rng.uniform(1, 3), rh=rng.uniform(1, 3), p12=rng.uniform(.01, .5),
             p21=rng.uniform(.01, .5), theta=0.0, b=rng.uniform(.2, .5), ph=rng.uniform(.1, .75),
             pv=rng.uniform(.1, .75), beta=rng.uniform(.1, 1/3), omega=rng.uniform(.1, .25),
             rho=rng.uniform(1/15, .2), muv=rng.uniform(.07, .2))
    Nh = p['Nh']
    # DH(0) = K (Proposition 3.4, chain rule)
    K = mfac(p) * cfac(p) * np.diag(Nh) @ p['P'] @ np.diag(p['Nv'] / (p['P'].T @ Nh) ** 2) @ p['Q'].T
    J = np.column_stack([H(1e-6 * e, p) / 1e-6 for e in np.eye(2)])
    max_jac = max(max_jac, float(np.abs(J - K).max() / np.abs(K).max()))
    # positive equilibria of the full model from 8 random starts
    sc = np.concatenate([np.tile(Nh, 4), np.tile(p['Nv'], 3)])
    f = lambda z: rhs(0, z * sc, p) / sc
    roots = []
    for _ in range(8):
        z, info, ier, _ = fsolve(f, rng.uniform(0.01, 0.9, 14), full_output=True, xtol=1e-13)
        if ier == 1 and np.abs(f(z)).max() < 1e-10 and z.min() > -1e-9 and z[[4, 5]].min() > 1e-9:
            roots.append(z * sc)
    if R0(p) <= 1:
        n_below += 1; n_below_none += (len(roots) == 0); continue
    n_above += 1
    g = lambda u: u - H(u * Nh, p) / Nh
    y = fsolve(g, np.array([.1, .1]), xtol=1e-14) * Nh
    for x in roots:
        n_roots += 1; d = float(np.abs(x[4:6] - y).max() / Nh.max())
        max_dev = max(max_dev, d); n_equal += d < 1e-9

out = dict(max_rel_diff_DH0_vs_K=max_jac, sets_R0_le_1=n_below, of_which_no_positive_equilibrium=n_below_none,
           sets_R0_gt_1=n_above, positive_roots=n_roots, roots_equal_to_fixed_point=n_equal,
           max_deviation_over_N=max_dev)
os.makedirs(os.path.join(HERE, 'en'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'en', 'results_endemic_check.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
