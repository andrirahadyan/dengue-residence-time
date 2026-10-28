"""
Model SEIR-SEI DBD dua cluster dengan matriks waktu tinggal (Lagrangian).
Penduduk cluster i (sehat: S,E,R) menghabiskan fraksi waktu p_ij di cluster j;
penduduk terinfeksi (I) memakai q_ij = (1-theta) p_ij (i != j), q_ii = 1 - sum_{j!=i} q_ij.
Nyamuk tidak berpindah. Semua hasil numerik naskah dihasilkan skrip ini.
Jalankan: python model_waktu_tinggal.py
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy import stats
from scipy.stats import qmc
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'en')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.formatter.use_locale': False})
koma = FuncFormatter(lambda x, pos: ('%g' % x))
fk = lambda x, d=3: ('%.*f' % (d, x))
fm = lambda x, d=3: fk(x, d)   # koma desimal di dalam mathtext
muh = 1 / (60 * 365)
LIT = dict(b=1/3, ph=0.5, pv=0.4, beta=1/5.9, omega=1/7, rho=1/8, muv=0.1, muh=muh)
NH = 10000.0          # total penduduk kedua cluster (simulasi dan R0; R0 tidak bergantung pada skala)

# ------------------------------------------------------------------ parameter
def make(ratio=1.0, rv=1.0, rh=1.0, p12=0.1, p21=None, theta=0.5, **kw):
    """ratio = N_v/N_h total; rv = rasio nyamuk cluster 1:2; rh = rasio penduduk 1:2."""
    p = dict(LIT); p.update(kw)
    if p21 is None: p21 = p12
    Nh = np.array([NH * rh / (1 + rh), NH / (1 + rh)])
    Nv = ratio * NH * np.array([rv / (1 + rv), 1 / (1 + rv)])
    p.update(Nh=Nh, Nv=Nv, P=np.array([[1 - p12, p12], [p21, 1 - p21]]), theta=theta)
    q12, q21 = (1 - theta) * p12, (1 - theta) * p21
    p['Q'] = np.array([[1 - q12, q12], [q21, 1 - q21]])
    return p

def mfac(p): return p['b'] * p['ph'] * p['rho'] / (p['muv'] * (p['rho'] + p['muv']))
def cfac(p): return p['b'] * p['pv'] * p['beta'] / ((p['beta'] + p['muh']) * (p['omega'] + p['muh']))

def K_matrix(p):
    Nt = p['P'].T @ p['Nh']                       # populasi efektif tiap cluster pada DFE
    return mfac(p) * cfac(p) * np.diag(p['Nh']) @ p['P'] @ np.diag(p['Nv'] / Nt**2) @ p['Q'].T

def R0(p): return float(np.sqrt(max(abs(np.linalg.eigvals(K_matrix(p))))))
def R0_local(p): return np.sqrt(mfac(p) * cfac(p) * p['Nv'] / p['Nh'])

def R0_ngm(p):
    """radius spektral F V^-1 pada sistem lengkap (x = Eh1,Eh2,Ih1,Ih2,Ev1,Ev2,Iv1,Iv2)."""
    P, Q, Nh, Nv = p['P'], p['Q'], p['Nh'], p['Nv']; Nt = P.T @ Nh; mh = p['muh']
    F = np.zeros((8, 8)); V = np.zeros((8, 8))
    for i in range(2):
        for j in range(2):
            F[i, 6 + j] = p['b'] * p['ph'] * Nh[i] * P[i, j] / Nt[j]
            F[4 + j, 2 + i] = p['b'] * p['pv'] * Nv[j] * Q[i, j] / Nt[j]
        V[i, i] = p['beta'] + mh; V[2 + i, 2 + i] = p['omega'] + mh; V[2 + i, i] = -p['beta']
        V[4 + i, 4 + i] = p['rho'] + p['muv']; V[6 + i, 6 + i] = p['muv']; V[6 + i, 4 + i] = -p['rho']
    return float(max(abs(np.linalg.eigvals(F @ np.linalg.inv(V)))))

def R0_closed_sym(r1, r2, d):
    S = r1 + r2; s = (1 + d) / 2
    return np.sqrt((s * S + np.sqrt((s * S) ** 2 - 4 * d * r1 * r2)) / 2)

# ------------------------------------------------------------------ ODE
def rhs(t, y, p):
    Sh, Eh, Ih, Rh = y[0:2], y[2:4], y[4:6], y[6:8]
    Sv, Ev, Iv = y[8:10], y[10:12], y[12:14]
    P, Q = p['P'], p['Q']; mh = p['muh']
    Nt = P.T @ (Sh + Eh + Rh) + Q.T @ Ih
    lam = p['b'] * p['ph'] * P @ (Iv / Nt)                  # gaya infeksi pada penduduk i
    lamv = p['b'] * p['pv'] * (Q.T @ Ih) / Nt                # gaya infeksi pada nyamuk j
    Ah = mh * p['Nh']; Av = p['muv'] * p['Nv']
    return np.concatenate([Ah - lam * Sh - mh * Sh, lam * Sh - (p['beta'] + mh) * Eh,
                           p['beta'] * Eh - (p['omega'] + mh) * Ih, p['omega'] * Ih - mh * Rh,
                           Av - lamv * Sv - p['muv'] * Sv, lamv * Sv - (p['rho'] + p['muv']) * Ev,
                           p['rho'] * Ev - p['muv'] * Iv])

def dfe(p): return np.concatenate([p['Nh'], np.zeros(6), p['Nv'], np.zeros(4)])

def jac_maxeig(p):
    x0 = dfe(p); J = np.zeros((14, 14)); h = 1e-6
    for i in range(14):
        e = np.zeros(14); e[i] = h; J[:, i] = (rhs(0, x0 + e, p) - rhs(0, x0 - e, p)) / (2 * h)
    return float(max(np.linalg.eigvals(J).real))

def simulate(p, T=730):
    y0 = dfe(p); y0[0] -= 1; y0[4] += 1          # satu penduduk cluster 1 terinfeksi
    t = np.linspace(0, T, 7301)
    s = solve_ivp(rhs, (0, T), y0, t_eval=t, method='LSODA', rtol=1e-8, atol=1e-8, args=(p,))
    return s.t, s.y

def p_star(r1, r2, theta):
    ds = (r1 + r2 - 2) / (2 * r1 * r2 - r1 - r2)
    if theta == 1: return (1 - ds) / 2, ds
    a = 2 - theta; return (a - np.sqrt(a * a - 4 * (1 - theta) * (1 - ds))) / (4 * (1 - theta)), ds

res = {}
# ------------------------------------------------------------------ 1. verifikasi
rng = np.random.default_rng(7); e1 = e2 = 0
for _ in range(300):
    p = make(ratio=rng.uniform(.5, 3), rv=rng.uniform(1, 3), rh=rng.uniform(1, 3), p12=rng.uniform(0, .5),
             p21=rng.uniform(0, .5), theta=rng.uniform(0, 1), b=rng.uniform(.2, .5), ph=rng.uniform(.1, .75),
             pv=rng.uniform(.1, .75), beta=rng.uniform(.1, 1/3), omega=rng.uniform(.1, .25),
             rho=rng.uniform(1/15, .2), muv=rng.uniform(.07, .2))
    e1 = max(e1, abs(R0(p) - R0_ngm(p)) / R0_ngm(p))   # rho(FV^-1) = sqrt(rho(K))
for _ in range(300):   # bentuk tertutup kasus simetris
    pp = rng.uniform(0, 1); th = rng.uniform(0, 1)
    p = make(ratio=rng.uniform(.5, 3), rv=rng.uniform(1, 5), p12=pp, theta=th)
    r1, r2 = R0_local(p) ** 2; d = (1 - 2 * pp) * (1 - 2 * (1 - th) * pp)
    e2 = max(e2, abs(R0(p) - R0_closed_sym(r1, r2, d)) / R0(p))
res['verif'] = dict(maks_rel_K_vs_FVinv=e1, maks_rel_tertutup_vs_numerik=e2)

# ------------------------------------------------------------------ 2. baseline dan ambang
res['R0_sim_baseline'] = R0(make())
for th in [0.0, 0.5, 1.0]:
    for pp in [0.0, 0.1, 0.33, 0.5]:
        assert abs(R0(make(p12=pp, theta=th)) - res['R0_sim_baseline']) < 1e-12
p2 = make(rv=2); r1, r2 = R0_local(p2) ** 2
res['r1'], res['r2'] = r1, r2
res['R0_2_1_p0'] = R0(make(rv=2, p12=0)); res['R0_2_1_mean'] = float(np.sqrt((r1 + r2) / 2))
res['R0_2_1_geo'] = float(np.sqrt(np.sqrt(r1 * r2)))
for th in [0, 0.5, 1]:
    ps, ds = p_star(r1, r2, th)
    res['pstar_theta%s' % th] = ps; res['dstar'] = ds
    res['pstar_numerik_theta%s' % th] = brentq(lambda x: R0(make(rv=2, p12=x, theta=th)) - 1, 1e-9, 0.5)
res['pstar_5_1_theta0.5'] = brentq(lambda x: R0(make(rv=5, p12=x, theta=.5)) - 1, 1e-9, 0.5)
# efek theta pada p = 0.1 dan p = 1/3
res['efek_theta'] = {('p=%.3f' % pp): {('theta=%g' % th): R0(make(rv=2, p12=pp, theta=th)) for th in [0, .5, 1]}
                     for pp in [0.1, 1/3]}
# ------------------------------------------------------------------ 3. Tabel 2
cols = [0.0, 0.05, 0.1, 1/3]
res['Tabel2'] = {str(r): [R0(make(rv=r, p12=c)) for c in cols] for r in [1, 1.5, 2, 3, 5]}
# ------------------------------------------------------------------ 4. kasus asimetris (nyamuk 2:1, penduduk sama)
g = np.linspace(0, 0.5, 201); Z = np.array([[R0(make(rv=2, p12=a, p21=b_)) for a in g] for b_ in g])
res['asim'] = dict(R0_max=float(Z.max()), R0_min=float(Z.min()),
                   argmax=[float(g[np.unravel_index(Z.argmax(), Z.shape)[1]]), float(g[np.unravel_index(Z.argmax(), Z.shape)[0]])],
                   argmin=[float(g[np.unravel_index(Z.argmin(), Z.shape)[1]]), float(g[np.unravel_index(Z.argmin(), Z.shape)[0]])],
                   R0_p12_0_p21_05=R0(make(rv=2, p12=0, p21=.5)), R0_p12_05_p21_0=R0(make(rv=2, p12=.5, p21=0)),
                   R0_p12_0_p21_02=R0(make(rv=2, p12=0, p21=.2)), R0_p12_02_p21_0=R0(make(rv=2, p12=.2, p21=0)),
                   lokal_max=float(np.sqrt(r1)))
# ------------------------------------------------------------------ 5. Jacobian
J = {}
for nm, kw in [('simetris', dict()), ('2:1 p=0.1', dict(rv=2)), ('2:1 p=0', dict(rv=2, p12=0)),
               ('2:1 p=0.05 theta=1', dict(rv=2, p12=0.05, theta=1)), ('sim p=0.1', dict(ratio=2, rv=2)),
               ('asim p12=0.5 p21=0', dict(rv=2, p12=.5, p21=0))]:
    q = make(**kw); J['%s (R0=%.3f)' % (nm, R0(q))] = jac_maxeig(q)
res['Jacobian'] = J
# ------------------------------------------------------------------ 6. Gambar 1
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
pg = np.linspace(0, 0.5, 300)
for r, c, l in [(1, '#444444', 'Symmetric (mosquitoes 1:1)'), (2, '#c0392b', 'Mosquitoes 2:1'), (5, '#2471a3', 'Mosquitoes 5:1')]:
    ax[0].plot(pg, [R0(make(rv=r, p12=x, theta=0)) for x in pg], color=c, lw=2, label=l + r', $\theta=0$')
    if r > 1: ax[0].plot(pg, [R0(make(rv=r, p12=x, theta=1)) for x in pg], color=c, lw=1.6, ls='--', label=l + r', $\theta=1$')
ax[0].axhline(1, color='k', ls=':', lw=.8)
ax[0].set_xlabel(r'$p$ (fraction of time residents spend in the other cluster)'); ax[0].set_ylabel(r'$R_0$')
ax[0].set_title(r'(a) $R_0$ vs $p$ (symmetric residence times)', fontsize=10); ax[0].legend(frameon=False, fontsize=8)
cs = ax[1].contourf(g, g, Z, levels=14, cmap='RdBu_r'); cb = fig.colorbar(cs, ax=ax[1]); cb.set_label(r'$R_0$')
cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fk(x, 2)))
c1 = ax[1].contour(g, g, Z, levels=[1.0], colors='k', linewidths=1.5); ax[1].clabel(c1, fmt={1.0: '$R_0=1$'}, fontsize=8)
ax[1].plot(g, g, color='w', ls='--', lw=1); ax[1].set_xlabel(r'$p_{12}$ (cluster 1 residents in cluster 2)')
ax[1].set_ylabel(r'$p_{21}$ (cluster 2 residents in cluster 1)')
ax[1].set_title(r'(b) Asymmetric residence times (mosquitoes 2:1, $\theta=0.5$)', fontsize=10)
for a in ax:
    a.xaxis.set_major_formatter(koma); a.yaxis.set_major_formatter(koma)
for a in [ax[0]]: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'gambar1.png'), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 7. sensitivitas lokal (Gambar 2)
base = make(rv=2); r0b = R0(base)
def perturb(name, f):
    q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    if name in ('Nh', 'Nv'): q[name] = base[name] * f
    elif name == 'p':
        pp = base['P'][0, 1] * f; return make(rv=2, p12=pp, theta=base['theta'])
    elif name == 'theta': return make(rv=2, p12=base['P'][0, 1], theta=base['theta'] * f)
    else: q[name] = base[name] * f
    return q
labs = [('$\\mu_v$', 'muv'), ('$b$', 'b'), ('$p_h$', 'ph'), ('$p_v$', 'pv'), ('$N_v$ (both clusters)', 'Nv'),
        ('$N_h$ (both clusters)', 'Nh'), ('$\\omega$', 'omega'), ('$\\rho$', 'rho'), ('$p$', 'p'), ('$\\theta$', 'theta'),
        ('$\\beta$', 'beta'), ('$\\mu_h$', 'muh')]
G = {}
for lab, nm in labs:
    h = 1e-4; G[lab] = (R0(perturb(nm, 1 + h)) - R0(perturb(nm, 1 - h))) / (2 * h) / r0b
res['Gamma'] = {k.replace('$', '').replace('\\', ''): v for k, v in G.items()}; res['R0_fig2'] = r0b
fig, ax = plt.subplots(figsize=(6.5, 4.2)); order = sorted(G, key=lambda k: abs(G[k]))
ax.barh(order, [G[k] for k in order], color=['#c0392b' if G[k] > 0 else '#2471a3' for k in order]); ax.axvline(0, color='k', lw=.8)
ax.set_title(r'Sensitivity of $R_0$ (mosquitoes 2:1, $p=0.1$, $\theta=0.5$, $R_0=%s$)' % fm(r0b), fontsize=10)
ax.set_xlabel(r'Normalized forward sensitivity index $\Gamma_x=\frac{\partial R_0}{\partial x}\frac{x}{R_0}$')
ax.xaxis.set_major_formatter(koma); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'gambar2.png'), dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 8. simulasi (Tabel 3, Gambar 3)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True); T3 = []
for i, pp in enumerate([0.02, 0.1, 1/3]):
    p = make(ratio=2, rv=2, p12=pp, theta=0.5); t, y = simulate(p)
    per = 1000 / p['Nh'][0]; I1, I2 = y[4] * per, y[5] * per; It = (y[4] + y[5]) * 1000 / NH
    att = lambda k: (y[6 + k][-1] + y[4 + k][-1] + y[2 + k][-1]) * 1000 / p['Nh'][k]
    T3.append(dict(p=pp, R0=R0(p), peak_tot=float(It.max()), t_tot=float(t[It.argmax()]), peak1=float(I1.max()),
                   t1=float(t[I1.argmax()]), peak2=float(I2.max()), t2=float(t[I2.argmax()]),
                   serang1=float(att(0)), serang2=float(att(1)), serang_tot=float((att(0) + att(1)) / 2),
                   cek_N=float(abs(y[:8].sum(0) - NH).max())))
    ax[i].plot(t, I1, color='#c0392b', lw=1.8, label=r'$I_{h1}$ (residents of the dense cluster)')
    ax[i].plot(t, I2, color='#2471a3', lw=1.8, label=r'$I_{h2}$')
    ax[i].set_title(r'$p=%s$, $R_0=%s$' % (fm(pp, 2) if pp != 1/3 else '1/3', fm(T3[-1]['R0'], 2)), fontsize=10)
    ax[i].set_xlabel('t (days)'); ax[i].xaxis.set_major_formatter(koma); ax[i].yaxis.set_major_formatter(koma)
    ax[i].spines['top'].set_visible(False); ax[i].spines['right'].set_visible(False)
ax[0].set_ylabel('infected per 1,000 residents'); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'gambar3.png'), dpi=200); plt.close(fig)
res['Tabel3'] = T3
# pembanding theta pada p = 0.1
cmp = {}
for th in [0, 1]:
    p = make(ratio=2, rv=2, p12=0.1, theta=th); t, y = simulate(p); It = (y[4] + y[5]) * 1000 / NH
    att = (y[6][-1] + y[7][-1] + y[4][-1] + y[5][-1] + y[2][-1] + y[3][-1]) * 1000 / NH
    cmp['theta=%g' % th] = dict(R0=R0(p), peak=float(It.max()), t=float(t[It.argmax()]), serang=float(att))
res['sim_theta_p0.1'] = cmp

# ------------------------------------------------------------------ 9. sensitivitas global (Gambar 4)
names = ['b', 'p_h', 'p_v', 'beta', 'omega', 'rho', 'mu_v', 'ratio', 'r_v', 'r_h', 'p12', 'p21', 'theta']
lo = dict(b=0.2, p_h=0.1, p_v=0.1, beta=1/10, omega=1/10, rho=1/15, mu_v=0.07, ratio=0.5, r_v=1.0, r_h=1.0, p12=0.0, p21=0.0, theta=0.0)
hi = dict(b=0.5, p_h=0.75, p_v=0.75, beta=1/3, omega=1/4, rho=1/5, mu_v=0.20, ratio=3.0, r_v=3.0, r_h=3.0, p12=0.5, p21=0.5, theta=1.0)
labels = {'b': '$b$', 'p_h': '$p_h$', 'p_v': '$p_v$', 'beta': r'$\beta$', 'omega': r'$\omega$', 'rho': r'$\rho$', 'mu_v': r'$\mu_v$',
          'ratio': '$N_v/N_h$', 'r_v': 'between-cluster mosquito ratio', 'r_h': 'between-cluster resident ratio',
          'p12': '$p_{12}$', 'p21': '$p_{21}$', 'theta': r'$\theta$'}
def scale(U): return np.column_stack([lo[n] + U[:, j] * (hi[n] - lo[n]) for j, n in enumerate(names)])

def R0_vec(X, zero_mob=False):
    c = {n: X[:, j] for j, n in enumerate(names)}
    b, ph, pv, be, om, rho, mv = c['b'], c['p_h'], c['p_v'], c['beta'], c['omega'], c['rho'], c['mu_v']
    p12, p21, th = c['p12'], c['p21'], c['theta']
    if zero_mob: p12 = np.zeros_like(p12); p21 = np.zeros_like(p21)
    Nh1 = c['r_h'] / (1 + c['r_h']); Nh2 = 1 / (1 + c['r_h'])
    Nv1 = c['ratio'] * c['r_v'] / (1 + c['r_v']); Nv2 = c['ratio'] / (1 + c['r_v'])
    m = b * ph * rho / (mv * (rho + mv)); cc = b * pv * be / ((be + muh) * (om + muh))
    P11, P12, P21, P22 = 1 - p12, p12, p21, 1 - p21
    q12, q21 = (1 - th) * p12, (1 - th) * p21; Q11, Q12, Q21, Q22 = 1 - q12, q12, q21, 1 - q21
    Nt1 = P11 * Nh1 + P21 * Nh2; Nt2 = P12 * Nh1 + P22 * Nh2
    w1, w2 = Nv1 / Nt1**2, Nv2 / Nt2**2
    # K = mc diag(Nh) P diag(w) Q^T
    K11 = Nh1 * (P11 * w1 * Q11 + P12 * w2 * Q12); K12 = Nh1 * (P11 * w1 * Q21 + P12 * w2 * Q22)
    K21 = Nh2 * (P21 * w1 * Q11 + P22 * w2 * Q12); K22 = Nh2 * (P21 * w1 * Q21 + P22 * w2 * Q22)
    tr = K11 + K22; det = K11 * K22 - K12 * K21
    return np.sqrt(m * cc * (tr + np.sqrt(np.maximum(tr * tr - 4 * det, 0))) / 2)

# cek R0_vec terhadap R0()
Uc = qmc.LatinHypercube(d=len(names), seed=1).random(50); Xc = scale(Uc); errv = 0
for x in Xc:
    d_ = dict(zip(names, x))
    q = make(ratio=d_['ratio'], rv=d_['r_v'], rh=d_['r_h'], p12=d_['p12'], p21=d_['p21'], theta=d_['theta'], b=d_['b'], ph=d_['p_h'],
             pv=d_['p_v'], beta=d_['beta'], omega=d_['omega'], rho=d_['rho'], muv=d_['mu_v'])
    errv = max(errv, abs(R0(q) - R0_vec(x[None, :])[0]) / R0(q))
res['verif']['maks_rel_vektor_vs_matriks'] = errv
N = 5000; X = scale(qmc.LatinHypercube(d=len(names), seed=11).random(N)); Y = R0_vec(X)
def prcc(X, Y):
    R = np.column_stack([stats.rankdata(X[:, j]) for j in range(X.shape[1])] + [stats.rankdata(Y)])
    Pm = np.linalg.pinv(np.corrcoef(R, rowvar=False)); d = -Pm[:-1, -1] / np.sqrt(Pm[:-1, :-1].diagonal() * Pm[-1, -1])
    n, k = X.shape; t = d * np.sqrt((n - 2 - (k - 1)) / (1 - d**2)); return d, 2 * stats.t.sf(abs(t), n - 2 - (k - 1))
rp, pval = prcc(X, Y)
problem = dict(num_vars=len(names), names=names, bounds=[[0, 1]] * len(names))
Us = sobol_sample.sample(problem, 4096, calc_second_order=False, seed=3); Ys = R0_vec(scale(Us))
Si = sobol_analyze.analyze(problem, Ys, calc_second_order=False, print_to_console=False, seed=3)
Y0 = R0_vec(X, zero_mob=True); ratio = Y / Y0
res['global'] = dict(median=float(np.median(Y)), p5=float(np.percentile(Y, 5)), p95=float(np.percentile(Y, 95)),
                     frac_gt1=float((Y > 1).mean()),
                     PRCC={names[j]: [float(rp[j]), float(pval[j])] for j in range(len(names))},
                     S1={names[j]: float(Si['S1'][j]) for j in range(len(names))},
                     ST={names[j]: float(Si['ST'][j]) for j in range(len(names))}, sumS1=float(Si['S1'].sum()),
                     mob_ratio_median=float(np.median(ratio)), mob_ratio_min=float(ratio.min()), mob_ratio_max=float(ratio.max()),
                     mob_ratio_p5=float(np.percentile(ratio, 5)), mob_ratio_p95=float(np.percentile(ratio, 95)),
                     frac_mob_naik=float((ratio > 1 + 1e-9).mean()),
                     frac_turun_lewat_ambang=float(((Y0 > 1) & (Y < 1)).mean()),
                     frac_naik_lewat_ambang=float(((Y0 < 1) & (Y > 1)).mean()))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2)); idx = np.argsort(abs(rp))
ax[0].barh([labels[names[j]] for j in idx], [rp[j] for j in idx], color=['#c0392b' if rp[j] > 0 else '#2471a3' for j in idx])
ax[0].axvline(0, color='k', lw=.8); ax[0].set_xlabel('PRCC with $R_0$'); ax[0].set_title('(a) PRCC', fontsize=10)
idx2 = np.argsort(Si['ST']); yy = np.arange(len(names)); h = 0.38
ax[1].barh(yy + h / 2, Si['ST'][idx2], h, color='#7f8c8d', label='$S_T$ (total)')
ax[1].barh(yy - h / 2, Si['S1'][idx2], h, color='#1e8449', label='$S_1$ (first order)')
ax[1].set_yticks(yy); ax[1].set_yticklabels([labels[names[j]] for j in idx2]); ax[1].set_xlabel('Sobol index')
ax[1].legend(frameon=False, loc='lower right'); ax[1].set_title('(b) Sobol indices', fontsize=10)
for a in ax:
    a.xaxis.set_major_formatter(koma); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'gambar4_sensitivitas_global.png'), dpi=200); plt.close(fig)

json.dump(res, open(os.path.join(OUT, 'results_en.json'), 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
