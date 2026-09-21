"""
Numerical verification of the Dirac frame representation of a graph-supported
continuous-time Markov generator, of the frame regularizer, of the constrained
estimation problem, and of the stability bound for the transition kernel.

The script is self-contained. It builds the augmented state space of a finite
graph, constructs the incidence Dirac operator and the associated frame,
verifies the frame identities, estimates a generator from censored endpoint
data by minimizing the frame regularizer subject to the Markov constraints and
a likelihood constraint, and checks the resulting estimate against the
stability bound.

Entry point:  run()

Requires numpy, scipy, matplotlib.
"""
import os
import sys
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.stats import chi2

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.cm import ScalarMappable

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
SEED = 123
K_STATES = 6                                   # population ladder 0..K
THETA_TRUE = (0.90, 0.25, 0.10, 2.00, 2.00)    # (r, d, a, kappa_L, kappa_R)
TIMES = (0.5, 1.0)                             # observation horizons
N_PER_TIME = 200_000                           # endpoint pairs per horizon
ALPHA = 0.05                                   # likelihood-ratio level
OUT = "figures"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "font.family": "serif",
    "axes.linewidth": 0.8, "axes.grid": False,
    "image.interpolation": "nearest",
})
DIV, POS, MAG = "coolwarm", "viridis", "magma"


# ======================================================================
# 1. Augmented state space, incidence matrix, Dirac operator
# ======================================================================
def build_geometry(K=K_STATES):
    """Path graph on V = {0,...,K} with edges e_j = (j, j+1).

    The augmented state space is Gamma = V u E, with |V| = K+1 vertices and
    |E| = K edges, so n = 2K+1. The signed incidence matrix B in R^{V x E}
    carries B[j, j] = -1 at the tail and B[j+1, j] = +1 at the head of edge
    e_j. The Dirac operator is the off-diagonal symmetric lift

        D = [[0, B], [B.T, 0]]   on   R^n,

    which couples every vertex to its incident edge-states and encodes the
    graph geometry the frame is built from.
    """
    nV, nE = K + 1, K
    n = nV + nE
    j = np.arange(nE)
    B = np.zeros((nV, nE))
    B[j, j] = -1.0                 # tail of edge e_j is vertex j
    B[j + 1, j] = +1.0             # head of edge e_j is vertex j+1
    D = np.zeros((n, n))
    D[:nV, nV:] = B
    D[nV:, :nV] = B.T
    edges = list(zip(j.tolist(), (j + 1).tolist()))
    return dict(K=K, nV=nV, nE=nE, n=n, B=B, D=D, edges=edges)


def support_pairs(D):
    """The incidence support S = {(alpha,beta) : D_{alpha beta} != 0}."""
    n = D.shape[0]
    return [(x, y) for x in range(n) for y in range(n) if D[x, y] != 0.0]


# ======================================================================
# 2. Frame elements
# ======================================================================
def build_frame(D):
    """M_lambda = diag(u_lambda) and G_{mu nu} = M_nu D M_mu."""
    n = D.shape[0]
    lam, U = np.linalg.eigh(D)
    M = [np.diag(U[:, m]) for m in range(n)]
    G = np.empty((n, n), dtype=object)
    for mu in range(n):
        for nu in range(n):
            G[mu, nu] = np.outer(U[:, nu], U[:, mu]) * D
    u_inf = np.array([np.max(np.abs(U[:, m])) for m in range(n)])
    a_val = max(np.abs(D).sum(axis=0).max(), np.abs(D).sum(axis=1).max())
    return dict(lam=lam, U=U, M=M, G=G, u_inf=u_inf, a=a_val, n=n, D=D)


def coeff_c(Q, fr):
    """c_{mu nu} = Tr(M_nu Q M_mu D)."""
    n, M, D = fr["n"], fr["M"], fr["D"]
    return np.array([[np.trace(M[nu] @ Q @ M[mu] @ D) for nu in range(n)]
                     for mu in range(n)])


def coeff_d(Q, fr):
    """d_lambda = Tr(M_lambda Q)."""
    n, M = fr["n"], fr["M"]
    return np.array([np.trace(M[l] @ Q) for l in range(n)])


def synthesize(c, d, fr):
    """Q = sum_{mu,nu} c_{mu nu} G_{mu nu} + sum_lambda d_lambda M_lambda."""
    n, G, M = fr["n"], fr["G"], fr["M"]
    Q = np.zeros((n, n))
    for mu in range(n):
        for nu in range(n):
            Q += c[mu, nu] * G[mu, nu]
    for l in range(n):
        Q += d[l] * M[l]
    return Q


def regularizer(c, d, fr):
    """R_Q = a sum |c| ||u_mu||_inf ||u_nu||_inf + sum |d_l| ||u_l||_inf."""
    ui, a = fr["u_inf"], fr["a"]
    return float(a * np.sum(np.abs(c) * np.outer(ui, ui))
                 + np.sum(np.abs(d) * ui))


def op_norm_11(A):
    return float(np.abs(A).sum(axis=0).max())


def op_norm_inf(A):
    return float(np.abs(A).sum(axis=1).max())


# ======================================================================
# 3. Logistic birth and death model, endpoint data
# ======================================================================
def generator(theta, geo):
    r, d, a, kL, kR = theta
    K, nV, n = geo["K"], geo["nV"], geo["n"]
    Q = np.zeros((n, n))
    for i in range(K):
        Q[i, nV + i] = a + r * i * (1.0 - i / K)
    for i in range(1, K + 1):
        Q[i, nV + i - 1] = d * i
    for i in range(K):
        Q[nV + i, i] = kL
        Q[nV + i, i + 1] = kR
    for x in range(n):
        Q[x, x] = -(Q[x].sum() - Q[x, x])
    return Q


def endpoint_counts(Q, times, n_per_time, rng):
    """Aggregated endpoint counts, sampled exactly from multinomials."""
    n = Q.shape[0]
    out = {}
    for T in times:
        P = np.clip(expm(T * Q), 0.0, None)
        P /= P.sum(1, keepdims=True)
        starts = rng.multinomial(n_per_time, np.ones(n) / n)
        C = np.zeros((n, n))
        for x in range(n):
            if starts[x]:
                C[x] = rng.multinomial(starts[x], P[x])
        out[T] = C
    return out


# ======================================================================
# 4. Constraint set and log-likelihood
# ======================================================================
class Problem:
    """
    Decision variable w on the incidence support, with
        Q(w)_{alpha beta} = w_{alpha beta},  Q(w)_{alpha alpha} = -sum_beta w.
    Together with w >= 0 this parametrizes exactly the set of graph-supported
    conservative generators with nonnegative off-diagonal entries.
    """
    FLOOR = 1e-300
    GFLOOR = 1e-12

    def __init__(self, geo, fr, counts):
        self.geo, self.fr, self.counts = geo, fr, counts
        self.n = geo["n"]
        self.S = support_pairs(geo["D"])
        self.m = len(self.S)
        Ac = np.zeros((self.n * self.n, self.m))
        Ad = np.zeros((self.n, self.m))
        for j, (al, be) in enumerate(self.S):
            E = np.zeros((self.n, self.n))
            E[al, be] = 1.0
            E[al, al] = -1.0
            Ac[:, j] = coeff_c(E, fr).ravel()
            Ad[:, j] = coeff_d(E, fr)
        self.Ac, self.Ad = Ac, Ad
        ui = fr["u_inf"]
        self.Wc = (fr["a"] * np.outer(ui, ui)).ravel()
        self.Wd = ui.copy()

    def Q(self, w):
        n = self.n
        Q = np.zeros((n, n))
        for j, (al, be) in enumerate(self.S):
            Q[al, be] = w[j]
        for x in range(n):
            Q[x, x] = -Q[x].sum()
        return Q

    def loglik(self, w):
        Q = self.Q(w)
        tot = 0.0
        for T, C in self.counts.items():
            P = expm(T * Q)
            mask = C > 0
            Pm = P[mask]
            if np.any(Pm <= self.FLOOR):
                return -np.inf
            tot += float(np.sum(C[mask] * np.log(Pm)))
        return tot

    def loglik_grad(self, w):
        """Adjoint Frechet derivative: two expm_frechet calls per horizon."""
        Q = self.Q(w)
        g = np.zeros(self.m)
        for T, C in self.counts.items():
            P = expm(T * Q)
            Gm = np.zeros_like(P)
            mask = C > 0
            Gm[mask] = C[mask] / np.maximum(P[mask], self.GFLOOR)
            if not np.all(np.isfinite(Gm)):
                return np.zeros(self.m)
            Z = expm_frechet((T * Q).T, Gm, compute_expm=False)
            for j, (al, be) in enumerate(self.S):
                g[j] += T * (Z[al, be] - Z[al, al])
        return g

    def R(self, w):
        return float(self.Wc @ np.abs(self.Ac @ w)
                     + self.Wd @ np.abs(self.Ad @ w))


# ======================================================================
# 5. Optimization
# ======================================================================
def maximum_likelihood(prob, w_ref, n_restarts=4, seed=1):
    """Unconstrained maximum likelihood in log-rate coordinates w = exp(v)."""
    rr = np.random.default_rng(seed)
    f = lambda v: -prob.loglik(np.exp(v))
    gf = lambda v: -prob.loglik_grad(np.exp(v)) * np.exp(v)
    starts = [np.zeros(prob.m), np.log(np.maximum(w_ref, 1e-3))]
    starts += [rr.normal(size=prob.m) for _ in range(max(0, n_restarts - 2))]
    best = None
    for v0 in starts:
        r = minimize(f, v0, jac=gf, method="L-BFGS-B",
                     options=dict(maxiter=20000, maxfun=50000,
                                  ftol=1e-18, gtol=1e-14))
        if best is None or -r.fun > best[0]:
            best = (-r.fun, np.exp(r.x), float(np.linalg.norm(r.jac)))
    return best


def solve_constrained(prob, w0, log_p, maxiter=300):
    """
    Minimize R_Q subject to w >= 0 and l(Q(w)) >= log_p, using the exact
    epigraph reformulation
        min  Wc.t + Wd.s   s.t.  t >= |Ac w|,  s >= |Ad w|,  l(Q(w)) >= log_p.
    The absolute values are represented exactly and are not smoothed.
    """
    m, n = prob.m, prob.n
    nc, nd = n * n, n
    Ac, Ad, Wc, Wd = prob.Ac, prob.Ad, prob.Wc, prob.Wd

    def split(z):
        return z[:m], z[m:m + nc], z[m + nc:]

    f = lambda z: float(Wc @ split(z)[1] + Wd @ split(z)[2])
    gradf = np.concatenate([np.zeros(m), Wc, Wd])

    Zt = np.zeros((nc, nd))
    Zs = np.zeros((nd, nc))
    Lin = np.block([
        [-Ac, np.eye(nc), Zt],
        [Ac, np.eye(nc), Zt],
        [-Ad, Zs, np.eye(nd)],
        [Ad, Zs, np.eye(nd)],
    ])
    cons = [
        {"type": "ineq", "fun": lambda z: Lin @ z, "jac": lambda z: Lin},
        {"type": "ineq",
         "fun": lambda z: np.array([prob.loglik(split(z)[0]) - log_p]),
         "jac": lambda z: np.concatenate(
             [prob.loglik_grad(split(z)[0]), np.zeros(nc + nd)])[None, :]},
    ]
    bounds = [(0.0, None)] * (m + nc + nd)
    z0 = np.concatenate([w0, np.abs(Ac @ w0), np.abs(Ad @ w0)])
    res = minimize(f, z0, jac=lambda z: gradf, bounds=bounds,
                   constraints=cons, method="SLSQP",
                   options=dict(maxiter=maxiter, ftol=1e-9))
    return split(res.x)[0], res


# ======================================================================
# 6. Verification
# ======================================================================
def check_frame_identities(fr, rng, log):
    n, D, G, M = fr["n"], fr["D"], fr["G"], fr["M"]
    mask = D != 0
    log("Frame identities")
    log("  |D_ab| = 1 on the incidence support   : %s"
        % np.allclose(np.abs(D[mask]), 1.0))
    A = rng.normal(size=(n, n)) * mask
    FA = sum(np.sum(A * G[mu, nu]) * G[mu, nu]
             for mu in range(n) for nu in range(n))
    log("  frame operator equals identity, error : %.3e" % np.abs(FA - A).max())
    gram = np.array([[np.sum(M[i] * M[j]) for j in range(n)] for i in range(n)])
    log("  multipliers orthonormal, ||Gram - I||  : %.3e"
        % np.abs(gram - np.eye(n)).max())
    cross = max(abs(np.sum(G[mu, nu] * M[l]))
                for mu in range(n) for nu in range(n) for l in range(n))
    log("  <G_{mu nu}, M_lambda> = 0, maximum     : %.3e" % cross)


def check_representation(Q, fr, tag, log):
    c, d = coeff_c(Q, fr), coeff_d(Q, fr)
    Qr = synthesize(c, d, fr)
    log("  %-6s reconstruction error (absolute) : %.3e"
        % (tag, np.abs(Qr - Q).max()))
    log("  %-6s reconstruction error (relative) : %.3e"
        % (tag, np.linalg.norm(Qr - Q) / np.linalg.norm(Q)))
    log("  %-6s Parseval residual               : %.3e"
        % (tag, abs((c ** 2).sum() + (d ** 2).sum() - (Q ** 2).sum())))
    return c, d


def check_generator(Q, tag, log):
    off = Q - np.diag(np.diag(Q))
    log("  %-6s minimum off-diagonal entry      : %.3e" % (tag, off.min()))
    log("  %-6s maximum |row sum|               : %.3e"
        % (tag, np.abs(Q.sum(axis=1)).max()))
    return off.min(), np.abs(Q.sum(axis=1)).max()


def check_norm_bounds(Q, c, d, fr, tag, log):
    R = regularizer(c, d, fr)
    kappa = min(fr["a"] / fr["n"], fr["n"] ** -0.5)
    hs = np.linalg.norm(Q)
    log("  %-6s R_Q                             : %.6f" % (tag, R))
    log("  %-6s ||Q||_{1,1} = %8.4f  <= R_Q    : %s"
        % (tag, op_norm_11(Q), op_norm_11(Q) <= R + 1e-9))
    log("  %-6s ||Q||_{inf,inf} = %8.4f <= R_Q : %s"
        % (tag, op_norm_inf(Q), op_norm_inf(Q) <= R + 1e-9))
    log("  %-6s R_Q >= kappa ||Q||_HS = %8.4f  : %s"
        % (tag, kappa * hs, R >= kappa * hs - 1e-9))
    return R


def check_semigroup(Q, tag, log, times=(0.25, 0.5, 1.0, 2.0, 5.0)):
    """A conservative generator must produce stochastic matrices."""
    worst_neg, worst_row = np.inf, 0.0
    for T in times:
        P = expm(T * Q)
        worst_neg = min(worst_neg, float(P.min()))
        worst_row = max(worst_row, np.abs(P.sum(axis=1) - 1.0).max())
    ev = np.linalg.eigvals(Q)
    order = np.argsort(-ev.real)
    log("  %-6s minimum kernel entry over t      : %.3e" % (tag, worst_neg))
    log("  %-6s max |row sum - 1| over t         : %.3e" % (tag, worst_row))
    log("  %-6s max Re spectrum                  : %.3e" % (tag, ev.real.max()))
    log("  %-6s second Re eigenvalue (gap)       : %.4f"
        % (tag, ev.real[order[1]]))
    return worst_neg, worst_row, ev


def check_stability(Q, Qh, log, T=2.0, n_grid=41, delta=1e-3, seed=5):
    """
    Verify the stability bound
        ||p0 e^{tQ} - p0t e^{t Qh}||_1 <= delta e^{Kt} + (eps/K)(e^{Kt} - 1)
    with K a Lipschitz constant for p -> pQ and eps a bound for the difference
    of the two vector fields on the region of interest. For row vectors in the
    1-norm, ||pA||_1 <= ||p||_1 ||A||_{inf,inf}, so K = ||Q||_{inf,inf} and
    eps = ||Q - Qh||_{inf,inf}. The bound is increasing in K, so replacing K by
    any upper bound, in particular by R_Q, keeps it valid.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    p0 = rng.dirichlet(np.ones(n))
    pert = rng.normal(size=n)
    pert -= pert.mean()
    pert = pert / np.abs(pert).sum() * delta
    p0t = p0 + pert
    d0 = np.abs(p0 - p0t).sum()
    Kc = op_norm_inf(Q)
    eps = op_norm_inf(Q - Qh)
    ts = np.linspace(0.0, T, n_grid)
    lhs = np.array([np.abs(p0 @ expm(t * Q) - p0t @ expm(t * Qh)).sum()
                    for t in ts])
    rhs = d0 * np.exp(Kc * ts) + (eps / Kc) * (np.exp(Kc * ts) - 1.0)
    log("Stability bound")
    log("  ||p0 - p0_tilde||_1 = delta            : %.3e" % d0)
    log("  K = ||Q||_{inf,inf}                    : %.4f" % Kc)
    log("  eps = ||Q - Qhat||_{inf,inf}           : %.4e" % eps)
    log("  bound holds on the whole grid          : %s"
        % bool(np.all(lhs <= rhs + 1e-12)))
    slack = rhs - lhs
    log("  minimum slack over the grid            : %.3e  (attained at t = %.2f)"
        % (slack.min(), ts[int(np.argmin(slack))]))
    log("  the bound is tight at t = 0, where both sides equal delta")
    log("  minimum slack for t > 0                : %.3e" % slack[1:].min())
    log("  ratio lhs/rhs at t = %.2f              : %.3e"
        % (T, lhs[-1] / rhs[-1]))
    return ts, lhs, rhs, Kc, eps, d0


def check_stability_with_R(Q, Qh, R, ts, lhs, eps, d0, log):
    """The same bound with the computable constant R_Q in place of K."""
    rhs_R = d0 * np.exp(R * ts) + (eps / R) * (np.exp(R * ts) - 1.0)
    log("  R_Q as Lipschitz constant, R_Q >= K    : %s" % (R >= op_norm_inf(Q)))
    log("  bound with R_Q holds on the grid       : %s"
        % bool(np.all(lhs <= rhs_R + 1e-12)))
    return rhs_R


# ======================================================================
# 7. Figures
# ======================================================================
def _blocks(ax, nV):
    ax.axhline(nV - 0.5, color="w", lw=1.3)
    ax.axvline(nV - 0.5, color="w", lw=1.3)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def fig_graph(geo, theta=THETA_TRUE):
    """Standalone diagram of the logistic birth-death ladder on Gamma.

    Vertices 0..K sit on a line; birth transitions i -> i+1 arc above and
    death transitions i -> i-1 arc below, each colored by its rate so the
    logistic hump in the birth rate is visible. Edge-states e_j are drawn as
    small diamonds on the connecting segments to expose the V u E structure
    that the Dirac operator lifts. Kept separate from the matrix figure.
    """
    K, nV = geo["K"], geo["nV"]
    r, d, a, kL, kR = theta

    births = np.array([a + r * i * (1.0 - i / K) for i in range(K)])   # i->i+1
    deaths = np.array([d * i for i in range(1, nV)])                   # i->i-1
    rmax = max(births.max(), deaths.max())
    cmap = plt.get_cmap(MAG)
    norm = plt.Normalize(0.0, rmax)

    xs = np.arange(nV, dtype=float)
    fig, ax = plt.subplots(figsize=(1.35 * nV + 1.6, 3.7),
                           constrained_layout=True)

    ax.plot(xs, np.zeros_like(xs), color="0.82", lw=1.2, zorder=0)

    for i in range(K):                                   # births, above
        ax.add_patch(FancyArrowPatch(
            (xs[i] + 0.17, 0.11), (xs[i + 1] - 0.17, 0.11),
            connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
            mutation_scale=13, lw=2.1, color=cmap(norm(births[i])), zorder=1))
    for i in range(1, nV):                               # deaths, below
        ax.add_patch(FancyArrowPatch(
            (xs[i] - 0.17, -0.11), (xs[i - 1] + 0.17, -0.11),
            connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
            mutation_scale=13, lw=2.1, color=cmap(norm(deaths[i - 1])), zorder=1))

    mids = 0.5 * (xs[:-1] + xs[1:])                      # edge-states e_j
    ax.scatter(mids, np.zeros_like(mids), marker="D", s=66, c="w",
               edgecolors="0.45", linewidths=1.1, zorder=2)
    for jj, m in enumerate(mids):
        ax.text(m, 0.0, rf"$e_{{{jj}}}$", ha="center", va="center",
                fontsize=6.2, color="0.4", zorder=3)

    ax.scatter(xs, np.zeros_like(xs), s=560, c="#2f6db0",
               edgecolors="k", linewidths=1.0, zorder=4)
    for i in xs:
        ax.text(i, 0.0, str(int(i)), color="w", ha="center", va="center",
                fontweight="bold", fontsize=10.5, zorder=5)

    ax.text(xs.mean(), 0.60, r"birth  $i \to i{+}1$",
            ha="center", color="0.3", fontsize=9)
    ax.text(xs.mean(), -0.60, r"death  $i \to i{-}1$",
            ha="center", color="0.3", fontsize=9)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_ylim(-0.85, 0.85)
    ax.axis("off")

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.032, pad=0.02)
    cb.set_label("transition rate", fontsize=9)
    ax.set_title(r"logistic ladder on $\Gamma = V \cup E$", fontsize=11, pad=6)
    fig.savefig(f"{OUT}/fig1_graph.png")
    plt.close(fig)


def fig_matrices(geo):
    """Signed incidence matrix B and the Dirac operator D (the matrices)."""
    nV, nE, B, D = geo["nV"], geo["nE"], geo["B"], geo["D"]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.7), constrained_layout=True)

    im = ax[0].imshow(B, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    ax[0].set_title(r"incidence  $B$", fontsize=10)
    ax[0].set_xlabel(r"edges $e_j$")
    ax[0].set_ylabel("vertices")
    ax[0].set_xticks(range(nE))
    ax[0].set_yticks(range(nV))
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)

    im = ax[1].imshow(D, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
    _blocks(ax[1], nV)
    ax[1].set_title(r"Dirac operator  $D$", fontsize=10)
    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel(r"$\alpha$")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)

    fig.savefig(f"{OUT}/fig1b_matrices.png")
    plt.close(fig)


def fig_spectrum(fr):
    lam, U, n = fr["lam"], fr["U"], fr["n"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    ax[0].scatter(np.arange(n), np.sort(lam), s=42, c="#c0431f",
                  edgecolors="k", linewidths=0.5, zorder=3)
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].set_xlabel("index")
    ax[0].set_ylabel(r"$\lambda_\mu$")
    _clean(ax[0])
    for k in np.argsort(np.abs(lam))[:4]:
        ax[1].plot(U[:, k], marker="o", ms=3.5, lw=1.4,
                   label=rf"$\lambda={lam[k]:.2f}$")
    ax[1].set_xlabel(r"$\alpha\in\Gamma$")
    ax[1].set_ylabel("amplitude")
    ax[1].legend(fontsize=8, frameon=False)
    _clean(ax[1])
    fig.savefig(f"{OUT}/fig2_spectrum_D.png")
    plt.close(fig)


def fig_counts(counts, geo):
    C = sum(counts.values())
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    im = ax.imshow(C, cmap=MAG)
    _blocks(ax, geo["nV"])
    ax.set_xlabel("end state")
    ax.set_ylabel("start state")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="count")
    fig.savefig(f"{OUT}/fig3_endpoint_counts.png")
    plt.close(fig)


def fig_operator(Q_true, Q_hat, geo):
    nV = geo["nV"]
    E = Q_hat - Q_true
    vmax = np.abs(np.concatenate([Q_true.ravel(), Q_hat.ravel()])).max()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for a, Mx, sig in zip(ax, [Q_true, Q_hat, E],
                          [vmax, vmax, np.abs(E).max() + 1e-12]):
        im = a.imshow(Mx, cmap=DIV, norm=TwoSlopeNorm(0, -sig, sig))
        _blocks(a, nV)
        a.set_xlabel(r"$\beta$")
        a.set_ylabel(r"$\alpha$")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    fig.savefig(f"{OUT}/fig4_operator_recovery.png")
    plt.close(fig)


def fig_kernels(Q_true, Q_hat, geo, times=TIMES):
    nV = geo["nV"]
    fig, ax = plt.subplots(len(times), 3, figsize=(10.5, 3.3 * len(times)),
                           constrained_layout=True)
    for r, T in enumerate(times):
        Pt, Ph = expm(T * Q_true), expm(T * Q_hat)
        row = ax[r] if len(times) > 1 else ax
        for a, Mx, diff in zip(row, [Pt, Ph, Ph - Pt], [False, False, True]):
            if diff:
                sig = np.abs(Ph - Pt).max() + 1e-12
                im = a.imshow(Mx, cmap=DIV, norm=TwoSlopeNorm(0, -sig, sig))
            else:
                im = a.imshow(Mx, cmap=POS, vmin=0, vmax=1)
            _blocks(a, nV)
            a.set_xlabel("end state")
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        row[0].set_ylabel(rf"$t={T}$", fontsize=11)
    fig.savefig(f"{OUT}/fig5_kernels.png")
    plt.close(fig)


def fig_frame_diagnostics(Q_hat, fr, geo):
    n, nV = fr["n"], geo["nV"]
    c, d = coeff_c(Q_hat, fr), coeff_d(Q_hat, fr)
    Qr = synthesize(c, d, fr)
    Res = Qr - Q_hat
    fig, ax = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    vg = np.abs(Q_hat).max()
    im = ax[0, 0].imshow(Qr, cmap=DIV, norm=TwoSlopeNorm(0, -vg, vg))
    _blocks(ax[0, 0], nV)
    ax[0, 0].set_xlabel(r"$\beta$")
    ax[0, 0].set_ylabel(r"$\alpha$")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.03)
    vr = np.abs(Res).max() + 1e-18
    im = ax[0, 1].imshow(Res, cmap=DIV, norm=TwoSlopeNorm(0, -vr, vr))
    _blocks(ax[0, 1], nV)
    ax[0, 1].set_xlabel(r"$\beta$")
    ax[0, 1].set_ylabel(r"$\alpha$")
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.03)
    vc = np.abs(c).max()
    im = ax[1, 0].imshow(c, cmap=DIV, norm=TwoSlopeNorm(0, -vc, vc))
    ax[1, 0].set_xlabel(r"$\nu$")
    ax[1, 0].set_ylabel(r"$\mu$")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.03)
    ax[1, 1].vlines(np.arange(n), 0, d, color="#2f6db0", lw=1)
    ax[1, 1].scatter(np.arange(n), d, s=36, c="#2f6db0",
                     edgecolors="k", linewidths=0.5)
    ax[1, 1].axhline(0, color="0.7", lw=0.8)
    ax[1, 1].set_xlabel(r"$\lambda$")
    ax[1, 1].set_ylabel(r"$d_\lambda$")
    _clean(ax[1, 1])
    fig.savefig(f"{OUT}/fig6_frame_diagnostics.png")
    plt.close(fig)


def fig_regularizer_spectrum(Q_hat, fr):
    c, d = coeff_c(Q_hat, fr), coeff_d(Q_hat, fr)
    ui = fr["u_inf"]
    W = fr["a"] * np.abs(c) * np.outer(ui, ui)
    ev = np.linalg.eigvals(Q_hat)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    im = ax[0].imshow(W, cmap=MAG)
    ax[0].set_xlabel(r"$\nu$")
    ax[0].set_ylabel(r"$\mu$")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
    ax[1].scatter(ev.real, ev.imag, s=42, c="#c0431f",
                  edgecolors="k", linewidths=0.5)
    ax[1].axvline(0, color="0.7", lw=0.8)
    ax[1].set_xlabel("real part")
    ax[1].set_ylabel("imaginary part")
    _clean(ax[1])
    fig.savefig(f"{OUT}/fig7_regularizer_spectrum.png")
    plt.close(fig)


def fig_stability(ts, lhs, rhs, rhs_R):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
    ax[0].plot(ts, lhs, lw=1.8, color="#20304d")
    ax[0].set_xlabel(r"$t$")
    ax[0].set_ylabel(r"$\|p(t)-\tilde p(t)\|_1$")
    ax[0].set_title("actual deviation", fontsize=10)
    _clean(ax[0])
    ax[1].semilogy(ts, np.maximum(lhs, 1e-18), lw=1.8, color="#20304d",
                   label="actual deviation")
    ax[1].semilogy(ts, rhs, lw=1.5, ls="--", color="#c0431f",
                   label=r"bound with $K=\|Q\|_{\infty,\infty}$")
    ax[1].semilogy(ts, rhs_R, lw=1.5, ls=":", color="#2f6db0",
                   label=r"bound with $R_Q$")
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel("deviation")
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")
    _clean(ax[1])
    fig.savefig(f"{OUT}/fig8_stability.png")
    plt.close(fig)


# ======================================================================
# 8. Driver
# ======================================================================
def run(n_per_time=N_PER_TIME, times=TIMES, alpha=ALPHA, seed=SEED,
        make_figures=True, out=OUT):
    global OUT
    OUT = out
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    rng = np.random.default_rng(seed)
    geo = build_geometry()
    fr = build_frame(geo["D"])
    S = support_pairs(geo["D"])

    log("=" * 72)
    log("DIRAC FRAME REPRESENTATION AND CONSTRAINED GENERATOR ESTIMATION")
    log("=" * 72)
    log("states n = %d, incidence support |S| = %d, frame elements = %d"
        % (fr["n"], len(S), fr["n"] ** 2 + fr["n"]))
    log("a = %.1f, seed = %d" % (fr["a"], seed))
    log()

    check_frame_identities(fr, rng, log)
    log()

    Q_true = generator(THETA_TRUE, geo)
    counts = endpoint_counts(Q_true, times, n_per_time, rng)
    N = int(sum(C.sum() for C in counts.values()))
    log("Data: %d endpoint pairs at horizons %s" % (N, list(times)))
    log()

    log("Reference generator")
    c_t, d_t = check_representation(Q_true, fr, "Q", log)
    check_generator(Q_true, "Q", log)
    R_true = check_norm_bounds(Q_true, c_t, d_t, fr, "Q", log)
    log()

    prob = Problem(geo, fr, counts)
    w_true = np.array([Q_true[a, b] for (a, b) in S])
    l_max, w_mle, gnorm = maximum_likelihood(prob, w_true)
    delta = 0.5 * chi2.ppf(1 - alpha, prob.m)
    log_p = l_max - delta
    lr = 2.0 * (l_max - prob.loglik(w_true))
    log("Likelihood level")
    log("  maximum log-likelihood                 : %.4f" % l_max)
    log("  gradient norm at the maximum           : %.2e" % gnorm)
    log("  delta = chi2_{%d,%.2f} / 2             : %.4f"
        % (prob.m, 1 - alpha, delta))
    log("  log p                                  : %.4f" % log_p)
    log("  2(l_max - l(Q)) = %.2f <= chi2 = %.2f  : %s"
        % (lr, chi2.ppf(1 - alpha, prob.m), lr <= chi2.ppf(1 - alpha, prob.m)))
    log()

    log("Constrained minimization of the regularizer")
    objs = []
    for name, w0 in (("maximum likelihood", w_mle),
                     ("unit rates", np.ones(prob.m)),
                     ("reference", w_true)):
        wj, rj = solve_constrained(prob, w0, log_p)
        objs.append((prob.R(wj), wj, rj, name))
        log("  start %-19s R = %10.6f  status %d" % (name, prob.R(wj), rj.status))
    objs.sort(key=lambda t: t[0])
    R_hat, w_hat, res, _ = objs[0]
    Q_hat = prob.Q(w_hat)
    log("  spread across starts                   : %.3e"
        % (objs[-1][0] - objs[0][0]))
    log()

    log("Estimated generator")
    c_h, d_h = check_representation(Q_hat, fr, "Qhat", log)
    check_generator(Q_hat, "Qhat", log)
    R_hat = check_norm_bounds(Q_hat, c_h, d_h, fr, "Qhat", log)
    ll_hat = prob.loglik(w_hat)
    log("  log-likelihood                         : %.4f" % ll_hat)
    log("  likelihood residual                    : %.3e" % (ll_hat - log_p))
    log("  regularizer decreased from reference   : %s  (%.6f vs %.6f)"
        % (R_hat <= R_true + 1e-9, R_hat, R_true))
    log()

    log("Markov semigroup generated by the estimate")
    check_semigroup(Q_hat, "Qhat", log)
    log()

    log("Accuracy")
    rel = np.linalg.norm(Q_hat - Q_true) / np.linalg.norm(Q_true)
    offmask = (geo["D"] == 0) & ~np.eye(fr["n"], dtype=bool)
    log("  relative Frobenius error               : %.4e" % rel)
    log("  maximum error off the support          : %.3e"
        % np.abs((Q_hat - Q_true)[offmask]).max())
    for T in times:
        log("  max kernel difference at t = %.2f       : %.4e"
            % (T, np.abs(expm(T * Q_hat) - expm(T * Q_true)).max()))
    log()

    ts, lhs, rhs, Kc, eps, d0 = check_stability(Q_true, Q_hat, log)
    rhs_R = check_stability_with_R(Q_true, Q_hat, R_hat, ts, lhs, eps, d0, log)
    log()

    if make_figures:
        fig_graph(geo)
        fig_matrices(geo)
        fig_spectrum(fr)
        fig_counts(counts, geo)
        fig_operator(Q_true, Q_hat, geo)
        fig_kernels(Q_true, Q_hat, geo, times)
        fig_frame_diagnostics(Q_hat, fr, geo)
        fig_regularizer_spectrum(Q_hat, fr)
        fig_stability(ts, lhs, rhs, rhs_R)
        log("Figures written to %s"
            % os.path.basename(os.path.normpath(OUT)))
        with open(os.path.join(OUT, "report.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    return dict(geo=geo, fr=fr, prob=prob, Q_true=Q_true, Q_hat=Q_hat,
                R_true=R_true, R_hat=R_hat, stability=(ts, lhs, rhs, rhs_R))


if __name__ == "__main__":
    run()
