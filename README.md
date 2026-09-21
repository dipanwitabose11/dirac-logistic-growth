# Generator estimation from endpoint-only data via a Dirac frame

Reconstructing the generator of a finite-state continuous-time Markov chain when
the data are severely censored: for every sampled path only the start `X(0)` and
the end `X(T)` are observed, never the trajectory in between.

Endpoint-only likelihoods are badly behaved. On a common horizon, if every
observed start state has the same empirical endpoint distribution, the likelihood
has a finite supremum that is approached only as the rates diverge, so no
maximiser exists. The fix here is a regularizer built from the geometry of the
graph rather than from the matrix entries, and the estimator minimises that
regularizer over the generators compatible with the data at a prescribed
likelihood level.

The model is a logistic birth and death process whose true generator is known in
closed form, so every claim below can be checked against ground truth.

## The construction

The state space carries both the population levels and the links between them:

```
V = {0, 1, ..., K}                        population levels
E = {e_0, ..., e_{K-1}},  e_i ~ (i, i+1)  edge states
Gamma = V u E,  n = |V| + |E| = 2K + 1
```

With `B` the signed incidence matrix, the graph Dirac operator is

```
D = [ 0    B  ]
    [ B^T  0  ]
```

Let `u_1, ..., u_n` be a real orthonormal eigenbasis of `D` and
`M_mu = diag(u_mu)`. The family

```
G_{mu,nu} = M_nu D M_mu        together with        M_lambda
```

is a **Parseval frame** for the space of graph-supported operators. Every
generator supported on the graph expands exactly, diagonal included:

```
Q         = sum_{mu,nu} c_{mu,nu} G_{mu,nu} + sum_lambda d_lambda M_lambda
c_{mu,nu} = Tr(M_nu Q M_mu D)
d_lambda  = Tr(M_lambda Q)
```

There is no Laplacian anywhere in this construction. The eigenvalues
`|lambda_mu|` act as graph frequencies, so smooth and oscillatory parts of the
dynamics separate.

## The estimator

The regularizer is a weighted `l1` functional of the frame coefficients,

```
R_Q = a * sum_{mu,nu} |c_{mu,nu}| ||u_mu||_inf ||u_nu||_inf
      +   sum_lambda  |d_lambda|  ||u_lambda||_inf,     a = ||D||_{1,1} = 2
```

It dominates the induced operator norms, `||Q||_{p,p} <= R_Q` for `p` in
`{1, inf}`, and it dominates a fixed multiple of the Hilbert-Schmidt norm, which
is what makes the constrained problem coercive. The estimator solves

```
minimise   R_Q
subject to Q_{alpha,beta} >= 0  for alpha != beta     (off-diagonal positivity)
           Q 1 = 0                                    (conservativity)
           l(Q) >= log p                              (likelihood floor)
```

The first two constraints are exactly the condition for `exp(tQ)` to be
stochastic for every `t >= 0`. The absolute values are handled by an exact
epigraph reformulation, not smoothed. The likelihood level is set from the
unconstrained maximum by `log p = l_max - chi2_{|S|, 0.95} / 2`; the quantile
only selects a level and no coverage statement is made anywhere.

The estimator is given no knowledge of the five-parameter family that generated
the data. It ranges over the full 24-dimensional set of graph-supported
conservative generators.

## Results

`K = 6`, so `n = 13` states and `|S| = 24` incidence pairs; 200000 endpoint pairs
per horizon at `t` in `{0.5, 1.0}`; seed 123.

**Frame identities**

| Quantity | Value |
|---|---|
| Frame operator minus identity on the support | 2.22e-15 |
| Multiplier Gram minus identity | 8.88e-16 |
| Cross terms `<G_{mu,nu}, M_lambda>` | 0 |
| Reconstruction of the estimate, relative | 9.61e-16 |

**Estimation**

| Quantity | Value |
|---|---|
| Maximum log-likelihood | -569631.0561 |
| Likelihood level `log p` | -569649.2636 |
| `R_Q` at the reference generator | 21.969216 |
| `R_Q` at the estimate | 21.548341 |
| Spread across three starts | 6.94e-12 |

All three starts (maximum likelihood, unit rates, reference generator) return the
same value, and the estimate lowers `R_Q` below the reference, as it must.

**Accuracy and structure**

| Quantity | Value |
|---|---|
| Relative Frobenius error | 1.81e-02 |
| Relative (1,1) / (inf,inf) error | 2.42e-02 / 2.49e-02 |
| Max kernel difference at `t = 0.5` / `1.0` | 5.16e-03 / 3.58e-03 |
| Error off the incidence support | exactly 0 |
| Minimum off-diagonal entry of the estimate | 0 |
| Max abs row sum of the estimate | 2.22e-16 |
| Spectral gap | 0.0382 |

The estimate is a bona fide Markov generator, its support is exactly the graph's,
and the stability bound for the transition kernel holds on the whole time grid,
both with `K = ||Q||_{inf,inf}` and with the computable constant `R_Q`.

## Layout

```
src/dirac_frame/        the package: geometry, frame, likelihood, optimisation
scripts/run_experiment.py   reproducible entry point; writes results/ and figures/
notebooks/              the same pipeline in one notebook, with output
tests/                  fast structural tests of the frame and generator facts
results/                summary.json, config.json, report.txt, endpoint_counts.csv,
                        generators.npz
figures/                the nine figures
```

`results/endpoint_counts.csv` holds the aggregated counts `(t, start, end, count)`
rather than 400000 individual pairs, because the counts are the sufficient
statistic the likelihood uses. The pairs regenerate deterministically from the
seed.

## Running it

```bash
pip install -r requirements.txt
python scripts/run_experiment.py
```

A few minutes end to end. For a quick look without reproducing the reported
values:

```bash
python scripts/run_experiment.py --n-per-time 5000 --no-figures
```

Tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Reproducibility

Reference environment for the numbers above: Python 3.11.15, numpy 2.4.4,
scipy 1.17.1, matplotlib 3.10.9.

Substantive quantities (`R_Q`, the errors, the spectral gap, the norms)
reproduce across environments. Diagnostics reported at the `1e-15` level are at
machine precision and will move with the BLAS and library versions; treat them as
bounds rather than as figures.

## License

Released under the MIT License. See [LICENSE](LICENSE).
