# Logistic growth: generator recovery under heavy censoring

This is the simplest, fully controlled test case in my work on learning the
dynamics of continuous-time Markov chains (CTMCs) from censored data. By censored I
mean the hard setting I study throughout my research: for each sampled path I only
observe the start X(0) and the end X(T), never the trajectory in between. The point
of this repository is to take that hard setting and put it somewhere I can check my
own answers, so the model here is a one dimensional logistic birth and death process
whose true generator Q is known in closed form.

If the method cannot recover a generator I built myself, it has no business being
trusted on real data. So I start here.

## The model

I work on an augmented state space that carries both the population levels and the
links between them:

```
V = {0, 1, ..., K}                        population levels
E = {e_0, ..., e_{K-1}},  e_i ~ (i, i+1)  edges between adjacent levels
n = |V| + |E| = 2K + 1                     total dimension
```

The logistic dynamics enter through the rates

```
birth   i  -> e_i      at  lambda_i = a + r*i*(1 - i/K)   for i < K
death   i  -> e_{i-1}  at  mu_i     = d*i                 for i > 0
edge    e_i -> i       at  kappa_left
        e_i -> i+1     at  kappa_right
```

From endpoint pairs (x, y, t) I estimate the five parameters
(r, d, a, kappa_left, kappa_right) by maximum likelihood, then rebuild the full
generator and measure how close it is to the truth.

## Why a Dirac operator

The estimator is geometry aware. I represent generators in a basis built from the
graph's incidence Dirac structure rather than fitting entries blindly. With the
signed incidence matrix B, the graph Dirac operator is

```
D = [ 0    B  ]
    [ B^T  0  ]
```

Its orthonormal eigenvectors give a Parseval frame in which any generator supported
on the graph expands exactly:

```
Q_rec = sum_mu a_mu M_mu  +  sum_{mu != nu} b_{mu,nu} M_nu D M_mu,
M_mu = diag(u_mu),   a_mu = u_mu^T diag(Q),   b_{mu,nu} = (U^T (Q .* D) U)_{mu,nu}.
```

The eigenvalues |lambda_mu| play the role of graph frequencies, so smooth and
oscillatory parts of the dynamics separate cleanly. There is no Laplacian anywhere
in this construction.

## What it recovers

Run on synthetic endpoint data (K = 6, N = 2000 observations per time,
t in {0.5, 1.0}, seed 123):

| Parameter   | True  | Estimate |
|-------------|-------|----------|
| r           | 0.900 | 0.764    |
| d           | 0.250 | 0.266    |
| a           | 0.100 | 0.122    |
| kappa_left  | 2.000 | 1.864    |
| kappa_right | 2.000 | 2.246    |

```
Full generator relative error   ||Q_hat - Q_true||_F / ||Q_true||_F = 6.41e-02
Average log-likelihood (true)   -1.4225
Average log-likelihood (fit)    -1.4211   (meets the likelihood floor)
Theorem-style objective         (1/tau)*log||exp(tau*Q_hat) lambda||_TV = -0.2276
```

The pipeline keeps increasing the sample size until the generator relative error
drops below the target tolerance, which is the behavior I want to see before moving
to data where the truth is unknown.

## What is in here

```
logistic growith model.ipynb   the full pipeline: graph and Dirac build, data
                               generation, MLE fit, Dirac reconstruction, figures
config.json                    run configuration (model size, true theta, optimizer)
summary.json                   results in machine-readable form
README.txt                     raw run log
dataset.csv                    endpoint observations (x, y, t)
```

## Running it

```bash
pip install numpy scipy matplotlib
jupyter notebook "logistic growith model.ipynb"
```

Run all cells to reproduce the parameter recovery, the Dirac reconstruction, and the
diagnostic figures (transition kernels, total variation growth, the lambda
evolution, the Dirac coefficients, and the spectrum of Q_hat).

## Where this fits

This is one of four repositories that carry the same framework from a fully
controlled test case to noisy real world data:

- Logistic growth (this repo): clean synthetic validation with known parameters
- Predator and prey: real algae and rotifer ecology, Doob and Schrodinger bridges, error bounds
- Rotational vector field: analytic ground truth, recovery to machine precision
- Yellow cab dynamics: real NYC taxi mobility with no known generator

The censored-data estimation studied here also underlies a paper I am preparing,
"Maximum likelihood estimation of Markov processes with censored data: the Ehrenfest
model and beyond."

## License

Released under the MIT License. See [LICENSE](LICENSE).
