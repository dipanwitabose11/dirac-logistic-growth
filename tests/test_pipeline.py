"""Fast structural tests.

These run in seconds on a small graph and assert the mathematical facts the
estimator relies on. They deliberately do not run the full 400000-pair
experiment; that lives in ``scripts/run_experiment.py`` and is checked
against ``results/summary.json`` by ``test_reported_results_are_consistent``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dirac_frame as df  # noqa: E402
from dirac_frame import pipeline as dl  # noqa: E402

SMALL_K = 3


@pytest.fixture(scope="module")
def small():
    geo = df.build_geometry(SMALL_K)
    return geo, df.build_frame(geo["D"])


def test_geometry_shapes(small):
    geo, _ = small
    assert geo["nV"] == SMALL_K + 1
    assert geo["nE"] == SMALL_K
    assert geo["n"] == 2 * SMALL_K + 1
    assert geo["D"].shape == (geo["n"], geo["n"])


def test_dirac_is_symmetric_hollow_and_unimodular(small):
    geo, _ = small
    d = geo["D"]
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(np.abs(d[d != 0]), 1.0)


def test_frame_operator_is_the_identity_on_the_support(small):
    geo, fr = small
    n, d, g = fr["n"], geo["D"], fr["G"]
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, n)) * (d != 0)
    fa = sum(np.sum(a * g[mu, nu]) * g[mu, nu]
             for mu in range(n) for nu in range(n))
    assert np.abs(fa - a).max() < 1e-12


def test_multipliers_are_orthonormal_and_orthogonal_to_the_frame(small):
    _, fr = small
    n, m, g = fr["n"], fr["M"], fr["G"]
    gram = np.array([[np.sum(m[i] * m[j]) for j in range(n)] for i in range(n)])
    assert np.abs(gram - np.eye(n)).max() < 1e-12
    cross = max(abs(np.sum(g[mu, nu] * m[lam]))
                for mu in range(n) for nu in range(n) for lam in range(n))
    assert cross < 1e-12


def test_reconstruction_and_parseval(small):
    geo, fr = small
    q = df.generator((0.9, 0.25, 0.1, 2.0, 2.0), geo)
    c, d = df.coeff_c(q, fr), df.coeff_d(q, fr)
    assert np.abs(df.synthesize(c, d, fr) - q).max() < 1e-12
    assert abs((c ** 2).sum() + (d ** 2).sum() - (q ** 2).sum()) < 1e-10


def test_generator_is_conservative_and_off_diagonally_nonnegative(small):
    geo, _ = small
    q = df.generator((0.9, 0.25, 0.1, 2.0, 2.0), geo)
    assert np.abs(q.sum(axis=1)).max() < 1e-12
    off = q - np.diag(np.diag(q))
    assert off.min() >= 0.0


def test_semigroup_is_stochastic(small):
    geo, _ = small
    q = df.generator((0.9, 0.25, 0.1, 2.0, 2.0), geo)
    for t in (0.25, 1.0, 5.0):
        p = expm(t * q)
        assert p.min() > -1e-14
        assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-12


def test_regularizer_dominates_the_induced_operator_norms(small):
    geo, fr = small
    q = df.generator((0.9, 0.25, 0.1, 2.0, 2.0), geo)
    r = df.regularizer(df.coeff_c(q, fr), df.coeff_d(q, fr), fr)
    assert df.op_norm_11(q) <= r + 1e-9
    assert df.op_norm_inf(q) <= r + 1e-9


def test_regularizer_dominates_a_multiple_of_the_hs_norm(small):
    geo, fr = small
    q = df.generator((0.9, 0.25, 0.1, 2.0, 2.0), geo)
    r = df.regularizer(df.coeff_c(q, fr), df.coeff_d(q, fr), fr)
    kappa = min(fr["a"] / fr["n"], fr["n"] ** -0.5)
    assert r >= kappa * np.linalg.norm(q) - 1e-9


def test_regularizer_is_a_norm(small):
    geo, fr = small
    q = df.generator((0.9, 0.25, 0.1, 2.0, 2.0), geo)
    c, d = df.coeff_c(q, fr), df.coeff_d(q, fr)
    r = df.regularizer(c, d, fr)
    assert r > 0.0
    assert df.regularizer(-2.0 * c, -2.0 * d, fr) == pytest.approx(2.0 * r)
    assert df.regularizer(0.0 * c, 0.0 * d, fr) == pytest.approx(0.0)


def test_incidence_support_size(small):
    geo, _ = small
    assert len(df.support_pairs(geo["D"])) == 4 * SMALL_K


def test_reported_results_are_consistent():
    """The committed summary must agree with itself and with the theory."""
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    est, ref, acc = (s["estimated_generator"], s["reference_generator"],
                     s["accuracy"])
    assert est["norm_1_1"] <= est["R_Q"]
    assert est["norm_inf_inf"] <= est["R_Q"]
    assert est["R_Q"] >= est["kappa_times_HS"]
    assert est["R_Q"] <= ref["R_Q"], "the estimate must not increase R_Q"
    assert est["R_Q"] == pytest.approx(
        est["R_Q_off_diagonal_part"] + est["R_Q_diagonal_part"], rel=1e-9)
    assert est["min_off_diagonal_entry"] >= 0.0
    assert est["max_abs_row_sum"] < 1e-12
    assert acc["max_error_off_support"] == 0.0
    assert s["spectrum_of_estimate"]["max_real_part"] < 1e-12
    assert s["stability_bound"]["bound_holds_with_K"]
    assert s["stability_bound"]["bound_holds_with_R_Q"]


def test_committed_counts_match_the_committed_configuration():
    import csv
    import collections
    cfg = json.loads((ROOT / "results" / "config.json").read_text())
    totals = collections.Counter()
    with open(ROOT / "results" / "endpoint_counts.csv") as fh:
        for row in csv.DictReader(fh):
            totals[row["t"]] += int(row["count"])
    assert sum(totals.values()) == cfg["n_obs_total"]
    for t in cfg["times"]:
        assert totals[f"{t:g}"] == cfg["n_obs_per_time"]
