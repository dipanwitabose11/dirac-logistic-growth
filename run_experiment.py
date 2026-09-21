#!/usr/bin/env python3
"""Run the full estimation pipeline and write every artifact.

Reproduces the numbers reported in ``results/summary.json`` with the default
arguments. A smaller ``--n-per-time`` runs faster but will not reproduce the
reported values, which are tied to seed 123 and 200000 pairs per horizon.

    python scripts/run_experiment.py
    python scripts/run_experiment.py --n-per-time 5000 --no-figures
"""
import argparse
import csv
import io
import json
import contextlib
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dirac_frame as df  # noqa: E402
from dirac_frame import pipeline as dl  # noqa: E402


def write_counts(counts, n, path):
    """Aggregated endpoint counts, the sufficient statistic of the likelihood."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "start_state", "end_state", "count"])
        for t in sorted(counts):
            c = counts[t]
            for x in range(n):
                for y in range(n):
                    w.writerow([f"{t:g}", x, y, int(c[x, y])])


def build_summary(out, report, n_total, figures):
    geo, fr, prob = out["geo"], out["fr"], out["prob"]
    qt, qh = out["Q_true"], out["Q_hat"]
    n = fr["n"]
    err = qh - qt
    c, d = dl.coeff_c(qh, fr), dl.coeff_d(qh, fr)
    ui, a = fr["u_inf"], fr["a"]
    off = float(a * np.sum(np.abs(c) * np.outer(ui, ui)))
    dia = float(np.sum(np.abs(d) * ui))
    ev = np.linalg.eigvals(qh)
    order = np.argsort(-ev.real)
    ts, lhs, rhs, rhs_r = out["stability"]
    kappa = min(a / n, n ** -0.5)
    off_support = (geo["D"] == 0) & ~np.eye(n, dtype=bool)

    summary = {
        "run": {
            "seed": dl.SEED,
            "n_states": n,
            "incidence_support": prob.m,
            "frame_elements": n * n + n,
            "a_norm_of_D": float(a),
            "endpoint_pairs": n_total,
            "horizons": list(dl.TIMES),
        },
        "frame_identities": {"dirac_unimodular_on_support": True},
        "reference_generator": {
            "R_Q": float(out["R_true"]),
            "norm_1_1": dl.op_norm_11(qt),
            "norm_inf_inf": dl.op_norm_inf(qt),
            "kappa_times_HS": float(kappa * np.linalg.norm(qt)),
        },
        "likelihood": {},
        "estimated_generator": {
            "R_Q": float(out["R_hat"]),
            "R_Q_off_diagonal_part": off,
            "R_Q_diagonal_part": dia,
            "norm_1_1": dl.op_norm_11(qh),
            "norm_inf_inf": dl.op_norm_inf(qh),
            "kappa_times_HS": float(kappa * np.linalg.norm(qh)),
            "min_off_diagonal_entry": float((qh - np.diag(np.diag(qh))).min()),
            "max_abs_row_sum": float(np.abs(qh.sum(1)).max()),
        },
        "accuracy": {
            "relative_frobenius_error":
                float(np.linalg.norm(err) / np.linalg.norm(qt)),
            "relative_error_1_1":
                dl.op_norm_11(err) / dl.op_norm_11(qt),
            "relative_error_inf_inf":
                dl.op_norm_inf(err) / dl.op_norm_inf(qt),
            "absolute_error_1_1": dl.op_norm_11(err),
            "absolute_error_inf_inf": dl.op_norm_inf(err),
            "max_entrywise_error": float(np.abs(err).max()),
            "max_error_off_support": float(np.abs(err[off_support]).max()),
            "max_kernel_difference": {
                f"t={t:g}": float(
                    np.abs(dl.expm(t * qh) - dl.expm(t * qt)).max())
                for t in dl.TIMES
            },
        },
        "spectrum_of_estimate": {
            "max_real_part": float(ev.real[order[0]]),
            "spectral_gap": float(abs(ev.real[order[1]])),
            "most_negative_eigenvalue": float(ev.real.min()),
            "max_abs_imaginary_part": float(np.abs(ev.imag).max()),
        },
        "stability_bound": {
            "delta": float(np.abs(lhs[0])),
            "K_norm_inf_inf_of_Q": dl.op_norm_inf(qt),
            "epsilon_norm_inf_inf_of_difference": dl.op_norm_inf(err),
            "bound_holds_with_K": bool(np.all(lhs <= rhs + 1e-12)),
            "bound_holds_with_R_Q": bool(np.all(lhs <= rhs_r + 1e-12)),
            "min_slack_positive_t": float((rhs - lhs)[1:].min()),
            "ratio_lhs_over_rhs_at_t_2": float(lhs[-1] / rhs[-1]),
        },
        "figures": figures,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
    }
    for line in report.splitlines():
        if "frame operator equals identity" in line:
            summary["frame_identities"]["frame_operator_minus_identity"] = \
                float(line.split(":")[1])
        elif "maximum log-likelihood" in line:
            summary["likelihood"]["max_loglik"] = float(line.split(":")[1])
        elif line.strip().startswith("log p"):
            summary["likelihood"]["log_p_level"] = float(line.split(":")[1])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=dl.SEED)
    ap.add_argument("--n-per-time", type=int, default=dl.N_PER_TIME)
    ap.add_argument("--alpha", type=float, default=dl.ALPHA)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = df.run(n_per_time=args.n_per_time, alpha=args.alpha,
                     seed=args.seed, make_figures=not args.no_figures,
                     out=str(args.figures))
    report = buf.getvalue()
    print(report, end="")

    prob, n = out["prob"], out["fr"]["n"]
    n_total = int(sum(c.sum() for c in prob.counts.values()))

    (args.results / "report.txt").write_text(report)
    write_counts(prob.counts, n, args.results / "endpoint_counts.csv")

    config = {
        "K": dl.K_STATES,
        "n_states": n,
        "n_vertices": out["geo"]["nV"],
        "n_edges": out["geo"]["nE"],
        "incidence_support_size": prob.m,
        "frame_elements": n * n + n,
        "theta_true": dict(zip(
            ("r", "d", "a", "kappa_left", "kappa_right"), dl.THETA_TRUE)),
        "times": list(dl.TIMES),
        "n_obs_per_time": args.n_per_time,
        "n_obs_total": n_total,
        "start_distribution_mode": "uniform_on_Gamma",
        "likelihood_ratio_alpha": args.alpha,
        "random_seed": args.seed,
        "estimator":
            "constrained minimisation of the Dirac frame regularizer R_Q",
        "decision_variable":
            "off-diagonal rates on the incidence support",
        "optimizer": {
            "likelihood": "L-BFGS-B in log-rate coordinates, 4 starts",
            "constrained": "SLSQP on the exact epigraph reformulation",
            "starts": ["maximum likelihood", "unit rates", "reference"],
        },
    }
    (args.results / "config.json").write_text(
        json.dumps(config, indent=2) + "\n")
    figures = sorted(p.name for p in args.figures.glob("*.png"))
    (args.results / "summary.json").write_text(
        json.dumps(build_summary(out, report, n_total, figures), indent=2) + "\n")

    np.savez(args.results / "generators.npz",
             Q_true=out["Q_true"], Q_hat=out["Q_hat"])
    print(f"Artifacts written to {args.results}")


if __name__ == "__main__":
    main()
