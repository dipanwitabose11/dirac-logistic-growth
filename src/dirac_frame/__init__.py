"""Dirac frame representation and constrained generator estimation.

Reconstruction of the generator of a finite-state continuous-time Markov
chain from endpoint-only observations, using a Parseval frame built from the
incidence Dirac operator of the underlying graph.

The public entry point is :func:`run`, which executes the whole pipeline:
it builds the augmented state space and the Dirac frame, verifies the frame
identities, samples censored endpoint data, fixes a likelihood level, solves
the constrained minimisation of the frame regularizer, and checks the
resulting estimate against the stability bound for the transition kernel.
"""

from .pipeline import (
    ALPHA,
    K_STATES,
    N_PER_TIME,
    SEED,
    THETA_TRUE,
    TIMES,
    build_frame,
    build_geometry,
    coeff_c,
    coeff_d,
    generator,
    op_norm_11,
    op_norm_inf,
    regularizer,
    run,
    support_pairs,
    synthesize,
)

__all__ = [
    "ALPHA",
    "K_STATES",
    "N_PER_TIME",
    "SEED",
    "THETA_TRUE",
    "TIMES",
    "build_frame",
    "build_geometry",
    "coeff_c",
    "coeff_d",
    "generator",
    "op_norm_11",
    "op_norm_inf",
    "regularizer",
    "run",
    "support_pairs",
    "synthesize",
]

__version__ = "1.0.0"
