"""Independent NumPy references and evidence helpers for FortBO."""

from .reference import (
    FUNCTION_SPECS,
    acquisition_values,
    check_gradient,
    gp_posterior,
    gp_posterior_derivative_observations,
)

__all__ = [
    "FUNCTION_SPECS",
    "acquisition_values",
    "check_gradient",
    "gp_posterior",
    "gp_posterior_derivative_observations",
]
