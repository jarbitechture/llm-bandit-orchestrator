"""Student-t quantile approximation (no scipy needed)."""

import math


def t_quantile(df: float, p: float) -> float:
    """Approximate inverse CDF of Student-t distribution.

    Uses the Abramowitz & Stegun rational approximation for the normal,
    then Cornish-Fisher correction for Student-t.
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0,1), got {p}")

    # Normal quantile via Abramowitz & Stegun 26.2.23
    if p < 0.5:
        sign = -1
        p_work = 1 - 2 * p
    else:
        sign = 1
        p_work = 2 * p - 1

    # Rational approximation
    if p_work < 1e-10:
        z = 0.0
    else:
        t_val = math.sqrt(-2 * math.log((1 - p_work) / 2))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t_val - (c0 + c1 * t_val + c2 * t_val ** 2) / (
            1 + d1 * t_val + d2 * t_val ** 2 + d3 * t_val ** 3
        )

    z *= sign

    # Cornish-Fisher expansion: convert normal quantile to Student-t
    if df >= 1000:
        return z
    g1 = (z ** 3 + z) / (4 * df)
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    return z + g1 + g2
