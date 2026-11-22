#!/usr/bin/env python3
"""Print the Debye effective temperatures held by ``debye_effective_temperature.rs``.

``T_eff = (3/θ_D³) ∫₀^θ_D (x³/2) coth(x/2T) dx`` in kelvin, computed with mpmath at 30 digits
and printed to 12 significant digits.
Run ``uv run --no-project --with mpmath python integral.py``.
"""

from mpmath import coth, mp, mpf, nstr, quad

mp.dps = 30


def effective_temperature(theta: str, kelvin: str) -> mpf:
    """Return T_eff in K for Debye temperature ``theta`` and temperature ``kelvin``."""
    theta, kelvin = mpf(theta), mpf(kelvin)
    return 3 / theta**3 * quad(lambda x: x**3 / 2 * coth(x / (2 * kelvin)), [0, theta])


for theta in ("225", "240"):
    for kelvin in ("300", "301.79"):
        print(theta, kelvin, nstr(effective_temperature(theta, kelvin), 12))
