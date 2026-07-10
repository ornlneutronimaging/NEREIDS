"""Show when broadening transmission differs from broadening both count arms.

The response matrix is conditioned on true bins (columns sum to one).  The
physically measured ratio is R(phi*T)/R(phi), whereas the current model uses
R(T).  A flat incident spectrum is the equality control.
"""

from __future__ import annotations

import json

import numpy as np


def main() -> None:
    response = np.array(
        [
            [0.80, 0.15, 0.00, 0.00, 0.00],
            [0.20, 0.70, 0.15, 0.00, 0.00],
            [0.00, 0.15, 0.70, 0.15, 0.00],
            [0.00, 0.00, 0.15, 0.70, 0.20],
            [0.00, 0.00, 0.00, 0.15, 0.80],
        ],
        dtype=float,
    )
    transmission = np.array([1.0, 1.0, 0.15, 1.0, 1.0])

    def comparison(flux: np.ndarray) -> dict[str, object]:
        direct = (response @ transmission) / (response @ np.ones(5))
        count_ratio = (response @ (flux * transmission)) / (response @ flux)
        delta = count_ratio - direct
        return {
            "direct_broadened_transmission": direct.tolist(),
            "count_response_ratio": count_ratio.tolist(),
            "delta": delta.tolist(),
            "max_abs_delta": float(np.max(np.abs(delta))),
        }

    print(
        json.dumps(
            {
                "response_column_sums": response.sum(axis=0).tolist(),
                "flat_flux_control": comparison(np.ones(5)),
                "structured_flux": comparison(
                    np.array([1.0, 0.55, 2.5, 0.8, 1.4], dtype=float)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
