"""Inject lightweight stubs to expose MCP single-spectrum route selection.

No fit is performed. The probe records which typed Python fitter and solver the
MCP manifest processor selects for raw-count inputs, including a misspelled
domain. A dummy FastMCP module avoids requiring the optional server dependency.
"""

import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class DummyFastMCP:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def tool(self, *_args: object, **_kwargs: object):
        return lambda function: function


fastmcp = types.ModuleType("fastmcp")
fastmcp.FastMCP = DummyFastMCP
sys.modules["fastmcp"] = fastmcp

from nereids.mcp import server as srv  # noqa: E402


calls: list[tuple[str, str]] = []


def result() -> SimpleNamespace:
    return SimpleNamespace(
        densities=np.array([1.0e-3]),
        uncertainties=np.array([1.0e-5]),
        reduced_chi_squared=1.0,
        deviance_per_dof=None,
        converged=True,
        iterations=1,
        temperature_k=293.6,
        anorm=1.0,
        background=[0.0, 0.0, 0.0],
    )


def fit_transmission(**kwargs: object) -> SimpleNamespace:
    calls.append(("transmission", str(kwargs.get("solver", "<python-default>"))))
    return result()


def fit_counts(**kwargs: object) -> SimpleNamespace:
    calls.append(("counts", str(kwargs.get("solver", "<python-default>"))))
    return result()


def main() -> None:
    srv.nereids.fit_spectrum_typed = fit_transmission
    srv.nereids.fit_counts_spectrum_typed = fit_counts
    srv._load_isotopes = lambda *_args, **_kwargs: ([('fake-isotope', 1.0e-3)], ['X-1'])

    cases = [
        ("no-fit-block", {}),
        ("kl-default-domain", {"solver": "kl"}),
        ("lm-explicit-counts", {"solver": "lm", "fit_domain": "counts"}),
        ("kl-domain-typo", {"solver": "kl", "fit_domain": "countz"}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        np.savez(
            base / "counts.npz",
            energies_ev=np.array([1.0, 2.0, 3.0]),
            sample_counts=np.array([80.0, 70.0, 60.0]),
            open_beam_counts=np.array([100.0, 100.0, 100.0]),
        )
        for label, fit in cases:
            manifest = {
                "frontmatter": {
                    "analysis": {
                        "data": {"path": "counts.npz", "kind": "counts_npz"},
                        "isotopes": ["X-1"],
                        "fit": fit,
                    }
                }
            }
            summary = srv._process_single_spectrum(base, manifest, base / f"out-{label}")
            print(label, calls[-1], Path(summary["output_npz"]).exists())


if __name__ == "__main__":
    main()
