"""Acceptance test for resonance thermometry on the April IPTS-37432 Ta foil.

Calibrate on the raw room-temperature pair, then read each heated pair within
10 K of its frame-thermocouple mean. Runs only with ``NEREIDS_ACCEPTANCE=1``,
then fails if data or workflow are missing; ``-s`` shows the readings.
"""

import glob
import math
import os

import pytest

if os.environ.get("NEREIDS_ACCEPTANCE") != "1":
    pytest.skip(
        "acceptance milestone only: set NEREIDS_ACCEPTANCE=1",
        allow_module_level=True,
    )

IPTS = "/SNS/VENUS/IPTS-37432"
MATERIAL = "Ta"
THICKNESS_MM = 0.127
TOLERANCE_K = 10.0

# (sample run, open-beam run, frame-thermocouple mean in K)
REFERENCE = (19383, 19385, 301.79)
SAMPLES = {
    "100C": (19387, 19388, 389.85),
    "200C": (19389, 19390, 502.06),
    "300C": (19391, 19392, 606.26),
    "400C": (19393, 19394, 708.30),
    "600C": (19395, 19396, 901.67),
    # Frame-thermocouple mean without the floating left thermocouple.
    "800C": (19397, 19398, 1093.0),
}


def _run(number):
    """Return the raw Timepix-1 folder and NeXus file of one run."""
    folders = glob.glob(f"{IPTS}/images/tpx1/*/**/*_Run_{number}_*", recursive=True)
    folders = [f for f in folders if os.path.isdir(f)]
    nexus = f"{IPTS}/nexus/VENUS_{number}.nxs.h5"
    assert len(folders) == 1, f"run {number}: expected one folder, found {folders}"
    assert os.path.isfile(nexus), f"run {number}: missing {nexus}"
    return folders[0], nexus


@pytest.fixture(scope="module")
def calibration():
    from nereids.thermometry import calibrate

    sample, open_beam, temperature_k = REFERENCE
    result = calibrate(
        sample=_run(sample),
        open_beam=_run(open_beam),
        material=MATERIAL,
        thickness_mm=THICKNESS_MM,
        temperature_k=temperature_k,
    )
    assert result.converged, "reference calibration did not converge"
    return result


@pytest.mark.parametrize("setpoint", list(SAMPLES))
def test_temperature_within_tolerance(calibration, setpoint):
    from nereids.thermometry import read_temperature

    sample, open_beam, key_k = SAMPLES[setpoint]
    reading = read_temperature(
        calibration,
        sample=_run(sample),
        open_beam=_run(open_beam),
    )
    assert reading.converged, f"{setpoint}: fit did not converge"
    assert math.isfinite(reading.temperature_k), f"{setpoint}: temperature is not finite"
    assert math.isfinite(reading.sigma_k) and reading.sigma_k > 0, f"{setpoint}: invalid uncertainty"
    residual = reading.temperature_k - key_k
    print(f"{setpoint}: {reading.temperature_k:.1f} ± {reading.sigma_k:.1f} K, key {key_k} K, residual {residual:+.1f} K")
    assert abs(residual) <= TOLERANCE_K, f"{setpoint}: residual {residual:+.1f} K exceeds {TOLERANCE_K} K"
