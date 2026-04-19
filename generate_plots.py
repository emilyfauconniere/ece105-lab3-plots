"""Utilities for generating and plotting synthetic sensor data.

This module provides generate_data(seed) which reproduces the synthetic
temperature readings used in the notebook.
"""

from typing import Tuple, Union
import numpy as np


def generate_data(seed: Union[int, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic temperature readings for two sensors.

    Parameters
    ----------
    seed : int or str
        Seed for the random number generator. If a string like ``"0707"`` is
        provided it will be converted to the integer ``707`` (leading zeros are
        not preserved in Python integer literals but are supported here as a
        string input).

    Returns
    -------
    timestamps : numpy.ndarray, shape (200,)
        Sorted timestamps uniformly sampled from the interval [0, 10] (seconds).

    sensor_a : numpy.ndarray, shape (200,)
        Simulated Sensor A temperature readings drawn from a normal
        distribution with mean 25.0 °C and standard deviation 3.0 °C.

    sensor_b : numpy.ndarray, shape (200,)
        Simulated Sensor B temperature readings drawn from a normal
        distribution with mean 27.0 °C and standard deviation 4.5 °C.

    Notes
    -----
    The function uses :func:`numpy.random.default_rng` for reproducible
    pseudo-random draws and sorts timestamps so they are strictly increasing,
    which matches typical time-series data expectations.
    """
    # Coerce seed to integer; strings like "0707" become 707
    try:
        seed_int = int(str(seed))
    except Exception:
        # Fallback: use Python's hash to produce a deterministic integer
        seed_int = abs(hash(seed)) % (2**32 - 1)

    rng = np.random.default_rng(seed_int)

    n = 200
    timestamps = np.sort(rng.uniform(0.0, 10.0, size=n))
    sensor_a = rng.normal(loc=25.0, scale=3.0, size=n)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=n)

    return timestamps, sensor_a, sensor_b


if __name__ == "__main__":
    # Quick sanity check when run as a script
    ts, a, b = generate_data("0707")
    print("timestamps (first 5):", ts[:5], "...", ts[-1])
    print("sensor_a mean/std:", round(a.mean(), 3), round(a.std(ddof=0), 3))
    print("sensor_b mean/std:", round(b.mean(), 3), round(b.std(ddof=0), 3))
