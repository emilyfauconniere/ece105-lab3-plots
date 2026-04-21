"""Utilities for generating and plotting synthetic sensor data.

This module provides generate_data(seed) which reproduces the synthetic
temperature readings used in the notebook, plus small plotting helpers.
"""

from typing import Tuple, Union
import numpy as np
from matplotlib.axes import Axes


def generate_data(seed: Union[int, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic temperature readings for two sensors.

    Parameters
    ----------
    seed : int or str
        Seed for the random number generator. Strings like ``"0707"`` are
        converted to the integer ``707`` so leading zeros are handled sensibly.

    Returns
    -------
    timestamps : numpy.ndarray, shape (200,)
        Sorted timestamps uniformly sampled from [0, 10] seconds.

    sensor_a : numpy.ndarray, shape (200,)
        Sensor A readings drawn from Normal(mean=25.0, std=3.0).

    sensor_b : numpy.ndarray, shape (200,)
        Sensor B readings drawn from Normal(mean=27.0, std=4.5).

    Notes
    -----
    Uses :func:`numpy.random.default_rng` for reproducible draws.
    """
    try:
        seed_int = int(str(seed))
    except Exception:
        seed_int = abs(hash(seed)) % (2**32 - 1)

    rng = np.random.default_rng(seed_int)
    n = 200
    timestamps = np.sort(rng.uniform(0.0, 10.0, size=n))
    sensor_a = rng.normal(loc=25.0, scale=3.0, size=n)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=n)
    return timestamps, sensor_a, sensor_b


def plot_scatter(ax: Axes, timestamps: np.ndarray, sensor_a: np.ndarray, sensor_b: np.ndarray) -> None:
    """Draw scatter traces for two sensors on the provided Axes.

    This function modifies the given Axes in-place and returns None. It is
    intended for use in scripts that create figures/axes externally, e.g.
    ``fig, ax = plt.subplots()``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to draw on (modified in place).

    timestamps : numpy.ndarray, shape (n,)
        1-D array of time values for the x-axis.

    sensor_a : numpy.ndarray, shape (n,)
        1-D array of Sensor A temperature readings.

    sensor_b : numpy.ndarray, shape (n,)
        1-D array of Sensor B temperature readings.

    Returns
    -------
    None

    Notes
    -----
    The function uses small marker sizes and alpha blending so overlapping
    points remain visible. It sets axis labels, a title, a legend, and a
    light grid.
    """
    # plot Sensor A
    ax.scatter(timestamps, sensor_a, s=36, alpha=0.85, marker='o', label='Sensor A', c='C0')
    # plot Sensor B
    ax.scatter(timestamps, sensor_b, s=36, alpha=0.85, marker='s', label='Sensor B', c='C1')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Synthetic Temperature Sensor Readings')
    ax.legend()
    ax.grid(alpha=0.3)
    # function modifies ax in place; return None explicitly
    return None


if __name__ == '__main__':
    # quick manual test
    import matplotlib.pyplot as plt

    ts, a, b = generate_data('0707')
    fig, ax = plt.subplots(figsize=(8,4))
    plot_scatter(ax, ts, a, b)
    plt.tight_layout()
    plt.show()
# Create plot_scatter(sensor_a, sensor_b, timestamps, ax) that draws
# the scatter plot from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.

# Create plot_histogram(sensor_a, sensor_b, timestamps, ax) that draws
# the histogram of temperature readings from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.

# Create plot_boxplot(sensor_a, sensor_b, timestamps, ax) that draws
# the box plot of temperature readings from the notebook onto the given Axes object.
# NumPy-style docstring. Modifies ax in place, returns None.

# Create main() that generates data, creates a 1x3 subplot figure,
# calls each plot function, adjusts layout, and saves as sensor_analysis.png
# at 150 DPI with tight bounding box.