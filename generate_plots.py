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


def main(seed: Union[int, str] = '0707') -> None:
    """Generate synthetic data and create example plots in a 2×2 grid.

    Parameters
    ----------
    seed : int or str, optional
        Seed for the synthetic data RNG. Defaults to ``'0707'`` for
        reproducible output.

    Returns
    -------
    None

    Notes
    -----
    Creates a figure with a 2×2 grid of subplots: scatter (time vs temperature),
    overlapping histograms, side-by-side boxplots, and a summary cell
    containing basic statistics. Displays the figure with Matplotlib's
    interactive window.
    """
    import matplotlib.pyplot as plt

    ts, a, b = generate_data(seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax_scatter = axes[0, 0]
    ax_hist = axes[0, 1]
    ax_box = axes[1, 0]
    ax_summary = axes[1, 1]

    # scatter
    plot_scatter(ax_scatter, ts, a, b)

    # histogram
    plot_histogram(ax_hist, a, b, bins=30)

    # boxplot
    plot_boxplot(ax_box, a, b, showfliers=True)

    # summary: show simple statistics in the fourth panel
    mean_a = float(np.mean(a))
    std_a = float(np.std(a, ddof=0))
    mean_b = float(np.mean(b))
    std_b = float(np.std(b, ddof=0))

    ax_summary.axis('off')
    summary_text = (
        f"Sensor A:\n  mean = {mean_a:.2f} °C\n  std = {std_a:.2f} °C\n\n"
        f"Sensor B:\n  mean = {mean_b:.2f} °C\n  std = {std_b:.2f} °C"
    )
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()