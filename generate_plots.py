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


def plot_histogram(ax: Axes, sensor_a: np.ndarray, sensor_b: np.ndarray, bins: int = 30, range=None) -> None:
    """Draw overlapping histograms for two sensors on the provided Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to draw on (modified in place).

    sensor_a : numpy.ndarray, shape (n,)
        Sensor A temperature readings.

    sensor_b : numpy.ndarray, shape (n,)
        Sensor B temperature readings.

    bins : int, optional
        Number of histogram bins (default is 30).

    range : tuple or None, optional
        The lower and upper range of the bins. If None, the range is inferred from the data.

    Returns
    -------
    None

    Notes
    -----
    Draws semi-transparent overlapping histograms with subtle edges so the
    two distributions can be compared. Sets axis labels, a title, a legend,
    and a light grid. Modifies the Axes in place.
    """
    if range is None:
        data_min = min(np.min(sensor_a), np.min(sensor_b))
        data_max = max(np.max(sensor_a), np.max(sensor_b))
        range = (data_min, data_max)

    ax.hist(sensor_a, bins=bins, range=range, alpha=0.6, color='C0', label='Sensor A', edgecolor='black', linewidth=0.5)
    ax.hist(sensor_b, bins=bins, range=range, alpha=0.6, color='C1', label='Sensor B', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Count')
    ax.set_title('Temperature Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    return None


def plot_boxplot(ax: Axes, sensor_a: np.ndarray, sensor_b: np.ndarray, showfliers: bool = True) -> None:
    """Draw side-by-side boxplots for two sensors on the provided Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to draw on (modified in place).

    sensor_a : numpy.ndarray, shape (n,)
        Sensor A temperature readings.

    sensor_b : numpy.ndarray, shape (n,)
        Sensor B temperature readings.

    showfliers : bool, optional
        Whether to show outlying points (fliers). Default is True.

    Returns
    -------
    None

    Notes
    -----
    Draws boxplots for Sensor A and Sensor B side-by-side, applies
    distinct colors (C0/C1) and a light grid. The function modifies the
    provided Axes in place and returns None.
    """
    data = [sensor_a, sensor_b]

    bp = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.6,
        patch_artist=True,
        showfliers=showfliers,
        medianprops={"color": "black"},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )

    colors = ["C0", "C1"]
    for patch, color in zip(bp.get('boxes', []), colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for whisker in bp.get('whiskers', []):
        whisker.set_color('black')
    for cap in bp.get('caps', []):
        cap.set_color('black')
    for median in bp.get('medians', []):
        median.set_color('black')
        median.set_linewidth(1.0)
    for flier in bp.get('fliers', []):
        flier.set_markerfacecolor('gray')
        flier.set_markeredgecolor('black')
        flier.set_markersize(4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Sensor A', 'Sensor B'])
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Distribution (boxplot)')
    ax.grid(axis='y', alpha=0.3)
    return None


def main(seed: Union[int, str] = '0707') -> None:
    """Generate synthetic data and create example plots.

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
    Creates a figure with three subplots: scatter (time vs temperature),
    overlapping histograms, and side-by-side boxplots. Displays the figure
    with Matplotlib's interactive window.
    """
    import matplotlib.pyplot as plt

    ts, a, b = generate_data(seed)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # scatter
    plot_scatter(axes[0], ts, a, b)

    # histogram
    plot_histogram(axes[1], a, b, bins=30)

    # boxplot
    plot_boxplot(axes[2], a, b, showfliers=True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
