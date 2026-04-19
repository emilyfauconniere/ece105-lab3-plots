"""Utilities for generating and plotting synthetic sensor data.

This module provides:
- generate_data(seed): create reproducible synthetic temperature readings
- plot_scatter(ax, ...): draw scatter traces onto an existing Axes
- plot_histogram(ax, ...): draw overlaid histograms onto an Axes
- plot_boxplot(ax, ...): draw side-by-side box plots onto an Axes

A simple CLI entrypoint saves the three figures to disk.
"""

from typing import Tuple, Union
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
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

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw on; modified in place.
    timestamps : numpy.ndarray, shape (n,)
        1-D array of time values for the x-axis.
    sensor_a : numpy.ndarray, shape (n,)
        1-D array of Sensor A temperature readings.
    sensor_b : numpy.ndarray, shape (n,)
        1-D array of Sensor B temperature readings.

    Returns
    -------
    None
        The function modifies ``ax`` in-place and returns nothing.
    """
    ax.scatter(timestamps, sensor_a, s=36, alpha=0.85, marker='o', label='Sensor A', c='C0')
    ax.scatter(timestamps, sensor_b, s=36, alpha=0.85, marker='s', label='Sensor B', c='C1')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Synthetic Temperature Sensor Readings')
    ax.legend()
    ax.grid(alpha=0.3)
    return None


def plot_histogram(ax: Axes, sensor_a: np.ndarray, sensor_b: np.ndarray, bins: int = 20) -> None:
    """Draw overlaid histograms for two sensors on the provided Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw on; modified in place.
    sensor_a : numpy.ndarray, shape (n,)
        Sensor A readings.
    sensor_b : numpy.ndarray, shape (n,)
        Sensor B readings.
    bins : int, optional
        Number of histogram bins (default: 20).

    Returns
    -------
    None
    """
    ax.hist(sensor_a, bins=bins, alpha=0.6, label='Sensor A', color='C0')
    ax.hist(sensor_b, bins=bins, alpha=0.6, label='Sensor B', color='C1')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Count')
    ax.set_title('Overlaid Histogram of Sensor Temperatures')
    ax.legend()
    ax.grid(alpha=0.3)
    return None


def plot_boxplot(ax: Axes, sensor_a: np.ndarray, sensor_b: np.ndarray) -> None:
    """Draw side-by-side box plots comparing the two sensors on the provided Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw on; modified in place.
    sensor_a : numpy.ndarray, shape (n,)
        Sensor A readings.
    sensor_b : numpy.ndarray, shape (n,)
        Sensor B readings.

    Returns
    -------
    None
    """
    bp = ax.boxplot([sensor_a, sensor_b], labels=['Sensor A', 'Sensor B'], patch_artist=True)
    colors = ['C0', 'C1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set_color('black')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Box Plot Comparison of Sensor Temperatures')
    ax.grid(alpha=0.3, axis='y')
    return None


def save_all(seed: Union[int, str], out_dir: str = '.', prefix: str = 'sensor') -> Tuple[str, str, str]:
    """Generate data and save scatter, histogram, and boxplot PNGs.

    Parameters
    ----------
    seed : int or str
        RNG seed passed to :func:`generate_data`.
    out_dir : str, optional
        Directory to save images into (created if necessary). Default is current dir.
    prefix : str, optional
        Filename prefix for saved images.

    Returns
    -------
    paths : tuple of str
        Paths to the saved (scatter_path, hist_path, boxplot_path).
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamps, sensor_a, sensor_b = generate_data(seed)

    # scatter
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_scatter(ax, timestamps, sensor_a, sensor_b)
    scatter_path = os.path.join(out_dir, f"{prefix}_scatter.png")
    fig.tight_layout()
    fig.savefig(scatter_path)
    plt.close(fig)

    # histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_histogram(ax, sensor_a, sensor_b)
    hist_path = os.path.join(out_dir, f"{prefix}_hist.png")
    fig.tight_layout()
    fig.savefig(hist_path)
    plt.close(fig)

    # boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_boxplot(ax, sensor_a, sensor_b)
    box_path = os.path.join(out_dir, f"{prefix}_boxplot.png")
    fig.tight_layout()
    fig.savefig(box_path)
    plt.close(fig)

    return scatter_path, hist_path, box_path


def main(argv=None):
    """Main entrypoint: parse CLI args, generate plots, and save PNGs.

    Parameters
    ----------
    argv : sequence of str or None
        Optional argument list to parse (like sys.argv[1:]). If ``None``, the
        real command-line arguments are used.

    Returns
    -------
    tuple of str
        Paths to the saved (scatter_path, hist_path, boxplot_path).
    """
    parser = argparse.ArgumentParser(description='Generate synthetic sensor plots and save PNGs.')
    parser.add_argument('--seed', default='0707', help='RNG seed (int or string).')
    parser.add_argument('--out-dir', default='.', help='Output directory for PNGs.')
    parser.add_argument('--prefix', default='sensor', help='Filename prefix for saved images.')
    args = parser.parse_args(argv)

    paths = save_all(args.seed, args.out_dir, args.prefix)
    return paths


if __name__ == '__main__':
    scatter_p, hist_p, box_p = main()
    print('Saved:', scatter_p, hist_p, box_p)
