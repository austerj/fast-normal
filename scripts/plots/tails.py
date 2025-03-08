import numpy as np
from scipy import stats

from fastnorm.norm import mixture
from scripts import utils

XMIN, XMAX = -4.1, 4.1
YMIN, YMAX = -0.0005, 0.42
XMIN_TAIL = 2.4
YMIN_TAIL, YMAX_TAIL = 0.0, 0.024


def plot_density(
    m: int = 10,
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    show_ref: bool = False,
):
    npartitions = 2**m
    x = np.linspace(xmin, xmax, num=100)
    q = mixture.optimal_q(npartitions)
    f, ax = utils.subplots(figsize=(7, 3))
    # # plot density
    counts, bins = mixture.hist(npartitions, q)
    ax.stairs(counts, bins, color="C0", fill=True)
    # plot samples
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # plot reference density
    if show_ref:
        ax.plot(x, stats.norm.pdf(x), linestyle=":", color="C3", linewidth=2)
    f.tight_layout()
    return f


def plot_density_tails(ms: tuple[int, ...]):
    xmin, xmax = XMIN_TAIL, XMAX
    ymin, ymax = YMIN_TAIL, YMAX_TAIL
    f, axes = utils.subplots(1, len(ms), figsize=(7, 3), sharey="row")
    for m, ax in zip(ms, axes):
        npartitions = 2**m
        q = mixture.optimal_q(npartitions)
        # # plot density
        counts, bins = mixture.hist(npartitions, q)
        ax.stairs(counts, bins, color="C0", fill=True)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([2.4, 3.2, 4.0])
        # plot reference density
        # ax.plot(x, stats.norm.pdf(x), linestyle=":", color="C3", linewidth=2)
        ax.set_title(f"$N=2^{{{m}}}$")
    f.tight_layout()
    return f


if __name__ == "__main__":
    # plot full density for N=10
    utils.savefig(plot_density(), "density.pdf")

    # plot density tail closeup for N=10
    utils.savefig(plot_density(xmin=XMIN_TAIL, ymin=YMIN_TAIL, ymax=YMAX_TAIL, show_ref=True), "density_tail.pdf")

    # plot density tails for increasing N
    utils.savefig(plot_density_tails(ms=tuple(range(10, 15))), "density_tails.pdf")
