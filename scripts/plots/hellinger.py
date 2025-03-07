import itertools
import json
import math
import typing

import numpy as np
from scipy import stats

from fastnorm.norm import mixture
from scripts import utils


def _regress(y: np.ndarray) -> float:
    regressors = np.vstack([np.ones(len(y))], dtype=np.float64)
    return typing.cast(float, np.linalg.lstsq(regressors.T, y)[0][0])


def plot_distance(ms: tuple[int, ...]):
    f, ax = utils.subplots()
    qs = np.geomspace(0.75, 0.9999999999999, 2000)
    for i, npartitions in enumerate([2**m for m in ms]):
        hdists = np.array([mixture.hellinger(q, npartitions) for q in qs][:-1] + [0.3])
        q_opt = mixture.minimize_hellinger(npartitions).x
        zorder = len(ms) - i
        ax.plot(qs, hdists, label=f"N={npartitions}", color=f"C{i}", zorder=zorder)
        ax.plot(q_opt, mixture.hellinger(q_opt, npartitions), color=f"C{i}", marker="o", zorder=zorder)
        ax.set_ylim(0, 0.3)
    ax.set_xlabel("q")
    ax.set_title("Hellinger distance")
    ax.legend(loc="lower left", ncol=3)
    f.tight_layout()
    return f


def plot_optimum(mmin: int, mmax: int, nsteps: int = 64):
    f, ax = utils.subplots()
    # minimize Hellinger distance for each N
    npartitions = np.unique(np.geomspace(2**mmin, 2**mmax, num=nsteps, dtype=np.uint64))
    qs = [mixture.minimize_hellinger(n).x for n in npartitions]
    log_qs = np.log([1 - q for q in qs])
    # c ~= log(1-q) + log(N)  <=>  q ~= 1 - exp(c) / N
    c = _regress(log_qs + np.log(npartitions))
    print("regression inverse factor:  ", json.dumps(1 / math.exp(c)))
    ## regression inverse factor:   26.679816319102272
    # NOTE: rounding down so graph matches the numbers written in the paper
    inverse_factor = round(1 / math.exp(c), 2)
    qs_approx = lambda n: 1 - 1 / (inverse_factor * n)
    log_qs_approx = lambda n: np.log(1 - qs_approx(n))
    print("q approximation rule:       ", r"$1-\frac{1}{" + str(inverse_factor) + r"N}$")
    ## q approximation rule: $1-\frac{1}{26.67N}$
    # plot optimums and approximation from rule
    npartitions_ext = np.unique(np.geomspace(2**mmin, 2**32, num=nsteps, dtype=np.uint64))
    ax.plot(npartitions, log_qs, color="C0", label="optimum")
    ax.plot(npartitions_ext, log_qs_approx(npartitions_ext), color="C1", label="approximation", linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_title("log(1-q) for optimal q given N")
    ax.legend()
    f.tight_layout()
    return f


def plot_densities(ms: tuple[int, ...], qs: tuple[float, ...]):
    xmin, xmax = -3.5, 3.5
    ymin, ymax = 0.0, 0.65
    f, axs = utils.grid(len(ms) * len(qs), ncols=len(qs))
    x = np.linspace(xmin, xmax, num=100)
    for ax, (m, q) in zip(axs, itertools.product(ms, qs)):
        counts, bins = mixture.hist(2**m, q)
        ax.stairs(counts, bins, color="C0", fill=True)
        ax.set_title(f"N={2**m}, {q=:.4f}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # plot reference density
        ax.plot(x, stats.norm.pdf(x), linestyle=":", color="C3", linewidth=2)
        # plot Hellinger distance value
        ax.text(
            0.05,
            0.95,
            f"H={mixture.hellinger(q, 2**m):.3f}",
            transform=ax.transAxes,
            color="C0",
            ha="left",
            va="top",
        )
    f.tight_layout()
    return f


if __name__ == "__main__":
    # plot Hellinger distances for different values of N
    utils.savefig(plot_distance(tuple(range(1, 7))), "hellinger_distance.pdf")

    # plot optimal qs as function of N
    utils.savefig(plot_optimum(1, 17), "hellinger_optimum.pdf")

    ms = (1, 2, 4, 5)
    # plot suboptimal densities
    qs = (0.5001, 0.75, 0.95, 0.9999)
    utils.savefig(plot_densities(ms, qs), "densities.pdf")
    # plot optimal densities
    qs_opt = tuple(mixture.minimize_hellinger(m**2).x for m in ms)
    utils.savefig(plot_densities(ms, qs_opt), "densities_optimized.pdf")
