import math
import os
import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# typed subplots (flattening dimensions of length 1)
ShareType = bool | typing.Literal["none", "all", "row", "col"]


@typing.overload
def subplots(
    nrows: typing.Literal[1] = ...,
    ncols: typing.Literal[1] = ...,
    figsize: tuple[float, float] | None = ...,
    *,
    squeeze: typing.Literal[True] = ...,
    sharex: ShareType = ...,
    sharey: ShareType = ...,
    **kwargs,
) -> tuple[Figure, Axes]:
    ...


@typing.overload
def subplots(
    nrows: int = ...,
    ncols: typing.Literal[1] = ...,
    figsize: tuple[float, float] | None = ...,
    *,
    squeeze: typing.Literal[True] = ...,
    sharex: ShareType = ...,
    sharey: ShareType = ...,
    **kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    ...


@typing.overload
def subplots(
    nrows: typing.Literal[1] = ...,
    ncols: int = ...,
    figsize: tuple[float, float] | None = ...,
    *,
    squeeze: typing.Literal[True] = ...,
    sharex: ShareType = ...,
    sharey: ShareType = ...,
    **kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    ...


@typing.overload
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    figsize: tuple[float, float] | None = ...,
    *,
    squeeze: typing.Literal[False, True] = ...,
    sharex: ShareType = ...,
    sharey: ShareType = ...,
    **kwargs,
) -> tuple[Figure, tuple[tuple[Axes, ...], ...]]:
    ...


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    *,
    squeeze: bool = True,
    sharex: ShareType = True,
    sharey: ShareType = False,
    **kwargs,
) -> tuple[Figure, typing.Any]:
    f, axs_ = plt.subplots(
        nrows,
        ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        figsize=figsize,
        **kwargs,
    )
    if squeeze:
        # return Figure, Axes if only one subplot
        if axs_.size == 1:
            return f, axs_[0, 0]
        # return Figure, tuple[Axes, ...] if one dimension has length 1
        if axs_.shape[0] == 1 or axs_.shape[1] == 1:
            axs_ = axs_.flatten()
            return f, tuple(axs_)
    # return Figure, tuple[tuple[Axes, ...], ...] otherwise
    return f, tuple(tuple(row) for row in axs_)


def grid(
    nitems: int,
    ncols: int | None = None,
    unit_size: tuple[float, float] = (3, 2),
    sharex: ShareType = "col",
    sharey: ShareType = "row",
    **kwargs,
) -> tuple[Figure, tuple[Axes, ...]]:
    ncols = min(nitems, 3) if ncols is None else ncols
    nrows = math.ceil(nitems / ncols)
    width = 2 + ncols * unit_size[0]
    height = 2 + nrows * unit_size[1]
    f, axs_ = subplots(
        nrows,
        ncols,
        figsize=(width, height),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        **kwargs,
    )
    return f, tuple(ax for axs in axs_ for ax in axs)


def pdfs(
    path: str,
    samples: dict[str, np.ndarray],
    bins: int,
    target_pdf: typing.Callable[..., np.ndarray] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    **kwargs,
) -> None:
    f, axs = grid(len(samples), **kwargs)

    for i, (name, s) in enumerate(samples.items()):
        axs[i].hist(s, color="C0", bins=bins, density=True)
        axs[i].set_title(name)
        if xlim:
            axs[i].set_xlim(*xlim)
        if ylim:
            axs[i].set_ylim(*ylim)
        # plot target distribution
        if target_pdf:
            x_min, x_max = axs[i].get_xlim()
            x = np.linspace(x_min, x_max, num=100)
            axs[i].plot(x, target_pdf(x), linestyle=":", color="C3", linewidth=2)

    if not os.path.isdir(path):
        os.mkdir(path)

    f.tight_layout()
    f.savefig(path)
