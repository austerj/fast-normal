import math
import typing

import matplotlib.pyplot as plt
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
) -> tuple[Figure, Axes]: ...


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
) -> tuple[Figure, tuple[Axes, ...]]: ...


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
) -> tuple[Figure, tuple[Axes, ...]]: ...


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
) -> tuple[Figure, tuple[tuple[Axes, ...], ...]]: ...


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
