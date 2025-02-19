import math
import time
import typing
from dataclasses import dataclass
from functools import cached_property

TKey = typing.TypeVar("TKey")
TReal = typing.TypeVar("TReal", bound=int | float)
TimeUnit = typing.Literal["ms", "ns", "s"]


def _time_unit_converter(unit: TimeUnit):
    match unit:
        case "ms":
            c = 1_000
        case "ns":
            c = 1_000_000
        case "s":
            c = 1
        case _:
            raise ValueError(f"Invalid unit {unit}")
    return lambda seconds: c * seconds


class StatisticsDict(typing.TypedDict):
    mean: float
    sd: float
    ci: tuple[float, float]
    min: float
    max: float


@dataclass
class Statistics(typing.Generic[TReal]):
    _xs: list[TReal]

    def __len__(self):
        return len(self._xs)

    # stats
    @cached_property
    def mean(self) -> float:
        return sum(self._xs) / len(self)

    @cached_property
    def sd(self) -> float:
        return math.sqrt(sum(x**2 for x in self._xs) / len(self) - self.mean**2)

    @cached_property
    def ci(self) -> tuple[float, float]:
        return (self.mean - 1.96 * self.sd, self.mean + 1.96 * self.sd)

    @cached_property
    def min(self) -> TReal:
        return min(self._xs)

    @cached_property
    def max(self) -> TReal:
        return max(self._xs)

    def to_dict(self, unit_converter: typing.Callable[[int | float], float] = float) -> StatisticsDict:
        uc = unit_converter
        return {
            "mean": uc(self.mean),
            "sd": uc(self.mean),
            "ci": (uc(self.ci[0]), uc(self.ci[1])),
            "min": uc(self.min),
            "max": uc(self.max),
        }


@dataclass
class BenchmarkResult(typing.Generic[TReal], Statistics[float]):
    output: Statistics[TReal]

    def __str__(self) -> str:
        return self.to_string()

    @property
    def times(self) -> list[float]:
        return self._xs

    @property
    def runs(self) -> int:
        return len(self.times)

    def to_string(self, unit: TimeUnit = "ms") -> str:
        uc = _time_unit_converter(unit)
        return f"{uc(self.mean):.2f}{unit} [{uc(self.ci[0]):.2f}, {uc(self.ci[1]):.2f}]"

    def to_dict(self, unit: TimeUnit = "ms") -> StatisticsDict:
        return super().to_dict(_time_unit_converter(unit))


class BenchmarkResults(dict[TKey, BenchmarkResult]):
    def to_dict(self, unit: TimeUnit = "ms") -> dict[TKey, StatisticsDict]:
        return {k: v.to_dict(unit) for k, v in self.items()}


def benchmark(f: typing.Callable, nruns: int, keep_output: bool = False) -> BenchmarkResult:
    """Benchmark the execution time of the provided callables."""
    times: list[float] = []
    output: list[typing.Any] = []
    f()  # ensure function is already compiled
    for _ in range(nruns):
        start = time.monotonic()
        x = f()
        end = time.monotonic()
        times.append(end - start)
        if keep_output:
            output.append(x)
    return BenchmarkResult(times, Statistics(output))


def benchmarks(callables: dict[TKey, typing.Callable], nruns: int, keep_output: bool = False) -> BenchmarkResults[TKey]:
    """Benchmark the execution time of the provided callables."""
    results = BenchmarkResults()
    for key, f in callables.items():
        results[key] = benchmark(f, nruns, keep_output)
    return results
