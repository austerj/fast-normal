import numpy as np

from fastnorm import OUT_PATH, splitmix64
from fastnorm.norm import box_muller, mixture
from scripts import utils

NSAMPLES = 5_000_000
NRUNS = 1_000
SEED = 0


def run_benchmarks() -> utils.BenchmarkResults:
    # preallocate arrays
    ints = splitmix64.sample_ints(NSAMPLES, SEED)
    z = np.empty_like(ints, dtype=np.float64)
    mixture_fill = mixture.filler(exponent=10)
    # benchmark computational time of filling z from ints
    return utils.benchmarks(
        {
            "Mixture": lambda: mixture_fill(z, ints),
            "Box-Muller": lambda: box_muller.fill(z, ints),
        },
        NRUNS,
    )


def to_latex(results: utils.BenchmarkResults, precision: int = 2):
    # create dict of benchmark results as strings
    to_str = lambda val: f"{val:.{precision}f}"
    dicts = [
        {
            "Mean": to_str(d["mean"]),
            "CI": f"({to_str(d["ci"][0])}, {to_str(d["ci"][1])})",
            "Min": to_str(d["min"]),
            "Max": to_str(d["max"]),
        }
        for d in [bm.to_dict("ms") for bm in results.values()]
    ]
    # prepare headers and values for rows
    headers = ["", *next(iter(dicts)).keys()]
    items = [[k, *v.values()] for k, v in zip(results.keys(), dicts)]
    # compose lines of tabular env
    rows = (
        r"\begin{tabular}{lrcrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        "\n".join(" & ".join(item) + r" \\" for item in items),
        r"\bottomrule",
        r"\end{tabular}",
    )
    return "\n".join(rows)


if __name__ == "__main__":
    results = run_benchmarks()

    with open(OUT_PATH / "benchmark.tex", "w+") as f:
        print(to_latex(results), file=f)
