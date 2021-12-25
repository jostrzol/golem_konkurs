from math import sqrt, exp, pi
from scipy.special import erf
from typing import List
import pandas as pd
import numpy as np


def prepare_sets():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    dfs = [train_df, test_df]

    for df in dfs:
        df.set_index("Id", inplace=True)

    train_df["readmitted"] = train_df["readmitted"].map(
        {">30": 2, "<30": 1, "NO": 0})

    return dfs


def to_ranges(bins: List[float]):
    return [(b0, b1)for b0, b1 in zip(bins[:-1], bins[1:])]


def J(m: float, sigma: float, bins: List[float], p_bin: List[float]):
    rng = to_ranges(bins)

    if len(rng) != len(p_bin):
        raise ValueError("bins must have length one greater than p_bin")

    def trans(x): return (x - m) / (sqrt(2) * sigma)

    sum = 0
    for p, (x0, x1) in zip(p_bin, rng):
        erfx0 = erf(trans(x0))
        erfx1 = erf(trans(x1))

        sum += 1 / 4 * (erfx1**2 + erfx0**2) + p**2 - 1 / 2 * \
            (erfx1 * erfx0) + p * (-erfx1 + erfx0)

    return sum


def Jm(m: float, sigma: float, bins: List[float], p_bin: List[float]):
    rng = to_ranges(bins)

    if len(rng) != len(p_bin):
        raise ValueError("bins must have length one greater than p_bin")

    def trans_erf(x): return (x - m) / (sqrt(2) * sigma)
    def trans_exp(x): return -(x - m)**2 / (2 * sigma**2)

    sum = 0
    for p, (x0, x1) in zip(p_bin, rng):
        erfx0 = erf(trans_erf(x0))
        erfx1 = erf(trans_erf(x1))

        expx0 = exp(trans_exp(x0))
        expx1 = exp(trans_exp(x1))

        sum += sqrt(2 / pi) / sigma * (
            + expx1 * erfx1 * (1 / 2 - 2 * p)
            + expx0 * erfx0 * (1 / 2 + 2 * p)
            + expx1 * erfx0 + expx0 * erfx1
        )

    return sum


def Jsigma(m: float, sigma: float, bins: List[float], p_bin: List[float]):
    rng = to_ranges(bins)

    if len(rng) != len(p_bin):
        raise ValueError("bins must have length one greater than p_bin")

    def trans_erf(x): return (x - m) / (sqrt(2) * sigma)
    def trans_exp(x): return -(x - m)**2 / (2 * sigma**2)

    sum = 0
    for p, (x0, x1) in zip(p_bin, rng):
        erfx0 = erf(trans_erf(x0))
        erfx1 = erf(trans_erf(x1))

        expx0 = exp(trans_exp(x0))
        expx1 = exp(trans_exp(x1))

        fctx0 = (x0 - m)
        fctx1 = (x1 - m)

        sum += sqrt(2 / pi) / sigma**2 * (
            + fctx1 * expx1 * erfx1 * (1 / 2 - 2 * p)
            + fctx0 * expx0 * erfx0 * (1 / 2 + 2 * p)
            + 1 / 2 * (fctx1 * expx1 * erfx0 + fctx0 * expx0 * erfx1)
        )

    return sum


if __name__ == "__main__":
    train_df, test_df = prepare_sets()

    age = train_df["age"].value_counts()
    counts = age.values
    ranges = age.index.map(lambda s: np.array(
        s[1:-1].split("-")).astype(float))
    means = ranges.map(lambda r: r.mean())

    avg = np.average(means, weights=counts)
    std = sqrt(np.average((means - avg)**2, weights=counts))

    rng = np.stack(ranges)
    rng = rng[rng[:, 0].argsort()]
    bins = [*rng[:, 0], rng[-1][1]]
    p_bin = counts / sum(counts)

    args = [bins, p_bin]

    mins = gradient_descend(
        lambda *x: J(*x, *args),
        (lambda *x: Jm(*x, *args),
         lambda *x: Jsigma(*x, *args)),
        (avg, std),
        2,
        0.01,
        1000)
