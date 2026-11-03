"""Reproducible micro-benchmark suite for pyscio kernels and routing (Phase R8)."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

import pyscio as sc
import pyscio.functional as F


def _bench(label: str, fn: Callable[[], object], repeats: int = 7) -> tuple[float, float]:
    """Run warm-up + timed repeats, returning (median_ms, iqr_ms)."""
    fn()  # warm-up
    timings_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms.sort()
    med = statistics.median(timings_ms)
    q25 = timings_ms[len(timings_ms) // 4]
    q75 = timings_ms[(3 * len(timings_ms)) // 4]
    iqr = q75 - q25
    print(f"{label:<45} | Median: {med:7.2f} ms | IQR: {iqr:6.2f} ms")
    return med, iqr


def main() -> None:
    rng = np.random.default_rng(2026)
    g1_large = rng.normal(0.0, 1.0, 50_000)
    g2_large = rng.normal(0.2, 1.2, 50_000)
    groups_k15 = [rng.normal(i * 0.3, 1.0 + 0.1 * (i % 3), 200) for i in range(15)]
    boot_arr = rng.lognormal(0.0, 0.5, 500)
    df_ols = pd.DataFrame(
        {
            "y": rng.normal(0.0, 1.0, 5_000),
            "x1": rng.normal(0.0, 1.0, 5_000),
            "x2": rng.normal(0.0, 1.0, 5_000),
            "x3": rng.normal(0.0, 1.0, 5_000),
        }
    )

    print("=" * 80)
    print("pyscio Kernel & Router Benchmark Suite (7 repeats after warm-up)")
    print("=" * 80)
    _bench("Welch t-test + NCT CI (N=100,000)", lambda: sc.compare(g1_large, g2_large))
    _bench("Cliff's delta O(N log N) (N=100,000)", lambda: F.cliffs_d(g1_large, g2_large))
    _bench("Games-Howell tensor post-hoc (K=15)", lambda: F.games_howell(*groups_k15))
    _bench("Dunn rank tensor post-hoc (K=15)", lambda: F.dunn(*groups_k15))
    _bench(
        "BCa Bootstrap CI (N=500, B=2,000)",
        lambda: F.bootstrap_ci(boot_arr, n_resamples=2000, seed=42),
    )
    _bench("OLS QR + HC3 Robust SE (N=5,000, p=4)", lambda: sc.ols(df_ols, "y ~ x1 + x2 + x3"))
    print("=" * 80)


if __name__ == "__main__":
    main()
