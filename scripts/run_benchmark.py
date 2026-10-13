"""Run every controller on every scenario, then write tables and figures.

    python scripts/run_benchmark.py              # everything
    python scripts/run_benchmark.py --only step  # one scenario
    python scripts/run_benchmark.py --plots-only # redraw from saved runs
"""

import argparse
import csv
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from quadbench.controllers import ALL_CONTROLLERS
from quadbench.metrics import summarize
from quadbench.params import QuadParams
from quadbench.scenarios import SCENARIOS
from quadbench.simulator import simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"

# Settling time only means something when the run starts away from the reference.
SETTLING_SCENARIOS = {"hover_recovery", "step", "long_step", "flip_recovery"}


def run_one(job):
    scenario_name, controller_name = job
    params = QuadParams()
    scenario = SCENARIOS[scenario_name]
    result = simulate(ALL_CONTROLLERS[controller_name](params), scenario, params)
    return result, summarize(result, scenario)


def write_tables(rows):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "benchmark.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    controllers = list(ALL_CONTROLLERS)
    lines = [
        "| Scenario | " + " | ".join(controllers) + " |",
        "|---|" + "---:|" * len(controllers),
    ]
    by_key = {(r["scenario"], r["controller"]): r for r in rows}
    for name in SCENARIOS:
        cells = []
        for c in controllers:
            r = by_key.get((name, c))
            if r is None:
                cells.append("")
            elif r["status"] != "ok":
                cells.append("crash")
            else:
                v = r['rmse_cm']
                cell = f"{v:.2f}" if v >= 0.01 else "<0.01"
                if name in SETTLING_SCENARIOS and np.isfinite(r["settle_s"]):
                    cell += f" ({r['settle_s']:.2f} s)"
                cells.append(cell)
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    compute = {c: np.mean([r["compute_ms"] for r in rows if r["controller"] == c]) for c in controllers}
    lines.append("")
    lines.append("Mean controller compute time per call (Python, single core): "
                 + ", ".join(f"{c} {compute[c]:.2f} ms" for c in controllers) + ".")
    (RESULTS / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario names to run")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    names = args.only or list(SCENARIOS)

    if not args.plots_only:
        jobs = [(s, c) for s in names for c in ALL_CONTROLLERS]
        tic = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            outputs = list(pool.map(run_one, jobs))
        print(f"{len(jobs)} runs in {time.time() - tic:.0f} s")
        for result, _ in outputs:
            with open(RAW / f"{result.scenario}__{result.controller}.pkl", "wb") as fh:
                pickle.dump(result, fh)

    results, rows = {}, []
    for s in SCENARIOS:
        for c in ALL_CONTROLLERS:
            path = RAW / f"{s}__{c}.pkl"
            if path.exists():
                with open(path, "rb") as fh:
                    results[(s, c)] = pickle.load(fh)
                rows.append(summarize(results[(s, c)], SCENARIOS[s]))
    write_tables(rows)

    from quadbench import plotting
    plotting.make_all(results, rows, RESULTS / "figures")


if __name__ == "__main__":
    main()
