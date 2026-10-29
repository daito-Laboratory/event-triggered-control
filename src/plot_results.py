"""Generate plots from CSV simulation output."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main() -> None:
    folder = Path(__file__).resolve().parents[1] / "results"
    event = load_csv(folder / "event_triggered.csv")
    periodic = load_csv(folder / "periodic.csv")
    times = [int(row["step"]) for row in event]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, [row["state"] for row in event], label="Event-triggered")
    ax.plot(times, [row["state"] for row in periodic], linestyle="--", label="Periodic")
    ax.set(xlabel="Step k", ylabel="State x[k]", title="State trajectories")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "state_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.5))
    event_times = [int(row["step"]) for row in event if row["sent"] == 1]
    ax.eventplot(event_times, orientation="horizontal", lineoffsets=1, linelengths=0.7)
    ax.set(xlim=(-0.5, max(times) + 0.5), ylim=(0.3, 1.7), xlabel="Step k", title="Event-triggered transmissions")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(folder / "transmission_times.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
