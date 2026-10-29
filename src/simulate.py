"""Small educational demo of event-triggered feedback control.

This is not a reproduction of unpublished research.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    a: float = 1.05
    b: float = 1.0
    feedback_gain: float = -0.65
    initial_state: float = 4.0
    steps: int = 30
    relative_threshold: float = 0.2
    absolute_threshold: float = 0.02


def simulate(config: Config, event_triggered: bool) -> list[dict[str, float | int]]:
    x = config.initial_state
    transmitted_state = x
    rows: list[dict[str, float | int]] = []

    for k in range(config.steps):
        error = x - transmitted_state
        threshold = config.relative_threshold * abs(x) + config.absolute_threshold
        send = int(k == 0 or not event_triggered or abs(error) >= threshold)
        if send:
            transmitted_state = x
        u = config.feedback_gain * transmitted_state
        rows.append({
            "step": k,
            "state": x,
            "control": u,
            "sent": send,
            "error_before_update": error,
            "threshold": threshold,
        })
        x = config.a * x + config.b * u
    return rows


def save_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int]]) -> tuple[int, float, float]:
    return (
        sum(int(row["sent"]) for row in rows),
        sum(float(row["state"]) ** 2 for row in rows),
        float(rows[-1]["state"]),
    )


def main() -> None:
    config = Config()
    output = Path(__file__).resolve().parents[1] / "results"
    for enabled, label in [(True, "event_triggered"), (False, "periodic")]:
        rows = simulate(config, enabled)
        save_csv(rows, output / f"{label}.csv")
        sends, cost, last_state = summarize(rows)
        print(f"{label}: transmissions={sends}/{config.steps}, sum(x^2)={cost:.4f}, last recorded x={last_state:.5f}")


if __name__ == "__main__":
    main()
