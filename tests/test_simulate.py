"""Basic tests for the educational demo."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import Config, simulate, summarize  # noqa: E402


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_lengths(self):
        self.assertEqual(len(simulate(self.cfg, True)), self.cfg.steps)
        self.assertEqual(len(simulate(self.cfg, False)), self.cfg.steps)

    def test_initial_send(self):
        self.assertEqual(simulate(self.cfg, True)[0]["sent"], 1)

    def test_periodic_sends_every_step(self):
        self.assertEqual(summarize(simulate(self.cfg, False))[0], self.cfg.steps)

    def test_event_triggered_sends_less(self):
        sends = summarize(simulate(self.cfg, True))[0]
        self.assertGreater(sends, 0)
        self.assertLess(sends, self.cfg.steps)


if __name__ == "__main__":
    unittest.main()
