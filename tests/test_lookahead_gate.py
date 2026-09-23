from fractions import Fraction as F
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lookahead_gate as lg  # noqa: E402


class LookaheadTests(unittest.TestCase):
    def test_all_binary_second_order_models_and_guesses(self):
        for table in itertools.product(range(2), repeat=4):
            def next_token(seq):
                return table[2 * seq[-2] + seq[-1]]
            for prefix in itertools.product(range(2), repeat=2):
                prefix = list(prefix)
                exact = []
                for _ in range(4):
                    exact.append(next_token(prefix + exact))
                for guess in itertools.product(range(2), repeat=3):
                    for rounds in range(4):
                        draft = lg.jacobi_proposal(next_token, prefix, guess, rounds)
                        emitted = lg.verify_prefix(next_token, prefix, draft)
                        self.assertEqual(emitted, exact[:len(emitted)])
                        if rounds == 3:
                            self.assertEqual(draft, exact[:3])

    def test_rejection_cost_is_counted(self):
        value = lg.analyze(F(9, 10), 2, F(1), F(1), F(1))
        self.assertEqual(F(value["expected_tokens"]), F(271, 100))
        self.assertEqual(F(value["expected_step_cost"]), F(219, 100))
        self.assertLess(F(value["speed_ratio"]), F(3, 2))
        self.assertFalse(lg.analyze(F(1, 2), 2, F(1), F(1), F(1))["profitable_under_assumptions"])

    def test_full_jacobi_convergence_cannot_prove_a_speedup(self):
        # w draft sweeps + one verification emit at most w+1 tokens. Even free
        # replay cannot improve on plain decode when every sweep costs one step.
        for window in range(1, 9):
            value = lg.analyze(F(1), window, F(1), F(window), F(0))
            self.assertEqual(F(value["speed_ratio"]), 1)


if __name__ == "__main__":
    unittest.main()
