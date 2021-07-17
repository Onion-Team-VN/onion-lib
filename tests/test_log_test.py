from onion_lib.decorator.decortor_log_test import log_test, reset_cache
import unittest

NUM_TEST_CASE = 2


class TestSolution(unittest.TestCase):
    @log_test(NUM_TEST_CASE)
    def test_optimizer_adagrad_2d(self):
        x1, x2, s1, s2 = 0.5, 0.67, 0, 0
        self.assertAlmostEqual(x1, 0.1000199985)
        self.assertAlmostEqual(x2, 0.2700000278458)
        self.assertAlmostEqual(s1, 0.01)
        self.assertAlmostEqual(s2, 7.1824000)

    @log_test(NUM_TEST_CASE)
    def test_passed(self):
        assert 1 == 1


if __name__ == "__main__":
    unittest.main()
