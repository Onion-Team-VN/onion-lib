from onion_lib.decorator import log_test, reset_cache
from onion_lib.decorator.decortor_log_test import new_log_test
import unittest
import time
from functools import wraps
import traceback

NUM_TEST_CASE = 2


# class TestSolution(unittest.TestCase):
#     abc = "abc"
#
#     @new_log_test
#     def test_optimizer_adagrad_2d(self, var1=None):
#         x1, x2, s1, s2 = 0.5, 0.67, 0, 0
#         self.assertAlmostEqual(x1, 0.1000199985)
#         self.assertAlmostEqual(x2, 0.2700000278458)
#         self.assertAlmostEqual(s1, 0.01)
#         self.assertAlmostEqual(s2, 7.1824000)
#         var1 = 5
#
#     # @log_test(NUM_TEST_CASE)
#     # def test_passed(self):
#     #     assert 1 == 1

def timer(func):
    """Print the runtime of the decorated function"""

    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        print(args)
        print(kwargs)
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


@new_log_test
def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i ** 2 for i in range(10000)])


class ABCD:
    a = "a"

    @new_log_test
    def do_something(self, a, b, c, d):
        try:
            if a == 1:
                return a
            else:
                raise RuntimeError(f"RuntimeError EROR")
        except Exception as error:
            print(error)
            # traceback.print_exc()
            return None


if __name__ == "__main__":
    a = ABCD()
    o = a.do_something(0, 1, 2, 3)
    print(o)
