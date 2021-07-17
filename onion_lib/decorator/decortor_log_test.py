from functools import wraps
from typing import Dict

LOG_RESULTS = []


def reset_cache():
    global LOG_RESULTS
    LOG_RESULTS = []


def log_test(num_test_case):
    def _log_test(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                output = f(*args, **kwargs)
                data = {
                    "name": f.__name__,
                    "pass": True
                }
                if output and isinstance(output, Dict):
                    data.update(output)
                LOG_RESULTS.append(data)
            except AssertionError:
                LOG_RESULTS.append({
                    "name": f.__name__,
                    "pass": False
                })
            if len(LOG_RESULTS) == num_test_case:
                print(LOG_RESULTS)

        return wrapper

    return _log_test
