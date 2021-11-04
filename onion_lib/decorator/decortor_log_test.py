from functools import wraps
from typing import Dict
import matplotlib.pyplot as plt
import io
import base64

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
                print(f"<TEST>{LOG_RESULTS}</TEST>")

        return wrapper

    return _log_test


def show_figure(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        output = f(*args, **kwargs)

        # Add image streamer
        io_bytes = io.BytesIO()
        plt.savefig(io_bytes, format='jpg')
        io_bytes.seek(0)
        fig1_encode = base64.b64encode(io_bytes.read()).decode('utf-8')
        if output is not None:
            output.update({
                "image": fig1_encode
            })
        else:
            output = {
                "image": fig1_encode
            }
        return output

    return wrapper
