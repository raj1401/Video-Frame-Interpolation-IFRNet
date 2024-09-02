import time
from functools import wraps

def timing_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper


# Example usage:
@timing_wrapper
def my_function():
    # Your function code here
    time.sleep(2)  # Simulating some computation


if __name__ == "__main__":
    my_function()
