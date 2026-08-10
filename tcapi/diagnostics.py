from __future__ import annotations
import os
import time
import functools
import numpy as np
from typing import Callable, Any


def _verbose_level() -> int:
    try:
        return int(os.environ.get("TCAPI_VERBOSE", "0"))
    except ValueError:
        return 0


def _tensor_summary(arg: Any) -> str:
    if isinstance(arg, np.ndarray):
        return f"ndarray(shape={arg.shape}, dtype={arg.dtype})"
    return repr(arg)


def _format_args(args: tuple, kwargs: dict) -> str:
    parts = []
    for i, a in enumerate(args):
        # Skip the context handle (first positional arg by convention)
        if i == 0:
            continue
        parts.append(_tensor_summary(a))
    for k, v in kwargs.items():
        parts.append(f"{k}={_tensor_summary(v)}")
    return ", ".join(parts)


def verbose_wrap(fn: Callable) -> Callable:
    """
    Decorator that honours TCAPI_VERBOSE for any TCAPI function.

    TCAPI_VERBOSE=0  →  silent (default)
    TCAPI_VERBOSE=1  →  print function name + salient input info
    TCAPI_VERBOSE=2  →  level-1 output + measured wall-clock time
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        level = _verbose_level()

        if level == 0:
            return fn(*args, **kwargs)

        # Level 1+: print function name and input summary
        arg_summary = _format_args(args, kwargs)
        print(f"[TCAPI] {fn.__name__}({arg_summary})", end="")

        if level >= 2:
            # Level 2: time the call
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"  |  elapsed={elapsed * 1e3:.4f} ms")
        else:
            print()  # newline after level-1 message
            result = fn(*args, **kwargs)

        return result

    return wrapper