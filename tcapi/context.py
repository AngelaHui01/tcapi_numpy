class NumpyContext:
    """
    Lightweight runtime context for the NumPy TCAPI backend.

    In the C++ spec, the context manages resources like GPU device handles
    or thread pools (Appendix C.1). For NumPy, no such resources exist —
    NumPy manages threading internally. The context is therefore a simple
    sentinel object that:
      1. Confirms the backend is initialised before any TCAPI call.
      2. Mirrors the mandatory (ctx, ...) function signature from the spec,
         keeping Python and C++ usage structurally identical.
    """

    def __init__(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def __repr__(self) -> str:
        state = "active" if self._active else "destroyed"
        return f"NumpyContext({state})"