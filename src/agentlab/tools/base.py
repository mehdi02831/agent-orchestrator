class ToolError(Exception):
    """Recoverable tool failure (bad path, network error, bad expression)."""


class ToolConfigError(Exception):
    """Misconfiguration (missing key, 401). Aborts the run; never counts as a tool failure."""
