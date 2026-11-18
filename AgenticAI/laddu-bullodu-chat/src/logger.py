_status_callback = None

def set_status_callback(cb):
    """Set the process-wide callback used to forward later status messages to the UI."""
    global _status_callback
    _status_callback = cb

def log_status(message: str):
    """Print a workflow update and forward it to the optional UI callback.

    Callback exceptions are suppressed so a display failure does not interrupt
    graph execution; the message is still written to standard output.
    """
    print(message)
    if _status_callback:
        try:
            _status_callback(message)
        except Exception:
            pass