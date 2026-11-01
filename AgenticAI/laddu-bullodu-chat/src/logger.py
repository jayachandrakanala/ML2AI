_status_callback = None

def set_status_callback(cb):
    """Register the UI callback that receives subsequent workflow updates."""
    global _status_callback
    _status_callback = cb

def log_status(message: str):
    """Print a workflow update and forward it to the registered UI callback."""
    print(message)
    if _status_callback:
        try:
            _status_callback(message)
        except Exception:
            pass