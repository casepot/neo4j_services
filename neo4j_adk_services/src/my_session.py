# my_session.py
from google.adk.sessions.session import Session as _ADKSession

class Session(_ADKSession):
    """
    Drop-in replacement that carries a loss-less millisecond value
    alongside the seconds-float expected by the ADK runtime.
    """
    last_update_time_ms: int | None = None