# src/telemetry.py
import contextlib
import logging
import os
import warnings
from pathlib import Path

# Keep Phoenix data in the project instead of its temporary launch directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
PHOENIX_DIR = PROJECT_DIR / "data" / "phoenix"
PHOENIX_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PHOENIX_WORKING_DIR"] = str(PHOENIX_DIR)
os.environ.setdefault("PHOENIX_PROJECT_NAME", "laddu-bullodu-chat")
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", category=Warning)
logging.disable(logging.WARNING)

import streamlit as st
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor


@st.cache_resource
def init_telemetry():
    """Initialize persistent Phoenix storage and export application traces across restarts."""
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            px_session = px.launch_app(use_temp_dir=False)
            tracer_provider = register(
                project_name=os.environ["PHOENIX_PROJECT_NAME"],
                batch=True,
                verbose=False,
            )
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        return tracer_provider
    except Exception:
        return None
