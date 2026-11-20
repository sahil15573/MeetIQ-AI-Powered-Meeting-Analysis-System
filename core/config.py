
import os
import streamlit as st
from dotenv import load_dotenv

# Load local .env variables
load_dotenv()


def get_setting(name: str, default=None):
    """
    Read configuration from environment variables
    or Streamlit secrets.
    """

    value = os.getenv(name)

    if value:
        return value

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    return value or default


def validate_setting(name: str):
    value = get_setting(name)

    if not value:
        raise ValueError(
            f"{name} is missing. Configure it in .env "
            "or Streamlit secrets."
        )

    return value
