"""Smoke tests: confirm the scaffold imports and is wired together."""

import app
from terminal import __version__
from terminal.config import APP_TITLE


def test_version_is_set():
    # The package should expose a non-empty version string.
    assert __version__


def test_title_is_set():
    assert APP_TITLE == "Trading Terminal"


def test_every_page_is_callable():
    # Each entry in PAGES must be a function Streamlit can call.
    assert app.PAGES
    for label, render in app.PAGES.items():
        assert callable(render), f"Page '{label}' is not callable"
