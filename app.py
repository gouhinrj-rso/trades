"""Trading Terminal - Streamlit entry point.

Run from this folder with:
    streamlit run app.py

This is an empty scaffold. Each page of the terminal should live in its own
function below (or in the `terminal` package) and be registered in PAGES.
"""

import streamlit as st

from terminal import __version__
from terminal.config import APP_TITLE


def render_home() -> None:
    """Placeholder landing page. Replace with the first real feature."""
    st.title(APP_TITLE)
    st.caption(f"Scaffold version {__version__}")
    st.info("No features built yet. Add pages to the PAGES dictionary in app.py.")


# Maps the sidebar label to the function that draws that page.
# To add a page: write a render_<name>() function and add one line here.
PAGES = {
    "Home": render_home,
}


def main() -> None:
    """Configure the page, draw the sidebar, and render the chosen page."""
    # set_page_config must be the first Streamlit call in the script.
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Sidebar navigation: the user picks a page by its label.
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))

    # Look up the matching function and call it.
    PAGES[choice]()


if __name__ == "__main__":
    main()
