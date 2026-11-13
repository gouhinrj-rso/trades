"""Central place for app-wide settings.

Keep constants here so they are not scattered through the code.
Secrets (API keys) do NOT belong in this file - read them from
environment variables instead (see .env.example).
"""

import os

# Text shown in the browser tab and at the top of the app.
APP_TITLE = "Trading Terminal"

# Example of reading an optional setting from the environment.
# Returns an empty string if the variable is not set.
MARKET_DATA_API_KEY = os.environ.get("MARKET_DATA_API_KEY", "")
