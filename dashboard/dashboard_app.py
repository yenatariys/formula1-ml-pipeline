df = load_results()
st.divider()
elif not df_preds.empty:
"""Compatibility shim that exposes the classic dashboard as the default app.

Historically the project used ``dashboard_app.py`` as the Streamlit entry
point. The classic dashboard has now moved to ``classic_app.py`` so we keep
this tiny shim to avoid breaking existing docs or scripts that still reference
the old file.
"""

from classic_app import *  # noqa: F401,F403