"""Streamlit control and analysis app for Prompt Garden experiment work."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chemistry_bot.promptops.review_app_analysis import (  # noqa: E402
    render_review_analysis_surface,
)
from chemistry_bot.promptops.review_app_control import (  # noqa: E402
    render_control_surface,
    render_workspace_status_header,
)
from chemistry_bot.promptops.review_app_data import (  # noqa: E402
    discover_review_scopes,
    load_garden_index_bundle,
    load_scope_bundle,
)
from chemistry_bot.promptops.review_app_support import (  # noqa: E402
    default_garden_root,
)


__all__ = [
    "discover_review_scopes",
    "load_scope_bundle",
    "main",
]


def main() -> None:
    """Launch the Prompt Garden control-panel shell."""

    st.set_page_config(
        page_title="Prompt Garden Control Panel",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Prompt Garden Control Panel")
    st.caption(
        "Browse the workspace from Control, then switch to Analysis for answer review."
    )

    with st.sidebar:
        st.header("Workspace")
        garden_root = st.text_input(
            "Prompt Garden Root",
            value=str(default_garden_root(REPO_ROOT)),
            help="Path to the Prompt Garden workspace root.",
        )
        if st.button("Reload Cached Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    scopes = discover_review_scopes(garden_root)
    index_bundle = load_garden_index_bundle(garden_root)
    render_workspace_status_header(
        garden_root=garden_root,
        index_bundle=index_bundle,
        scopes=scopes,
    )

    control_tab, analysis_tab = st.tabs(["Control", "Analysis"])

    with control_tab:
        render_control_surface(
            garden_root=garden_root,
            scopes=scopes,
        )

    with analysis_tab:
        render_review_analysis_surface(
            garden_root=garden_root,
            scopes=scopes,
        )


if __name__ == "__main__":
    main()
