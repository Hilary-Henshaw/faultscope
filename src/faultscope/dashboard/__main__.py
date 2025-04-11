"""Entry point for ``python -m faultscope.dashboard``.

Launches the Streamlit application by delegating to the ``streamlit``
CLI.  All Streamlit configuration (server port, address, theme) can be
controlled via the standard ``STREAMLIT_*`` environment variables or
a ``.streamlit/config.toml`` file.

Usage::

    python -m faultscope.dashboard
    # or
    faultscope-dashboard
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Invoke ``streamlit run`` on the FaultScope app module."""
    app_path = Path(__file__).parent / "streamlit" / "app.py"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
