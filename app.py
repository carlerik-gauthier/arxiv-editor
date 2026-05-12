"""Local Streamlit entry point.

Run with:
    streamlit run app.py
"""

from src.ui.streamlit_app import run_app


def main() -> None:
    """Launch the Julius Streamlit app."""
    run_app()


if __name__ == "__main__":
    main()
