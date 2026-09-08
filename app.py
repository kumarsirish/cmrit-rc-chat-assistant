"""Entry point Streamlit runs: `streamlit run app.py`.

All dashboard/UI code lives in rag/ui/dashboard.py; this file just
launches it.
"""

from rag.ui.dashboard import main

if __name__ == "__main__":
    main()
