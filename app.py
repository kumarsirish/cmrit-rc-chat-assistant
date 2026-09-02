"""Entry point Streamlit runs: `streamlit run app.py`.

All dashboard/UI code lives in rag/dashboard.py; this file just
launches it.
"""

from rag.dashboard import main

if __name__ == "__main__":
    main()
