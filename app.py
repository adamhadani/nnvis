"""
nnvis — Interactive neural network visualizations.

Thin entrypoint that registers all pages via st.navigation.
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="nnvis", layout="wide")

pg = st.navigation(
    [
        st.Page(
            "pages/0_Neural_Net_Visualizer.py",
            title="Neural Net Visualizer",
            default=True,
        ),
        st.Page("pages/1_Autograd_Visualizer.py", title="Autograd Visualizer"),
    ]
)

pg.run()
