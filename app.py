"""
BloomChain – Low-emission cold chain for Colombian flowers.

Punto de entrada de la app. Solo define la navegación y la barra lateral;
cada página está en la carpeta pages/.

Para correrla:
    streamlit run app.py
"""
import streamlit as st

from utils import theme as th

st.set_page_config(page_title="BloomChain", page_icon="🌹", layout="wide", initial_sidebar_state="expanded")
th.inject_css()

pages = [
    st.Page("pages/1_Scenario_setup.py", title="Scenario setup", icon=":material/tune:", default=True),
    st.Page("pages/2_Optimization_panel.py", title="Optimization panel", icon=":material/public:"),
    st.Page("pages/3_Analytics_dashboard.py", title="Analytics dashboard", icon=":material/monitoring:"),
    st.Page("pages/4_Model_and_method.py", title="Model & method", icon=":material/functions:"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown(
        f"""
        <div style="margin-top:8px">
          <div style="font-size:1.35rem;font-weight:700;color:{th.INK}">🌹 BloomChain</div>
          <div style="color:{th.INK2};font-size:0.88rem;line-height:1.45;margin-top:4px">
            Low-emission cold chain for Colombian flowers: which cold hubs to open and which
            flows can move from air to sea without losing vase life.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    th.legend([(th.AIR, "Air"), (th.SEA, "Sea"), (th.CRITICAL, "Closed / alert")])
    st.divider()
    st.caption(
        "Personal project by **Alejandro Morantes** · Industrial Engineering, Universidad de La Sabana (Colombia).\n\n"
        "k-medoids clustering · XGBoost disruption forecast · mixed-integer optimisation (PuLP + CBC).\n\n"
        "All data are synthetic (fixed seed) on real coordinates."
    )

nav.run()
