"""
Página 3 – Analytics dashboard: 12 meses de envíos, el plan optimizado y el desempeño del modelo de predicción.
"""
import numpy as np
import pandas as pd
import streamlit as st

from utils import charts, loaders as L
from utils import theme as th

data = L.load_data()
metrics = L.load_metrics()
out = L.get_results()
res = out["res"]
market = res["market"]
mk = L.market_info(market)
ship_all = data["shipments_history"][data["shipments_history"].market == market]
market_dests = [d for d in th.DEST_LABEL if d in set(ship_all.destination)]

st.markdown(f'<h1 style="display:flex;align-items:center;gap:14px">Analytics dashboard '
            f'<span style="font-size:1rem;font-weight:500;color:{th.INK2};display:inline-flex;align-items:center;gap:8px">'
            f'{th.flag(market, 20)} {mk["name"]}</span></h1>', unsafe_allow_html=True)
st.caption(f"Twelve months of shipment history to {mk['name']} (today's mostly-air operation), the optimised plan from the current "
           "scenario, and the out-of-sample performance of the disruption models. Change the market in Scenario setup.")

# ---- Una sola fila de filtros para todos los gráficos del histórico ----
f1, f2, f3, f4 = st.columns([1.2, 0.8, 1.5, 2.3])
dmin, dmax = ship_all.date.min().date(), ship_all.date.max().date()
period = f1.date_input("Period", value=(dmin, dmax), min_value=dmin, max_value=dmax)
modes = f2.multiselect("Mode", ["air", "sea"], default=["air", "sea"], format_func=th.MODE_LABEL.get)
flowers = f3.multiselect("Flower", list(th.FLOWER_LABEL), default=list(th.FLOWER_LABEL), format_func=th.FLOWER_LABEL.get)
dests = f4.multiselect("Destination", market_dests, default=market_dests, format_func=th.DEST_LABEL.get)

d0, d1 = (period[0], period[-1]) if isinstance(period, (tuple, list)) and len(period) else (dmin, dmax)
ship = ship_all[(ship_all.date.dt.date >= d0) & (ship_all.date.dt.date <= d1) & ship_all["mode"].isin(modes)
                & ship_all.flower.isin(flowers) & ship_all.destination.isin(dests)]

if ship.empty:
    st.warning("No shipments match these filters.")
    st.stop()

# ---- KPIs ----
boxes = float(ship.boxes.sum())
lost = float(ship.boxes_lost_quality.sum())
avg_delay = float(np.average(ship.delay_minutes, weights=ship.boxes))
days = max((d1 - d0).days + 1, 1)
late = float(ship.loc[ship.delay_minutes > 24 * 60, "boxes"].sum())
sea_share = float(ship.loc[ship["mode"] == "sea", "boxes"].sum() / boxes)
th.kpi_row([
    th.kpi("Boxes shipped", f"{boxes / 1000:,.0f}<small>k</small>", f"{boxes / days:,.0f} per day over {days} days"),
    th.kpi("Boxes lost to quality", f"{lost:,.0f}", f"{lost / boxes:.2%} of boxes shipped · {lost / days * 365:,.0f} per year"),
    th.kpi("Average delay", f"{avg_delay:,.0f} <small>min</small>", "per box, against the planned transit time"),
    th.kpi("Delayed more than 24 h", f"{late / boxes:.1%}", f"{late:,.0f} boxes"),
    th.kpi("Shipped by sea", f"{sea_share:.0%}", "today's practice: almost everything flies",
           bar=[(th.SEA, sea_share), (th.AIR, 1 - sea_share)]),
])
st.write("")

# ---- Fila 1: Sankey del plan optimizado + treemap ----
c1, c2 = st.columns([1.45, 1], gap="medium")
with c1:
    th.section("Optimised weekly flow: farm municipality → cold hub → Colombian gateway → European arrival → customer",
               "From the current scenario in the Optimization panel. Orange links fly, blue links sail; the last links take the colour of the European mode.")
    st.plotly_chart(charts.sankey(data, res["after"], res["candidates"]), use_container_width=True, config=th.PLOTLY_CONFIG)
with c2:
    th.section("Boxes by destination and flower type", "Shipment history, filtered period.")
    st.plotly_chart(charts.treemap(ship), use_container_width=True, config=th.PLOTLY_CONFIG)
    th.legend([(th.FLOWER_COLOR[f], th.FLOWER_LABEL[f]) for f in th.FLOWER_LABEL])

# ---- Fila 2: volumen mensual por modo + violines ----
c1, c2 = st.columns(2, gap="medium")
with c1:
    th.section("Boxes shipped per month by mode", "Peaks: Valentine's Day (January–February) and Mother's Day (April–May).")
    st.plotly_chart(charts.monthly_mode(ship), use_container_width=True, config=th.PLOTLY_CONFIG)
with c2:
    th.section("Transit time by mode", "Separate panels because the scales differ: one to two days by air, more than two weeks by sea.")
    st.plotly_chart(charts.violins(ship), use_container_width=True, config=th.PLOTLY_CONFIG)

# ---- Fila 3: gráfico de control ----
fig, n_out = charts.control_chart(ship)
th.section("Control chart of daily delays (mean ± 3σ)",
           f"{n_out} days are out of control in this selection. They line up with fog, strike, congestion and storm days.")
st.plotly_chart(fig, use_container_width=True, config=th.PLOTLY_CONFIG)

# ---- Fila 4: de X a Y ----
th.section("Boxes shipped from X to Y", "Pick an origin gateway and a destination.")
g1, g2, g3 = st.columns([1, 1, 3])
gw = g1.selectbox("From (gateway)", ["BOG", "MDE", "CTG", "SMR"], format_func=th.NODE_LABEL.get)
dest = g2.selectbox("To (destination)", market_dests, format_func=th.DEST_LABEL.get)
od = ship[(ship.gateway == gw) & (ship.destination == dest)]
with g3:
    if od.empty:
        st.info("No shipments on this pair with the current filters.")
    else:
        b = float(od.boxes.sum())
        th.kpi_row([
            th.kpi("Boxes shipped", f"{b:,.0f}", f"{len(od):,} consignments"),
            th.kpi("Average transit", f"{np.average(od.transit_minutes, weights=od.boxes) / 60:,.0f} <small>h</small>",
                   f"{np.average(od.transit_minutes, weights=od.boxes):,.0f} minutes"),
            th.kpi("Average delay", f"{np.average(od.delay_minutes, weights=od.boxes):,.0f} <small>min</small>", "per box"),
            th.kpi("Lost to quality", f"{od.boxes_lost_quality.sum():,.0f}", f"{od.boxes_lost_quality.sum() / b:.2%} of boxes"),
        ])

# ---- Fila 5: modelo XGBoost ----
st.write("")
th.section("Disruption models (XGBoost): what drives each prediction and how well it separates risky days",
           f"Trained on {metrics['n_train']} days, tested on the following {metrics['n_test']} days (temporal split, no shuffling).")
keys = list(metrics["models"])
key = st.segmented_control("Disruption type", keys, default=keys[0], format_func=lambda k: metrics["models"][k]["title"]) or keys[0]
m = metrics["models"][key]
(tn, fp), (fn, tp) = m["confusion_matrix"]
c1, c2, c3 = st.columns([1.1, 1, 0.95], gap="medium")
with c1:
    st.plotly_chart(charts.importance_chart(metrics, key), use_container_width=True, config=th.PLOTLY_CONFIG)
with c2:
    st.plotly_chart(charts.roc_chart(metrics, key), use_container_width=True, config=th.PLOTLY_CONFIG)
with c3:
    st.markdown(th.kpi("AUC on unseen year", f"{m['auc']:.2f}", f"disruption on {m['base_rate_test']:.1%} of test days"), unsafe_allow_html=True)
    st.write("")
    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual normal", "Actual disrupted"], columns=["Pred. normal", "Pred. disrupted"])
    st.dataframe(cm, width="stretch")
    st.caption(f"Confusion matrix at a 50% threshold. Recall {tp / max(tp + fn, 1):.0%}, precision {tp / max(tp + fp, 1):.0%}.")
