"""
Página 1 – Scenario setup: el usuario arma el escenario y lanza la optimización.
"""
import datetime as dt

import streamlit as st

from utils import loaders as L
from utils import maps
from utils import theme as th

data = L.load_data()
p = L.get_params()
start, end, showcase = L.test_dates()

th.hero(
    "Personal project · logistics optimisation",
    "BloomChain – Low-emission cold chain for Colombian flowers",
    "Colombia flies almost every rose, carnation and chrysanthemum it exports to Europe. BloomChain decides which "
    "refrigerated consolidation hubs to open, which flows can travel in reefer containers by sea instead of by air, "
    "and which European chain takes each box to the customer: truck, rail, Rhine barge, feeder ship, Baltic ferry or a "
    "connecting flight. Pick the market (the Netherlands, France, Switzerland, Sweden or Finland), balance cost, CO₂, "
    "transit time and vase life, and let a machine-learning model forecast tomorrow's disruptions so the plan re-optimises around them.",
)

c1, c2, c3 = st.columns(3)
steps = [
    ("1", "Cluster", "k-medoids (PAM) groups 40 farms and proposes real farm sites as candidate cold hubs."),
    ("2", "Predict", "Four XGBoost classifiers estimate tomorrow's probability of fog, strikes, port congestion and storms."),
    ("3", "Optimise", "A mixed-integer model opens hubs, activates air or sea lanes and routes every box. Closed nodes are avoided."),
]
for col, (n, title, text) in zip((c1, c2, c3), steps):
    col.markdown(f'<div class="card"><h4><span class="step-num">{n}</span>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

st.write("")
st.subheader("Build your scenario")

# ----------------------------------------------------------------------------
# Mercado de destino
# ----------------------------------------------------------------------------
th.section("Destination market", "Where the flowers are sold. Each market has its own customers, European gateways and multimodal chains.")
markets = data["markets"].set_index("market")
market = st.segmented_control("Destination market", L.market_codes(), default=p["market"], key="market_pick",
                              format_func=lambda c: markets.loc[c, "name"], label_visibility="collapsed") or p["market"]
m = L.market_info(market)
chains = data["chains"][data["chains"].market == market]
lanes = data["lanes"][data["lanes"].dest.isin(chains.arrival)]
nodes = data["nodes"].set_index("node_id")

mk_left, mk_right = st.columns([1, 1.3], gap="medium")
with mk_left:
    if market == "NL":
        sizing = (f"The Dutch auctions buy for re-export to the whole of Europe, so they take 88% of the farms' "
                  f"weekly supply: <b>{m['weekly_boxes']:,} boxes</b>.")
    else:
        sizing = (f"{m['population_m']:g} M people × €{m['spend_eur_capita']} per person per year = €{m['spend_meur_year']:,} M "
                  f"of cut flowers a year × 2.6 boxes per €M = <b>{m['weekly_boxes']:,} boxes / week</b>.")
    used_modes = list(dict.fromkeys(mode for ms in chains.modes for mode in ms.split("+")))
    pills = "".join(f'<span class="mode-pill"><i style="background:{th.EU_MODE_COLOR[x]}"></i>{th.EU_MODE_LABEL[x]}</span>' for x in used_modes)
    gws = "".join(
        f'<span class="mode-pill"><i style="background:{th.MODE_COLOR[l.mode]}"></i>{nodes.loc[l.origin, "short"]} → {nodes.loc[l.dest, "short"]}'
        f' · {l.time_h / 24:.1f} d</span>' for l in lanes.itertuples())
    cities = ", ".join(nodes.loc[d, "short"] for d in dict.fromkeys(chains.dest_id))
    multimodal = int((chains.modes != "truck").sum())
    card = f"""
    <div class="market">
      <div class="head">{th.flag(market, 26)}<span class="name">{m['name']}</span></div>
      <div class="role">{m['role']}.</div>
      <div class="stats">
        <div class="stat"><div class="v">{m['weekly_boxes']:,}</div><div class="l">boxes / week</div></div>
        <div class="stat"><div class="v">{len(lanes)}</div><div class="l">air & sea lanes</div></div>
        <div class="stat"><div class="v">{len(chains)}</div><div class="l">European chains · {multimodal} multimodal</div></div>
      </div>
      <div class="sub">Customers</div><div class="role" style="margin:0 0 4px 0">{cities}</div>
      <div class="sub">Transatlantic lanes into this market</div>{gws}
      <div class="sub">European modes the model can combine</div>{pills}
      <div class="sub">Weekly order</div><div class="role" style="margin:0">{sizing}</div>
    </div>"""
    # Sin sangría: markdown leería las líneas con sangría como bloque de código
    st.markdown(" ".join(line.strip() for line in card.splitlines()), unsafe_allow_html=True)
with mk_right:
    st.plotly_chart(maps.market_preview(data, market), use_container_width=True, config=th.PLOTLY_CONFIG, key=f"preview_{market}")
    st.caption("Every European chain the model can choose from (dotted: the end of the transatlantic lanes). "
               "Hover a line for its distance and hours; the optimiser decides which ones carry flowers.")

st.write("")

a, b, c = st.columns(3, gap="large")

with a:
    th.section("Network design", "Where the cold hubs can be and how big they are.")
    n_range = st.slider("Cold hubs to open (minimum – maximum)", 1, 8, tuple(p["n_range"]))
    max_farm_hub_km = st.slider("Maximum farm → hub distance (km by road)", 10, 60, int(p["max_farm_hub_km"]), step=5)
    min_hub_gap_km = st.slider("Minimum distance between open hubs (km)", 0, 30, int(p["min_hub_gap_km"]))
    max_hub_capacity = st.slider("Maximum capacity per hub (boxes / week)", 1500, 5000, int(p["max_hub_capacity"]), step=250)
    source_labels = {"both": "Pre-screened sites + k-medoids medoids", "sites": "8 pre-screened sites only", "medoids": "k-medoids medoids only"}
    candidate_source = st.radio("Candidate hub sites", list(source_labels), index=list(source_labels).index(p["candidate_source"]),
                                format_func=source_labels.get)

with b:
    th.section("Sustainability and quality", "The trade-off between cost, emissions and freshness.")
    carbon_price = st.slider("Carbon price (€ per tonne of CO₂)", 0, 300, int(p["carbon_price"]), step=10)
    min_vase_life = st.slider("Minimum vase life on arrival (days)", 6.0, 10.5, float(p["min_vase_life"]), step=0.5,
                              help="Routes that would deliver a flower with fewer vase-life days are removed from the model.")
    quality_pct = st.slider("Value lost per vase-life day lost (% of box value)", 0.0, 12.0, float(p["quality_pct"]), step=0.5,
                            help="Quality penalty in the objective: slower routes cost freshness.")
    flowers = st.multiselect("Flower types", list(th.FLOWER_LABEL), default=p["flowers"], format_func=th.FLOWER_LABEL.get)

with c:
    th.section("Disruption forecast", "XGBoost decides which nodes and lanes are closed tomorrow.")
    date = st.date_input("Date to predict", value=dt.date.fromisoformat(p["date"]), min_value=dt.date.fromisoformat(start),
                         max_value=dt.date.fromisoformat(end),
                         help="Any day of the 12-month hold-out period that the models never saw during training.")
    threshold = st.slider("Closure threshold (probability)", 0.10, 0.90, float(p["threshold"]), step=0.05,
                          help="If P(disruption) ≥ threshold, the node or lane is closed (z = 0) and the plan is re-optimised.")
    pred = L.predict(date.isoformat())
    manual_labels = {k: v["title"] for k, v in pred["probs"].items()}
    manual = st.multiselect("What-if: force a closure", list(manual_labels), default=p["manual"], format_func=manual_labels.get)

    # Vista previa del pronóstico para la fecha elegida
    rows = ""
    for key, v in pred["probs"].items():
        closed = v["prob"] >= threshold or key in manual
        color = th.CRITICAL if closed else th.INK2
        tag = "CLOSED" if closed else "open"
        rows += (
            f'<div style="margin:9px 0 0 0;font-size:0.84rem;color:{th.INK2};display:flex;justify-content:space-between">'
            f'<span>{v["title"]}</span><span style="color:{color};font-weight:600">{v["prob"]:.0%} · {tag}</span></div>'
            f'<div style="background:#22262f;border-radius:3px;height:6px;margin-top:4px">'
            f'<div style="width:{v["prob"] * 100:.0f}%;background:{color};height:6px;border-radius:3px"></div></div>'
        )
    st.markdown(f'<div class="card" style="padding:10px 16px 14px 16px">{rows}</div>', unsafe_allow_html=True)

st.write("")
left, mid, right = st.columns([1, 2, 1])
with mid:
    run = st.button("Run optimization", type="primary", use_container_width=True, icon=":material/play_arrow:")
    if st.button("Reset to defaults", use_container_width=True, type="tertiary"):
        st.session_state.pop("params", None)
        st.session_state.pop("market_pick", None)
        st.rerun()

if run:
    if not flowers:
        st.error("Select at least one flower type.")
    else:
        new = dict(market=market, n_range=tuple(n_range), max_farm_hub_km=max_farm_hub_km, min_hub_gap_km=min_hub_gap_km,
                   max_hub_capacity=max_hub_capacity, candidate_source=candidate_source, carbon_price=carbon_price,
                   min_vase_life=min_vase_life, quality_pct=quality_pct, flowers=list(flowers), threshold=threshold,
                   manual=list(manual), date=date.isoformat())
        st.session_state["params"] = new
        with st.spinner("Forecasting disruptions and solving the mixed-integer model…"):
            L.compute(new)
        st.session_state["fresh"] = True
        st.switch_page("pages/2_Optimization_panel.py")
