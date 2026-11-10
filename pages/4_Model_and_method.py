"""
Página 4 – Model & method: formulación matemática y supuestos, para quien quiera ver el detalle.
"""
import pandas as pd
import streamlit as st

from optimization import network as net
from utils import loaders as L
from utils import theme as th

data = L.load_data()
market = L.get_params()["market"]

st.title("Model & method")
st.caption("How BloomChain works under the hood: three small, transparent models chained together.")

c1, c2, c3 = st.columns(3)
cards = [
    ("1", "k-medoids (PAM)", "Farms are clustered per region with a hand-written PAM (BUILD + SWAP) on Haversine distances. "
                              "k is chosen by silhouette. Medoids are real farms, so they become candidate hub sites."),
    ("2", "XGBoost classifiers", "One binary model per disruption type, trained on two years of daily data and tested on the following "
                                  "year. If P(disruption) ≥ threshold, the affected lane or arc gets z = 0."),
    ("3", "Mixed-integer model", "A fixed-charge location + transshipment model in PuLP, solved with CBC. It is re-solved with the "
                                 "closed elements removed, and compared with the ADD and DROP greedy heuristics."),
]
for col, (n, t, txt) in zip((c1, c2, c3), cards):
    col.markdown(f'<div class="card"><h4><span class="step-num">{n}</span>{t}</h4><p>{txt}</p></div>', unsafe_allow_html=True)

st.write("")
left, right = st.container(), st.container()

with left:
    st.subheader("Mathematical formulation")
    st.markdown(
        "**Sets.** Farms $F$, candidate hubs $H$, lanes $L$ (each lane $l$ joins a Colombian gateway to a European arrival "
        "node by air or sea), European chains $E$ (each chain $e$ leaves the arrival node of a lane and ends at one customer "
        "$d(e)$, by truck or by a multimodal combination of rail, barge, feeder ship, ferry or connecting flight), customers $D$ "
        "of the chosen market, flower types $P$. A *route* is a triple $(h,l,e)$ with $e$ leaving the arrival node of $l$."
    )
    st.markdown("**Decision variables.**")
    st.latex(r"""
    \begin{aligned}
    y_h &\in \{0,1\} && \text{open cold hub } h\\
    m_l &\in \{0,1\} && \text{activate lane } l \text{ (air or sea)}\\
    x_{fhp} &\ge 0 && \text{boxes of flower } p \text{ from farm } f \text{ to hub } h\\
    w_{hlep} &\ge 0 && \text{boxes of } p \text{ from hub } h \text{ via lane } l \text{ and European chain } e\\
    v_{lp},\; u_{ldp} &\ge 0 && \text{boxes on lane } l \text{ and from its arrival node to } d\\
    s_{dp} &\ge 0 && \text{demand not delivered}\\
    z_l,\; z_{hl} &\in \{0,1\} && \text{availability, fixed by the forecast: } z = 0 \text{ if } \hat{P}(\text{disruption}) \ge \tau
    \end{aligned}
    """)
    st.markdown("**Objective.** Minimise fixed costs, transport, carbon cost and the quality penalty:")
    st.latex(r"""
    \begin{aligned}
    \min \;\; & \sum_{h} F_h\, y_h + \sum_{l} G_l\, m_l && \text{fixed cost of hubs and lanes}\\
    & + \sum_{f,h,p} \big(c_{fh} + \pi\, g_{fh} + q_p\, \delta_{fh}\big)\, x_{fhp} && \text{farm} \to \text{hub: transport, carbon, quality}\\
    & + \sum_{h,l,e,p} \big(c_{hle} + \pi\, g_{hle} + q_p\, \delta_{hle}\big)\, w_{hlep} && \text{hub} \to \text{customer: transport, carbon, quality}\\
    & + M \sum_{d,p} s_{dp} && \text{penalty for demand not delivered}
    \end{aligned}
    """)
    st.caption("c = € per box, g = kg CO₂ per box, π = carbon price (€ per kg), δ = vase-life days lost on the arc or route, "
               "q = € lost per box and per vase-life day, M = penalty per box not delivered.")
    st.markdown("**Constraints.**")
    st.latex(r"""
    \begin{aligned}
    &\textstyle\sum_{h} x_{fhp} \le S_{fp} && \forall f,p && \text{farm supply}\\
    &\textstyle\sum_{f} x_{fhp} = \sum_{l,e} w_{hlep} && \forall h,p && \text{flow balance at each hub}\\
    &\textstyle\sum_{h,e} w_{hlep} = v_{lp} = \sum_{d} u_{ldp} && \forall l,p && \text{balance at gateway and arrival node}\\
    &\textstyle u_{ldp} = \sum_{h} \sum_{e:\, d(e) = d} w_{hlep} && \forall l,d,p && \text{arrival} \to \text{customer through the chains}\\
    &\textstyle\sum_{l} u_{ldp} + s_{dp} = D_{dp} && \forall d,p && \text{demand}\\
    &\textstyle\sum_{f,p} x_{fhp} \le K_h\, y_h && \forall h && \text{hub capacity, only if open}\\
    &\textstyle\sum_{p} v_{lp} \le Q_l\, m_l\, z_l && \forall l && \text{flow only on active, available lanes}\\
    &w_{hlep} = 0 \;\; \text{if } z_{hl} = 0 \;\text{ or }\; V_p - \delta_{hle} - \delta^{\max}_{h} < V^{\min} && \forall h,l,e,p && \text{closures and minimum vase life}\\
    &x_{fhp} = 0 \;\; \text{if } \mathrm{dist}_{fh} > R && \forall f,h,p && \text{maximum farm–hub distance}\\
    &N^{\min} \le \textstyle\sum_h y_h \le N^{\max}, \qquad y_h + y_{h'} \le 1 \;\; \text{if } \mathrm{dist}_{hh'} < G^{\min} && && \text{number and spacing of hubs}
    \end{aligned}
    """)
    st.caption("Vase life depends on the whole path, so flow after the hub is modelled by route (h, l, e): a route that would deliver a "
               "flower with less than the minimum vase life is removed exactly, using the worst first mile of that hub.")

with right:
    st.write("")
    st.subheader("Assumptions behind the numbers")
    st.markdown("Unit of flow: one standard 15 kg box. Distances are Haversine × a road factor.")
    legs = pd.DataFrame([
        dict(Leg="Farm → hub (small truck)", **{"Road factor": net.LEG["farm_hub"]["circuity"], "km/h": net.LEG["farm_hub"]["speed_kmh"],
             "€/box·km": net.LEG["farm_hub"]["eur_box_km"], "kg CO₂/t·km": net.LEG["farm_hub"]["kg_co2_tkm"], "Vase days lost per day": net.LEG["farm_hub"]["vase_loss"]}),
        dict(Leg="Hub → airport / port (reefer truck)", **{"Road factor": net.LEG["hub_gateway"]["circuity"], "km/h": net.LEG["hub_gateway"]["speed_kmh"],
             "€/box·km": net.LEG["hub_gateway"]["eur_box_km"], "kg CO₂/t·km": net.LEG["hub_gateway"]["kg_co2_tkm"], "Vase days lost per day": net.LEG["hub_gateway"]["vase_loss"]}),
        dict(Leg="Air lane", **{"Road factor": None, "km/h": net.MODE["air"]["speed_kmh"], "€/box·km": net.MODE["air"]["eur_box_km"],
             "kg CO₂/t·km": net.MODE["air"]["kg_co2_tkm"], "Vase days lost per day": net.MODE["air"]["vase_loss"]}),
        dict(Leg="Sea lane (reefer container)", **{"Road factor": None, "km/h": net.MODE["sea"]["speed_kmh"], "€/box·km": net.MODE["sea"]["eur_box_km"],
             "kg CO₂/t·km": net.MODE["sea"]["kg_co2_tkm"], "Vase days lost per day": net.MODE["sea"]["vase_loss"]}),
    ])
    st.dataframe(legs, hide_index=True, width="stretch", column_config={"€/box·km": st.column_config.NumberColumn(format="%.5f")})

    st.markdown("**European modes** (last leg, from the European arrival to the customer)")
    eu = pd.DataFrame([dict(Mode=v["label"], **{
        "km/h": v["speed_kmh"],
        "€/box·km": (f"{v['eur_box_km']['pallet']:.4f} / {v['eur_box_km']['container']:.4f}"
                     if isinstance(v["eur_box_km"], dict) else f"{v['eur_box_km']:.4f}"),
        "Terminal €/box": v["eur_box_fixed"], "Terminal h": v["fixed_h"], "kg CO₂/t·km": v["kg_co2_tkm"]}) for v in net.EU_MODE.values()])
    st.dataframe(eu, hide_index=True, width="stretch")
    st.caption("Truck cost: €0.0030 per box·km for pallets after a flight / €0.0044 for a full sea container. "
               "Vase-life days lost per day in Europe: 0.4 on pallets after a flight, 0.2 inside the sealed reefer container after the "
               "ship, 0.9 on a connecting flight. River and sea legs follow drawn waterways; trucks to Sweden cross the Danish bridges.")

    st.markdown("**Market size** (weekly order from this group of farms)")
    mk = data["markets"].assign(rule=["88% of supply (auction re-export)" if c == "NL" else f"{pop:g} M × €{sp} × 2.6 boxes / €M"
                                      for c, pop, sp in zip(data["markets"].market, data["markets"].population_m,
                                                            data["markets"].spend_eur_capita)])
    st.dataframe(mk[["name", "population_m", "spend_eur_capita", "spend_meur_year", "rule", "weekly_boxes"]].rename(columns={
        "name": "Market", "population_m": "People (M)", "spend_eur_capita": "€ per person / year", "spend_meur_year": "€ M / year",
        "rule": "Rule", "weekly_boxes": "Boxes / week"}), hide_index=True, width="stretch")

    t1, t2 = st.columns([1.5, 1], gap="large")
    t1.markdown(f"**Lanes into the selected market ({L.market_info(market)['name']})**")
    t2.markdown("**Flowers**")
    lanes = data["lanes"][data["lanes"].dest.isin(data["chains"].loc[data["chains"].market == market, "arrival"])]
    lanes = lanes.assign(days=lanes.time_h / 24)[["lane_id", "distance_km", "days", "cost_per_box", "co2_kg_per_box", "capacity_boxes", "fixed_cost_week"]]
    t1.dataframe(lanes.rename(columns={"lane_id": "Lane", "distance_km": "km", "days": "Days", "cost_per_box": "€/box", "co2_kg_per_box": "kg CO₂/box",
                                       "capacity_boxes": "Capacity (boxes/wk)", "fixed_cost_week": "Fixed (€/wk)"}),
                 hide_index=True, width="stretch", column_config={"Days": st.column_config.NumberColumn(format="%.1f")})

    t2.dataframe(data["flowers"].rename(columns={"label": "Flower", "stems_per_box": "Stems per box", "vase_life_days": "Vase life at harvest (days)",
                                                 "box_value_eur": "Box value (€)"}).drop(columns="flower"), hide_index=True, width="stretch")

    st.info("All figures are synthetic and illustrative: farm locations are approximate, and costs, emission factors and "
            "vase-life rates are order-of-magnitude assumptions chosen to make the trade-offs visible, not measured data.")
    st.markdown("**What a thesis would add.** Real cost and emission data, stochastic transit times, temperature-dependent quality "
                "decay, time-windowed auction deliveries, capacities and timetables on the European rail, barge and feeder services, "
                "European disruptions (low water on the Rhine, Baltic ice) and a rolling-horizon version of the re-optimisation.")
