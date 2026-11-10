"""
Página 2 – Optimization panel: KPIs, mapa global, animación, pronóstico y comparación antes / después.
"""
import pandas as pd
import streamlit as st

from optimization.milp import CLOSURES, UNMET_PENALTY
from utils import charts, loaders as L, maps
from utils import theme as th

out = L.get_results()
res, p, pred = out["res"], out["params"], out["pred"]
after, before, all_air = res["after"], res["before"], res["all_air"]
active = out["active"]
data, cl = L.load_data(), L.clusters()
cand = res["candidates"]

# ----------------------------------------------------------------------------
# Encabezado y alertas
# ----------------------------------------------------------------------------
mk = L.market_info(res["market"])
st.markdown(f'<h1 style="display:flex;align-items:center;gap:14px">Optimization panel '
            f'<span style="font-size:1rem;font-weight:500;color:{th.INK2};display:inline-flex;align-items:center;gap:8px">'
            f'{th.flag(res["market"], 20)} {mk["name"]} · {mk["weekly_boxes"]:,} boxes / week</span></h1>', unsafe_allow_html=True)
chips = [f"Departure {pd.Timestamp(p['date']):%d %b %Y}", f"Carbon €{p['carbon_price']}/t", f"Vase life ≥ {p['min_vase_life']:g} days",
         f"Quality {p['quality_pct']:g}% per day", f"Threshold {p['threshold']:.0%}", f"Hubs {p['n_range'][0]}–{p['n_range'][1]}",
         f"Farm → hub ≤ {p['max_farm_hub_km']} km"]
st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

if active:
    lines = []
    for key in active:
        v = pred["probs"][key]
        why = "forced as a what-if" if (key in p["manual"] and v["prob"] < p["threshold"]) else f"P = {v['prob']:.0%} ≥ {p['threshold']:.0%}"
        lines.append(f"<b>{v['title']}</b> ({why}) → {v['effect']}.")
    st.markdown('<div class="alert"><div class="dot"></div><div><b>Disruption alert.</b> ' + " ".join(lines)
                + " The plan below has been re-optimised around the closure.</div></div>", unsafe_allow_html=True)
    if st.session_state.pop("fresh", False):
        for key in active:
            st.toast(f"{pred['probs'][key]['title']}: {pred['probs'][key]['effect']}", icon="🚨")
else:
    st.session_state.pop("fresh", None)
    st.markdown('<div class="ok-banner">No disruption above the threshold for this date: the whole network is available. '
                "Try another date or force a closure in <b>Scenario setup</b>.</div>", unsafe_allow_html=True)

if after["unmet"] > 0.5:
    st.warning(f"{after['unmet']:,.0f} boxes of weekly demand cannot be delivered with these settings "
               "(not enough reachable hub capacity, lane capacity or vase life). Relax a constraint in Scenario setup.")

# ----------------------------------------------------------------------------
# Tarjetas KPI
# ----------------------------------------------------------------------------
pct = lambda new, old: (new / old - 1.0) * 100.0 if old else 0.0
cost, co2 = after["cost"]["total"], after["co2_kg"] / 1000.0
hub_names = cand.set_index("hub_id").name.str.replace(" cold hub", "", regex=False)
vs_before = f" · {th.delta_html(pct(cost, before['cost']['total']))} vs plan" if active else ""
th.kpi_row([
    th.kpi("Total weekly cost", th.fmt_eur(cost), f"€{cost / max(after['delivered'], 1):.2f} per box{vs_before}"),
    th.kpi("CO₂ emissions", f"{co2:,.0f} <small>t / week</small>", th.delta_html(pct(co2, all_air["co2_kg"] / 1000.0), suffix=" vs all-air")),
    th.kpi("Share by sea", f"{after['sea_share']:.0%} <small>sea · {1 - after['sea_share']:.0%} air</small>",
           f"{after['boxes_sea']:,.0f} sail, {after['boxes_air']:,.0f} fly · {after['multimodal_share']:.0%} multimodal in Europe",
           bar=[(th.SEA, after["sea_share"]), (th.AIR, 1 - after["sea_share"])]),
    th.kpi("Average vase life", f"{after['avg_vase_life']:.1f} <small>days</small>",
           f"minimum {p['min_vase_life']:g} · transit {after['avg_transit_h'] / 24:.1f} days on average"),
    th.kpi("Cold hubs open", f"{len(after['open_hubs'])} <small>of {len(cand)} candidates</small>",
           ", ".join(hub_names[h] for h in after["open_hubs"]) or "none"),
    th.kpi("Saving vs all-air", th.fmt_eur(all_air["cost"]["total"] - cost) + " <small>/ week</small>",
           th.delta_html(pct(cost, all_air["cost"]["total"]), suffix=" total cost")),
])
st.write("")

# ----------------------------------------------------------------------------
# Mapa global + medidores de probabilidad
# ----------------------------------------------------------------------------
left, right = st.columns([2.7, 1], gap="medium")
with left:
    c1, c2 = st.columns([1, 1.6])
    view = c1.segmented_control("Plan shown", ["After (re-optimised)", "Before (planned)"] if active else ["Optimal plan"],
                                default="After (re-optimised)" if active else "Optimal plan")
    focus = c2.segmented_control("Zoom to", list(maps.VIEWS), default="Global")
    show_before = bool(view and view.startswith("Before"))
    plan = before if show_before else after
    closed_nodes = set() if show_before else {n for k in active for n in CLOSURES[k]["nodes"]}
    st.pydeck_chart(maps.build_deck(data, cl, res, plan, focus or "Global", closed_nodes), height=540)
    th.legend([(th.AIR, "Air lane"), (th.SEA, "Sea lane"), (th.INK2, "Farm → hub collection"), (th.CRITICAL, "Closed node or lane"),
               ("#f5efec", "Warehouse icon: open hub (grey: candidate not opened)"),
               (th.CLUSTER_COLORS[0], "Farm dots: one colour per k-medoids cluster; white ring = medoid")])
    market_modes = {x for ms in res["chains"].modes for x in ms.split("+")}
    th.legend([(th.EU_MODE_COLOR[x], f"Europe · {th.EU_MODE_LABEL[x]}") for x in th.EU_MODE_COLOR if x in market_modes])
    st.caption("Line width is proportional to weekly boxes; faint lines are options the model did not choose. "
               "Zoom to Europe to see the last leg. Hover any element for details.")

with right:
    th.section(f"Disruption risk · {pd.Timestamp(p['date']):%d %b %Y}", "XGBoost probability; the yellow mark is the closure threshold.")
    for key, v in pred["probs"].items():
        st.plotly_chart(charts.gauge(v["prob"], p["threshold"], v["title"]), use_container_width=True, config=th.PLOTLY_CONFIG, key=f"g_{key}")
    happened = [v["title"] for v in pred["probs"].values() if v["actual"] == 1]
    st.caption("What actually happened that day in the (synthetic) record: " + (", ".join(happened) if happened else "no disruption") + ".")

# ----------------------------------------------------------------------------
# Animación
# ----------------------------------------------------------------------------
st.write("")
th.section("Air vs sea, in motion", "One frame is a quarter of a day. A flight leaves every two days; the reefer vessel needs more than two weeks. Closed nodes pulse in red.")
st.plotly_chart(maps.animation_figure(data, res, after, {n for k in active for n in CLOSURES[k]["nodes"]}), use_container_width=True, config=th.PLOTLY_CONFIG)

# ----------------------------------------------------------------------------
# Secciones de detalle
# ----------------------------------------------------------------------------
# Un selector en vez de pestañas: solo se dibuja la sección elegida
DETAILS = ["Before / after", "European chains", "Route table", "Exact model vs ADD / DROP", "Farm clustering (k-medoids)"]
st.write("")
detail = st.segmented_control("Details", DETAILS, default=DETAILS[0], label_visibility="collapsed") or DETAILS[0]

if detail == DETAILS[0]:
    if not active:
        st.info("No closure in this scenario, so the plan before and after is the same. The comparison below is against the all-air baseline instead.")
    ref, ref_name = (before, "Before (planned)") if active else (all_air, "All-air baseline")
    now_name = "After (re-optimised)" if active else "Optimal plan"
    rows = [
        ("Total weekly cost (€)", ref["cost"]["total"], after["cost"]["total"], "{:,.0f}"),
        ("  Transport (€)", ref["cost"]["transport"], after["cost"]["transport"], "{:,.0f}"),
        ("  Carbon cost (€)", ref["cost"]["carbon"], after["cost"]["carbon"], "{:,.0f}"),
        ("  Quality penalty (€)", ref["cost"]["quality"], after["cost"]["quality"], "{:,.0f}"),
        ("  Fixed costs: hubs + lanes (€)", ref["cost"]["hub_fixed"] + ref["cost"]["lane_fixed"], after["cost"]["hub_fixed"] + after["cost"]["lane_fixed"], "{:,.0f}"),
        ("CO₂ (t / week)", ref["co2_kg"] / 1000, after["co2_kg"] / 1000, "{:,.1f}"),
        ("Share by sea (%)", ref["sea_share"] * 100, after["sea_share"] * 100, "{:.1f}"),
        ("Average vase life (days)", ref["avg_vase_life"], after["avg_vase_life"], "{:.2f}"),
        ("Average transit (days)", ref["avg_transit_h"] / 24, after["avg_transit_h"] / 24, "{:.1f}"),
        ("Boxes not delivered", ref["unmet"], after["unmet"], "{:,.0f}"),
    ]
    table = pd.DataFrame([{"Metric": n, ref_name: f.format(a), now_name: f.format(b),
                           "Change": "–" if abs(a) < 1e-9 else f"{(b / a - 1) * 100:+.1f}%"} for n, a, b, f in rows])
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        st.dataframe(table, hide_index=True, width="stretch", height=388, column_config={
            "Metric": st.column_config.TextColumn(width="medium"), ref_name: st.column_config.TextColumn(width="small"),
            now_name: st.column_config.TextColumn(width="small"), "Change": st.column_config.TextColumn(width="small")})
    with c2:
        st.plotly_chart(charts.lanes_before_after(res["lanes"], ref, after), use_container_width=True, config=th.PLOTLY_CONFIG)
        st.caption(f"Faded bars: {ref_name.lower()}. Solid bars: {now_name.lower()}. Orange = air, blue = sea.")

elif detail == DETAILS[1]:
    chains = res["chains"]
    plan_r = after["routes"]
    c1, c2 = st.columns([1, 1.05], gap="large")
    with c1:
        th.section("Every way to cover the last leg in Europe",
                   "Each dot is one chain from a European gateway to a customer. Big, outlined dots are the ones the plan uses; "
                   "faint dots are options the optimiser rejected. Left and low is fast and cheap.")
        st.plotly_chart(charts.chain_tradeoff(chains, plan_r), use_container_width=True, config=th.PLOTLY_CONFIG)
    with c2:
        used = plan_r.groupby("chain_id").boxes.sum() if len(plan_r) else pd.Series(dtype=float)
        t = chains.assign(
            From=chains.arrival.map(th.NODE_LABEL), To=chains.dest_id.map(th.DEST_LABEL),
            Via=chains.via.map(lambda v: " → ".join(th.NODE_LABEL.get(x, x) for x in str(v).split()) or "direct"),
            boxes=chains.chain_id.map(used).fillna(0.0),
        ).sort_values(["boxes", "cost_per_box"], ascending=[False, True])
        t = t[["From", "To", "label", "boxes", "time_h", "cost_per_box", "co2_kg_per_box", "vase_loss", "Via"]]
        st.dataframe(t, hide_index=True, width="stretch", height=380, column_config={
            "From": st.column_config.TextColumn(width="small"), "To": st.column_config.TextColumn(width="small"),
            "label": st.column_config.TextColumn("Chain", width="medium"),
            "boxes": st.column_config.ProgressColumn("Boxes / week", format="%.0f", min_value=0, max_value=max(float(t.boxes.max()), 1.0)),
            "time_h": st.column_config.NumberColumn("Hours", format="%.0f"),
            "cost_per_box": st.column_config.NumberColumn("€ / box", format="%.2f"),
            "co2_kg_per_box": st.column_config.NumberColumn("kg CO₂ / box", format="%.2f"),
            "vase_loss": st.column_config.NumberColumn("Vase days lost", format="%.2f"),
        })
        st.caption("After a flight the flowers travel on pallets in a reefer truck and lose 0.4 vase-life days per day. After the ship "
                   "they stay in the same sealed, controlled-atmosphere container (0.2 per day), whatever carries it: truck, train, barge or feeder.")

elif detail == DETAILS[2]:
    plan_r = after["routes"]
    if not len(plan_r):
        st.info("No flow in this plan.")
    else:
        t = plan_r.assign(
            Hub=plan_r.hub_id.map(hub_names), Gateway=plan_r.gateway.map(th.NODE_LABEL), Mode=plan_r["mode"].map(th.MODE_LABEL),
            Arrival=plan_r.arrival.map(th.NODE_LABEL), Destination=plan_r.dest_id.map(th.DEST_LABEL),
            Flower=plan_r.flower.map(th.FLOWER_LABEL), transit_days=plan_r.transit_h / 24, **{"European leg": plan_r.chain_label},
        )[["Hub", "Gateway", "Mode", "Arrival", "European leg", "Destination", "Flower", "boxes", "cost_per_box", "co2_kg_per_box",
           "transit_days", "vase_at_arrival"]]
        t = t.sort_values("boxes", ascending=False)
        st.dataframe(t, hide_index=True, width="stretch", height=420, column_config={
            "boxes": st.column_config.ProgressColumn("Boxes / week", format="%.0f", min_value=0, max_value=float(t.boxes.max())),
            "cost_per_box": st.column_config.NumberColumn("€ / box (hub → door)", format="%.2f"),
            "co2_kg_per_box": st.column_config.NumberColumn("kg CO₂ / box", format="%.1f"),
            "transit_days": st.column_config.NumberColumn("Transit (days)", format="%.1f"),
            "vase_at_arrival": st.column_config.NumberColumn("Vase life on arrival (days)", format="%.1f"),
        })
        st.download_button("Download routes (CSV)", t.to_csv(index=False).encode("utf-8"), "bloomchain_routes.csv", "text/csv")

elif detail == DETAILS[3]:
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        th.section("Which hubs to open: exact optimum vs greedy heuristics",
                   "ADD opens one hub at a time while total cost falls; DROP starts with every hub open and closes one at a time. "
                   "Both use a simplified cost table: the weekly order is filled with the cheapest farm–hub pairs, respecting farm supply and "
                   "hub capacity but not lane capacity. Their hub sets are then priced with the full model.")
        h = res["heuristics"]["table"].copy()
        h["total_cost"] = h.total_cost.round(0)
        h["hubs"] = h.hubs.apply(lambda s: ", ".join(hub_names.get(x, x) for x in s.split(", ") if x))
        h = h[["method", "n_hubs", "total_cost", "gap_pct", "unmet", "hubs"]]
        st.dataframe(h.rename(columns={"method": "Method", "hubs": "Hubs opened", "n_hubs": "Hubs", "total_cost": "Cost (€ / week)",
                                       "unmet": "Not delivered", "gap_pct": "Gap (%)"}),
                     hide_index=True, width="stretch", column_config={
                         "Method": st.column_config.TextColumn(width="medium"),
                         "Hubs": st.column_config.NumberColumn(width="small"),
                         "Cost (€ / week)": st.column_config.NumberColumn(format="localized", width="small"),
                         "Not delivered": st.column_config.NumberColumn(format="%.0f", width="small"),
                         "Gap (%)": st.column_config.NumberColumn(format="%.2f", width="small"),
                         "Hubs opened": st.column_config.TextColumn(width="large")})
        st.caption(f"Total cost includes a penalty of €{UNMET_PENALTY:.0f} per box not delivered. "
                   f"The exact model has {after['n_vars']:,} variables and {after['n_cons']:,} constraints and is solved with CBC (branch and bound).")
    with c2:
        st.plotly_chart(charts.heuristic_curves(res["heuristics"]), use_container_width=True, config=th.PLOTLY_CONFIG)

else:
    regions = list(cl["best_k"])
    region = st.segmented_control("Region", regions, default=regions[0]) or regions[0]
    c1, c2 = st.columns([1, 1.1], gap="large")
    with c1:
        st.plotly_chart(charts.silhouette_chart(cl["scores"], region, cl["best_k"][region]), use_container_width=True, config=th.PLOTLY_CONFIG)
        s = cl["scores"][(cl["scores"].region == region) & (cl["scores"].k == cl["best_k"][region])].iloc[0]
        st.caption(f"With k = {int(s.k)}, the average k-means centroid lies {s.centroid_to_nearest_farm_km:.1f} km from the nearest farm: "
                   "it is a point on the map, not a site. A k-medoids medoid is always a real farm, so it can host a cold hub.")
    with c2:
        m = cl["medoids"][cl["medoids"].region == region]
        st.dataframe(m.assign(cluster=m.cluster + 1)[["cluster", "name", "municipality", "n_farms", "boxes", "max_km"]].rename(columns={
            "cluster": "Cluster", "name": "Medoid farm", "municipality": "Municipality", "n_farms": "Farms", "boxes": "Boxes / week",
            "max_km": "Farthest farm (km)"}), hide_index=True, width="stretch",
            column_config={"Farthest farm (km)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Each region is clustered separately and k is chosen with the silhouette coefficient. "
                   "Medoids farther than 3 km from a pre-screened site are added as extra candidate hubs.")
