"""
Gráficos de plotly de la app. Cada función recibe datos ya calculados y devuelve una figura.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import theme as th


# ----------------------------------------------------------------------------
# Página 2: panel de optimización
# ----------------------------------------------------------------------------
def gauge(prob: float, threshold: float, title: str) -> go.Figure:
    """Medidor de probabilidad de disrupción para mañana, con la línea del umbral."""
    alert = prob >= threshold
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number=dict(suffix="%", font=dict(size=26, color=th.INK), valueformat=".0f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 50, 100], tickfont=dict(color=th.MUTED, size=10), tickcolor=th.AXIS),
            bar=dict(color=th.CRITICAL if alert else th.INK2, thickness=0.6),
            bgcolor="#22262f", borderwidth=0,
            threshold=dict(line=dict(color=th.WARNING, width=3), thickness=0.9, value=threshold * 100),
        ),
        title=dict(text=title, font=dict(size=12.5, color=th.INK2)),
    ))
    fig.update_layout(height=186, margin=dict(l=26, r=26, t=46, b=8), paper_bgcolor=th.SURFACE,
                      font=dict(family='system-ui, "Segoe UI", sans-serif'))
    return fig


def lanes_before_after(lanes: pd.DataFrame, before: dict, after: dict) -> go.Figure:
    """Cajas por lane del mercado, antes y después de la disrupción (barras agrupadas, un solo eje)."""
    b = before["routes"].groupby("lane_id").boxes.sum() if len(before["routes"]) else pd.Series(dtype=float)
    a = after["routes"].groupby("lane_id").boxes.sum() if len(after["routes"]) else pd.Series(dtype=float)
    labels = [f"{'Air' if m == 'air' else 'Sea'} {o}→{d}" for m, o, d in zip(lanes["mode"], lanes.origin, lanes.dest)]
    if len(labels) > 5:                       # con muchos lanes, etiqueta en dos líneas
        labels = [lab.replace(" ", "<br>", 1) for lab in labels]
    colors = [th.MODE_COLOR[m] for m in lanes["mode"]]
    vb = [float(b.get(l, 0)) for l in lanes.lane_id]
    va = [float(a.get(l, 0)) for l in lanes.lane_id]
    fig = go.Figure()
    fig.add_bar(name="Before (planned)", x=labels, y=vb, marker=dict(color=colors, opacity=0.38, cornerradius=4),
                text=[f"{v:,.0f}" for v in vb], textposition="outside", textfont=dict(color=th.MUTED, size=11),
                hovertemplate="%{x}<br>Before: %{y:,.0f} boxes<extra></extra>")
    fig.add_bar(name="After (re-optimised)", x=labels, y=va, marker=dict(color=colors, cornerradius=4),
                text=[f"{v:,.0f}" for v in va], textposition="outside", textfont=dict(color=th.INK, size=11),
                hovertemplate="%{x}<br>After: %{y:,.0f} boxes<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08, yaxis_title="Boxes per week", showlegend=False)
    fig.update_yaxes(range=[0, max(vb + va + [1]) * 1.18])
    return th.style_fig(fig, height=330, legend_top=False)


def heuristic_curves(heur: dict) -> go.Figure:
    """Costo estimado contra número de centros abiertos en ADD y DROP (la curva de clase)."""
    fig = go.Figure()
    add, drop = heur["add_history"], heur["drop_history"]
    if len(add):
        fig.add_scatter(x=add.n_hubs, y=add.est_cost / 1000, mode="lines+markers", name="ADD",
                        line=dict(color=th.CATEGORICAL[0], width=2), marker=dict(size=8), text=add.hub,
                        hovertemplate="ADD · %{x} hubs · opened %{text}<br>€%{y:,.0f}k<extra></extra>")
    if len(drop):
        fig.add_scatter(x=drop.n_hubs, y=drop.est_cost / 1000, mode="lines+markers", name="DROP",
                        line=dict(color=th.CATEGORICAL[1], width=2), marker=dict(size=8), text=drop.hub,
                        hovertemplate="DROP · %{x} hubs · closed %{text}<br>€%{y:,.0f}k<extra></extra>")
    fig.update_layout(xaxis_title="Hubs open", yaxis_title="Estimated weekly cost (€ thousand)")
    fig.update_xaxes(dtick=1)
    # Si el primer paso de ADD es enorme (casi nadie atendido), se recorta el eje para ver la parte útil
    vals = pd.concat([add.est_cost if len(add) else pd.Series(dtype=float), drop.est_cost if len(drop) else pd.Series(dtype=float)]) / 1000
    if len(vals) > 2 and vals.max() > 1.6 * vals.min():
        fig.update_yaxes(range=[vals.min() * 0.97, vals.min() * 1.35])
    return th.style_fig(fig, height=330)


def silhouette_chart(scores: pd.DataFrame, region: str, best_k: int) -> go.Figure:
    """Silueta por k: k-medoids (PAM) contra k-means."""
    s = scores[scores.region == region]
    fig = go.Figure()
    fig.add_scatter(x=s.k, y=s.silhouette_pam, mode="lines+markers", name="k-medoids (PAM)",
                    line=dict(color=th.CATEGORICAL[0], width=2), marker=dict(size=8),
                    hovertemplate="k = %{x}<br>PAM silhouette %{y:.3f}<extra></extra>")
    fig.add_scatter(x=s.k, y=s.silhouette_kmeans, mode="lines+markers", name="k-means",
                    line=dict(color=th.CATEGORICAL[1], width=2), marker=dict(size=8, symbol="diamond"),
                    hovertemplate="k = %{x}<br>k-means silhouette %{y:.3f}<extra></extra>")
    best = s[s.k == best_k]
    if len(best):
        fig.add_annotation(x=best_k, y=float(best.silhouette_pam.iloc[0]), text=f"chosen k = {best_k}", showarrow=True,
                           arrowcolor=th.INK2, font=dict(color=th.INK, size=12), ay=-30)
    fig.update_layout(xaxis_title="Number of clusters k", yaxis_title="Silhouette coefficient")
    fig.update_xaxes(dtick=1)
    return th.style_fig(fig, height=300)


# ----------------------------------------------------------------------------
# Página 3: tablero de analítica
# ----------------------------------------------------------------------------
def sankey(data: dict, plan: dict, candidates: pd.DataFrame) -> go.Figure:
    """Sankey municipio -> centro de acopio -> salida en Colombia -> llegada a Europa -> cliente."""
    farms = data["farms"].set_index("farm_id")
    hub_name = candidates.set_index("hub_id").name.str.replace(" cold hub", " hub", regex=False).to_dict()
    fh, rt = plan["farm_hub"], plan["routes"]
    if not len(fh) or not len(rt):
        return th.style_fig(go.Figure(), height=420)
    lab = th.NODE_LABEL
    rt = rt.assign(eu=rt.eu_modes.map(th.main_eu_mode))
    a = fh.assign(src=fh.farm_id.map(farms.municipality), dst=fh.hub_id.map(hub_name)).groupby(["src", "dst"], as_index=False).boxes.sum()
    b = rt.assign(src=rt.hub_id.map(hub_name), dst=rt.gateway.map(lab)).groupby(["src", "dst", "mode"], as_index=False).boxes.sum()
    c = rt.assign(src=rt.gateway.map(lab), dst=rt.arrival.map(lambda x: lab[x] + " ")).groupby(["src", "dst", "mode"], as_index=False).boxes.sum()
    d = rt.assign(src=rt.arrival.map(lambda x: lab[x] + " "), dst=rt.dest_id.map(th.DEST_LABEL)).groupby(["src", "dst", "eu"], as_index=False).boxes.sum()
    names = list(dict.fromkeys(list(a.src) + list(a.dst) + list(b.dst) + list(c.dst) + list(d.dst)))
    idx = {n: i for i, n in enumerate(names)}
    kind = data["nodes"].set_index("node_id").kind
    air_nodes = {lab[i] for i in set(rt.gateway) | set(rt.arrival) if kind[i] == "airport"} | {lab[i] + " " for i in set(rt.arrival) if kind[i] == "airport"}
    sea_nodes = {lab[i] for i in set(rt.gateway) | set(rt.arrival) if kind[i] == "seaport"} | {lab[i] + " " for i in set(rt.arrival) if kind[i] == "seaport"}
    node_color = [th.AIR if n in air_nodes else th.SEA if n in sea_nodes else "#6b7280" for n in names]
    rgba = lambda h, al: "rgba({},{},{},{})".format(*th.hex_to_rgb(h), al)
    src = [idx[x] for x in list(a.src) + list(b.src) + list(c.src) + list(d.src)]
    dst = [idx[x] for x in list(a.dst) + list(b.dst) + list(c.dst) + list(d.dst)]
    val = list(a.boxes) + list(b.boxes) + list(c.boxes) + list(d.boxes)
    col = (["rgba(195,194,183,0.22)"] * len(a) + [rgba(th.MODE_COLOR[m], 0.45) for m in b["mode"]]
           + [rgba(th.MODE_COLOR[m], 0.45) for m in c["mode"]] + [rgba(th.EU_MODE_COLOR[m], 0.55) for m in d["eu"]])
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[n.strip() for n in names], color=node_color, pad=12, thickness=14, line=dict(color=th.SURFACE, width=2),
                  hovertemplate="%{label}<br>%{value:,.0f} boxes / week<extra></extra>"),
        link=dict(source=src, target=dst, value=val, color=col, hovertemplate="%{source.label} → %{target.label}<br>%{value:,.0f} boxes / week<extra></extra>"),
        textfont=dict(color=th.INK, size=11),
    ))
    return th.style_fig(fig, height=490, legend_top=False)


def chain_tradeoff(chains: pd.DataFrame, routes: pd.DataFrame) -> go.Figure:
    """Cada cadena europea como un punto: horas contra euros por caja. Las elegidas, grandes y con borde."""
    used = routes.groupby("chain_id").boxes.sum() if len(routes) else pd.Series(dtype=float)
    c = chains.assign(main=chains.modes.map(th.main_eu_mode), boxes=chains.chain_id.map(used).fillna(0.0))
    ref = max(float(c.boxes.max()), 1.0)
    fig = go.Figure()
    for mode, g in c.groupby("main", sort=False):
        fig.add_scatter(
            x=g.time_h, y=g.cost_per_box, mode="markers", name=th.EU_MODE_LABEL[mode],
            marker=dict(color=th.EU_MODE_COLOR[mode], size=[9 + 22 * (b / ref) ** 0.5 if b > 0.5 else 8 for b in g.boxes],
                        opacity=[0.95 if b > 0.5 else 0.35 for b in g.boxes],
                        line=dict(color=[th.INK if b > 0.5 else th.SURFACE for b in g.boxes], width=1.5)),
            customdata=np.stack([g.arrival.map(th.NODE_LABEL), g.dest_id.map(th.DEST_LABEL), g.label, g.boxes, g.co2_kg_per_box], axis=1),
            hovertemplate="%{customdata[0]} → %{customdata[1]}<br>%{customdata[2]}<br>%{x:.0f} h · €%{y:.2f} per box · "
                          "%{customdata[4]:.2f} kg CO₂<br>%{customdata[3]:,.0f} boxes / week<extra></extra>")
    fig.update_layout(xaxis_title="Hours from European arrival to customer", yaxis_title="€ per box (European leg)")
    return th.style_fig(fig, height=360)


def treemap(ship: pd.DataFrame) -> go.Figure:
    """Cajas enviadas por destino y tipo de flor."""
    g = ship.groupby(["destination", "flower"], as_index=False).boxes.sum()
    g["dest"] = g.destination.map(th.DEST_LABEL)
    g["fl"] = g.flower.map(th.FLOWER_LABEL)
    tot = g.groupby("dest").boxes.sum()
    labels = list(tot.index) + list(g.fl)
    ids = list(tot.index) + [f"{d}/{f}" for d, f in zip(g.dest, g.fl)]
    parents = [""] * len(tot) + list(g.dest)
    values = list(tot.values) + list(g.boxes)
    colors = ["#22262f"] * len(tot) + [th.FLOWER_COLOR[f] for f in g.flower]
    fig = go.Figure(go.Treemap(
        labels=labels, ids=ids, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=colors, line=dict(color=th.SURFACE, width=2)), textfont=dict(color="#ffffff", size=12),
        texttemplate="%{label}<br>%{value:,.0f}", hovertemplate="%{id}<br>%{value:,.0f} boxes<extra></extra>",
        tiling=dict(pad=2), pathbar=dict(visible=False),
    ))
    return th.style_fig(fig, height=380, legend_top=False)


def violins(ship: pd.DataFrame) -> go.Figure:
    """Minutos en tránsito por modo. Dos paneles porque las escalas son muy distintas (1 día contra 16)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Air", "Sea"), horizontal_spacing=0.12)
    for i, mode in enumerate(("air", "sea"), start=1):
        s = ship[ship["mode"] == mode]
        if len(s) > 2500:
            s = s.sample(2500, random_state=1)
        fig.add_trace(go.Violin(y=s.transit_minutes, name=th.MODE_LABEL[mode], line=dict(color=th.MODE_COLOR[mode], width=1.5),
                                fillcolor="rgba({},{},{},0.30)".format(*th.hex_to_rgb(th.MODE_COLOR[mode])),
                                box_visible=True, meanline_visible=True, points=False, spanmode="hard",
                                hoverinfo="y", showlegend=False), row=1, col=i)
        fig.update_yaxes(title_text="Transit time (minutes)" if i == 1 else None, tickformat=",", row=1, col=i)
        fig.update_xaxes(showticklabels=False, row=1, col=i)
    fig.update_annotations(font=dict(color=th.INK2, size=12))
    return th.style_fig(fig, height=380, legend_top=False)


def control_chart(ship: pd.DataFrame) -> tuple[go.Figure, int]:
    """Gráfico de control del retraso medio diario: línea central y límites de media ± 3 sigma."""
    daily = ship.groupby("date").apply(lambda d: np.average(d.delay_minutes, weights=d.boxes), include_groups=False).rename("delay").reset_index()
    mean, sigma = daily.delay.mean(), daily.delay.std()
    ucl, lcl = mean + 3 * sigma, max(mean - 3 * sigma, 0.0)
    out = daily[(daily.delay > ucl) | (daily.delay < lcl)]
    fig = go.Figure()
    fig.add_scatter(x=daily.date, y=daily.delay, mode="lines", name="Daily mean delay", line=dict(color=th.INK2, width=1.6),
                    hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} min<extra></extra>")
    fig.add_scatter(x=out.date, y=out.delay, mode="markers", name="Out of control",
                    marker=dict(color=th.CRITICAL, size=9, symbol="diamond", line=dict(color=th.SURFACE, width=2)),
                    hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} min · out of control<extra></extra>")
    for y, text, color, dash in ((ucl, f"UCL = mean + 3σ = {ucl:,.0f}", th.WARNING, "dash"), (mean, f"Mean = {mean:,.0f}", th.MUTED, "solid"),
                                 (lcl, f"LCL = {lcl:,.0f}", th.WARNING, "dash")):
        fig.add_hline(y=y, line=dict(color=color, width=1.2, dash=dash), annotation_text=text,
                      annotation_position="top left", annotation_font=dict(color=th.INK2, size=11))
    fig.update_layout(yaxis_title="Delay (minutes, weighted by boxes)")
    return th.style_fig(fig, height=360), len(out)


def importance_chart(metrics: dict, key: str, top: int = 10) -> go.Figure:
    """Importancia de variables del XGBoost (una sola serie: un solo color)."""
    imp = pd.Series(metrics["models"][key]["importance"]).head(top)[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=[n.replace("_", " ") for n in imp.index], orientation="h",
                           marker=dict(color=th.CATEGORICAL[0], cornerradius=4),
                           text=[f"{v:.2f}" for v in imp.values], textposition="outside", textfont=dict(color=th.INK2, size=11),
                           hovertemplate="%{y}<br>importance %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="Importance (share of gain)")
    fig.update_xaxes(range=[0, float(imp.max()) * 1.18])
    return th.style_fig(fig, height=360, legend_top=False)


def roc_chart(metrics: dict, key: str) -> go.Figure:
    """Curva ROC del modelo en los 12 meses de prueba."""
    m = metrics["models"][key]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(color=th.AXIS, width=1.5, dash="dash"), hoverinfo="skip")
    fig.add_scatter(x=m["roc"]["fpr"], y=m["roc"]["tpr"], mode="lines", name=f"XGBoost (AUC = {m['auc']:.2f})",
                    line=dict(color=th.CATEGORICAL[0], width=2.2),
                    hovertemplate="False positive rate %{x:.2f}<br>True positive rate %{y:.2f}<extra></extra>")
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return th.style_fig(fig, height=360)


def monthly_mode(ship: pd.DataFrame) -> go.Figure:
    """Cajas por mes y modo (barras apiladas)."""
    g = ship.assign(month=ship.date.dt.to_period("M").dt.to_timestamp()).groupby(["month", "mode"], as_index=False).boxes.sum()
    fig = go.Figure()
    for mode in ("air", "sea"):
        s = g[g["mode"] == mode]
        fig.add_bar(x=s.month, y=s.boxes, name=th.MODE_LABEL[mode], marker=dict(color=th.MODE_COLOR[mode], cornerradius=3,
                    line=dict(color=th.SURFACE, width=2)), hovertemplate="%{x|%b %Y}<br>" + th.MODE_LABEL[mode] + ": %{y:,.0f} boxes<extra></extra>")
    fig.update_layout(barmode="stack", yaxis_title="Boxes shipped", bargap=0.25)
    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
    return th.style_fig(fig, height=360)
