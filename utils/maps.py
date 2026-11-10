"""
Mapa global (pydeck), mapa pequeño del mercado (plotly) y animación del avión y el barco (plotly).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from optimization import network as net
from utils import theme as th

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# En pydeck un texto suelto se lee como nombre de columna; un valor literal va envuelto en pdk.types.String
PIXELS = pdk.types.String("pixels")
FONT = pdk.types.String("Segoe UI, system-ui, sans-serif")
CHARSET_AUTO = pdk.types.String("auto")   # el TextLayer arma su alfabeto con los textos (tildes de Malmö, Travemünde...)

VIEWS = {
    "Global": dict(latitude=33.0, longitude=-34.0, zoom=2.1, pitch=28, bearing=0),
    "Colombia": dict(latitude=7.6, longitude=-74.7, zoom=5.7, pitch=30, bearing=0),
    "Sabana de Bogotá": dict(latitude=4.88, longitude=-74.08, zoom=9.2, pitch=30, bearing=0),
    "Rionegro, Antioquia": dict(latitude=6.12, longitude=-75.40, zoom=9.8, pitch=30, bearing=0),
    "Europe": None,   # depende del mercado
}
# Vista de Europa para cada mercado: (lat, lon, zoom)
MARKET_VIEW = {"NL": (52.10, 4.45, 8.6), "FR": (47.4, 3.8, 5.1), "CH": (48.0, 6.2, 4.95),
               "SE": (56.6, 11.5, 4.8), "FI": (57.4, 14.5, 4.4)}

KIND_COLOR = {"airport": th.AIR, "seaport": th.SEA, "river_port": th.EU_MODE_COLOR["barge"],
              "ferry_port": th.EU_MODE_COLOR["ferry"], "destination": th.INK}


def view_state(focus: str, market: str) -> pdk.ViewState:
    if focus == "Europe":
        lat, lon, z = MARKET_VIEW[market]
        return pdk.ViewState(latitude=lat, longitude=lon, zoom=z, pitch=30, bearing=0)
    return pdk.ViewState(**VIEWS[focus])


@st.cache_data(show_spinner=False)
def _icon(name: str) -> dict:
    """Ícono PNG incrustado como data URL (así el mapa no depende de archivos externos)."""
    b64 = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return dict(url=f"data:image/png;base64,{b64}", width=128, height=128, anchorY=118)


def _width(boxes, ref):
    """Grosor de la línea en pixeles, proporcional al volumen."""
    return float(1.5 + 5.5 * boxes / ref) if ref > 0 else 1.5


def market_nodes(data: dict, res: dict) -> pd.DataFrame:
    """Nodos que tocan el mercado: salida en Colombia, llegadas de sus lanes, transbordos y clientes."""
    lanes, chains = res["lanes"], res["chains"]
    ids = set(lanes.origin) | set(lanes.dest) | set(chains.dest_id)
    for v in chains.via:
        ids |= set(str(v).split())
    n = data["nodes"]
    return n[n.node_id.isin(ids)].copy()


def segments_table(chains: pd.DataFrame, flow: dict | None = None) -> pd.DataFrame:
    """Una fila por tramo europeo (origen, destino, modo) con su ruta y las cajas que pasan por él."""
    rows = []
    for c in chains.itertuples():
        boxes = (flow or {}).get(c.chain_id, 0.0)
        for s in json.loads(c.segments):
            rows.append(dict(origin=s["origin"], dest=s["dest"], mode=s["mode"], path=s["path"], boxes=boxes,
                             km=s["distance_km"], hours=s["time_h"], chains=c.label))
    seg = pd.DataFrame(rows)
    if seg.empty:
        return seg
    return (seg.groupby(["origin", "dest", "mode"], as_index=False)
               .agg(path=("path", "first"), boxes=("boxes", "sum"), km=("km", "first"), hours=("hours", "first")))


def build_deck(data: dict, cl: dict, res: dict, plan: dict, focus: str = "Global", closed_nodes: set | None = None) -> pdk.Deck:
    """
    Arma el mapa de un plan (antes o después de la disrupción).

    res          = salida completa de run_scenario (candidatos, lanes y cadenas del mercado)
    plan         = res["before"] o res["after"]
    closed_nodes = nodos que se pintan en rojo (cerrados por una disrupción)
    """
    nodes = data["nodes"].set_index("node_id")
    lanes = res["lanes"].set_index("lane_id")
    farms = cl["farms"]
    cand = res["candidates"].set_index("hub_id")
    open_hubs = set(plan["open_hubs"])
    routes, fh = plan["routes"], plan["farm_hub"]
    ref = max(float(routes.groupby("lane_id").boxes.sum().max()) if len(routes) else 1.0, 1.0)
    layers = []

    # ---- 1) Finca -> centro de acopio (líneas finas) ----
    if len(fh):
        g = fh.groupby(["farm_id", "hub_id"], as_index=False).boxes.sum()
        f_idx = farms.set_index("farm_id")
        g["src"] = [[f_idx.loc[f, "lon"], f_idx.loc[f, "lat"]] for f in g.farm_id]
        g["dst"] = [[cand.loc[h, "lon"], cand.loc[h, "lat"]] for h in g.hub_id]
        g["tip"] = [f"<b>{f_idx.loc[r.farm_id, 'name']}</b> → {cand.loc[r.hub_id, 'name']}<br/>{r.boxes:,.0f} boxes / week" for r in g.itertuples()]
        layers.append(pdk.Layer("LineLayer", g, get_source_position="src", get_target_position="dst",
                                get_color=th.hex_to_rgb(th.INK2, 120), get_width=1.2, pickable=True))

    # ---- 2) Centro de acopio -> aeropuerto o puerto (arco bajo, color del modo) ----
    if len(routes):
        g = routes.groupby(["hub_id", "gateway", "mode"], as_index=False).boxes.sum()
        g["src"] = [[cand.loc[h, "lon"], cand.loc[h, "lat"]] for h in g.hub_id]
        g["dst"] = [[nodes.loc[n, "lon"], nodes.loc[n, "lat"]] for n in g.gateway]
        g["color"] = [th.hex_to_rgb(th.MODE_COLOR[m], 200) for m in g["mode"]]
        g["w"] = [_width(b, ref) * 0.7 for b in g.boxes]
        g["tip"] = [f"<b>{cand.loc[r.hub_id, 'name']}</b> → {nodes.loc[r.gateway, 'name']}<br/>Reefer truck · {r.boxes:,.0f} boxes / week" for r in g.itertuples()]
        layers.append(pdk.Layer("ArcLayer", g, get_source_position="src", get_target_position="dst", get_source_color="color",
                                get_target_color="color", get_width="w", get_height=0.35, pickable=True))

    # ---- 3) Lanes intercontinentales: el avión es un arco, el barco sigue la ruta por el agua ----
    lane_flow = routes.groupby("lane_id").boxes.sum().to_dict() if len(routes) else {}
    air_rows, sea_rows, idle_rows, closed_rows = [], [], [], []
    for lid, l in lanes.iterrows():
        o, d = nodes.loc[l.origin], nodes.loc[l.dest]
        boxes = lane_flow.get(lid, 0.0)
        closed = lid in plan["closed_lanes"]
        tip = (f"<b>{'Air' if l['mode'] == 'air' else 'Sea'} lane {o.short} → {d.short}</b><br/>"
               + ("CLOSED by predicted disruption" if closed else f"{boxes:,.0f} of {l.capacity_boxes:,.0f} boxes / week")
               + f"<br/>{l.time_h / 24:.1f} days · €{l.cost_per_box:.2f} and {l.co2_kg_per_box:.1f} kg CO₂ per box")
        path = [[lo, la] for la, lo in net.SEA_WAYPOINTS[lid]] if l["mode"] == "sea" else None
        row = dict(src=[o.lon, o.lat], dst=[d.lon, d.lat], path=path, tip=tip, w=_width(boxes, ref), mode=l["mode"])
        if closed:
            closed_rows.append({**row, "color": th.hex_to_rgb(th.CRITICAL, 190), "w": 2.0})
        elif boxes > 0.5:
            (air_rows if l["mode"] == "air" else sea_rows).append({**row, "color": th.hex_to_rgb(th.MODE_COLOR[l["mode"]], 230)})
        else:
            idle_rows.append({**row, "color": th.hex_to_rgb(th.MODE_COLOR[l["mode"]], 55), "w": 1.2,
                              "tip": tip + "<br/>Available, not used in this plan"})
    for rows, alpha_layer in ((idle_rows, "idle"), (closed_rows, "closed"), (air_rows, "air"), (sea_rows, "sea")):
        air = [r for r in rows if r["mode"] == "air"]
        sea = [r for r in rows if r["mode"] == "sea"]
        if air:
            layers.append(pdk.Layer("ArcLayer", pd.DataFrame(air), get_source_position="src", get_target_position="dst",
                                    get_source_color="color", get_target_color="color", get_width="w", get_height=0.28,
                                    great_circle=True, pickable=True))
        if sea:
            layers.append(pdk.Layer("PathLayer", pd.DataFrame(sea), get_path="path", get_color="color", get_width="w",
                                    width_units=PIXELS, cap_rounded=True, joint_rounded=True, pickable=True))

    # ---- 4) Cadenas europeas: cada tramo con el color de su modo ----
    chain_flow = routes.groupby("chain_id").boxes.sum().to_dict() if len(routes) else {}
    seg = segments_table(res["chains"], chain_flow)
    if len(seg):
        eu_ref = max(float(seg.boxes.max()), 1.0)
        seg["used"] = seg.boxes > 0.5
        seg["color"] = [th.hex_to_rgb(th.EU_MODE_COLOR[m], 235 if u else 95) for m, u in zip(seg["mode"], seg.used)]
        seg["w"] = [(1.8 + 5.0 * b / eu_ref) if u else 1.3 for b, u in zip(seg.boxes, seg.used)]
        seg["src"] = [p[0] for p in seg.path]
        seg["dst"] = [p[-1] for p in seg.path]
        seg["tip"] = [f"<b>{th.EU_MODE_LABEL[r.mode]}: {th.NODE_LABEL.get(r.origin, r.origin)} → {th.NODE_LABEL.get(r.dest, r.dest)}</b><br/>"
                      f"{r.km:,.0f} km · {r.hours:.0f} h<br/>"
                      + (f"{r.boxes:,.0f} boxes / week" if r.used else "Option the model did not choose") for r in seg.itertuples()]
        for used in (False, True):
            part = seg[seg.used == used]
            ground, flights = part[part["mode"] != "flight"], part[part["mode"] == "flight"]
            if len(ground):
                layers.append(pdk.Layer("PathLayer", ground, get_path="path", get_color="color", get_width="w",
                                        width_units=PIXELS, cap_rounded=True, joint_rounded=True, pickable=True))
            if len(flights):
                layers.append(pdk.Layer("ArcLayer", flights, get_source_position="src", get_target_position="dst",
                                        get_source_color="color", get_target_color="color", get_width="w", get_height=0.5,
                                        great_circle=True, pickable=True))

    # ---- 5) Fincas, con el color de su grupo de k-medoids ----
    f = farms.copy()
    served = set(fh.farm_id) if len(fh) else set()
    box_cols = [c for c in f.columns if c.startswith("boxes_")]
    f["boxes"] = f[box_cols].sum(axis=1)
    f["color"] = [th.hex_to_rgb(th.cluster_color(int(c)), 235 if fid in served else 90) for c, fid in zip(f.cluster, f.farm_id)]
    f["r"] = 3.5 + 4.5 * np.sqrt(f.boxes / f.boxes.max())
    f["line"] = [[255, 255, 255, 255] if m else [14, 16, 20, 255] for m in f.is_medoid]
    f["lw"] = [2.5 if m else 1.0 for m in f.is_medoid]
    f["tip"] = [f"<b>{r.name}</b> · {r.municipality}<br/>{r.boxes:,.0f} boxes / week · cluster {int(r.cluster) + 1}"
                + ("<br/><b>Medoid</b> of its cluster (k-medoids)" if r.is_medoid else "")
                + ("" if r.farm_id in served else "<br/>Not served in this plan") for r in f.itertuples()]
    layers.append(pdk.Layer("ScatterplotLayer", f, get_position=["lon", "lat"], get_fill_color="color", get_radius="r",
                            radius_units=PIXELS, stroked=True, get_line_color="line", get_line_width="lw",
                            line_width_units=PIXELS, pickable=True))

    # ---- 6) Puertos, aeropuertos, transbordos y clientes del mercado ----
    n = market_nodes(data, res)
    closed_nodes = closed_nodes or set()
    n["closed"] = n.node_id.isin(closed_nodes)
    n["color"] = [th.hex_to_rgb(th.CRITICAL if c else KIND_COLOR[k], 255) for k, c in zip(n.kind, n.closed)]
    n["r"] = [7 if t == "destination" else 6 if t != "transfer" else 5 for t in n.type]
    n["tip"] = [f"<b>{r.name}</b><br/>{r.country}" + ("<br/><b>CLOSED – disruption predicted</b>" if r.closed else "") for r in n.itertuples()]
    n["label"] = [s if t != "transfer" else "" for s, t in zip(n.short, n.type)]   # transbordos: solo al pasar el mouse
    for radius, alpha in ((26, 60), (18, 110), (11, 190)):      # halo rojo en los nodos cerrados
        if n.closed.any():
            layers.append(pdk.Layer("ScatterplotLayer", n[n.closed], get_position=["lon", "lat"], get_radius=radius,
                                    radius_units=PIXELS, filled=False, stroked=True,
                                    get_line_color=th.hex_to_rgb(th.CRITICAL, alpha), get_line_width=2, line_width_units=PIXELS))
    layers.append(pdk.Layer("ScatterplotLayer", n, get_position=["lon", "lat"], get_fill_color="color", get_radius="r",
                            radius_units=PIXELS, stroked=True, get_line_color=[14, 16, 20, 255], get_line_width=1.5,
                            line_width_units=PIXELS, pickable=True))
    layers.append(pdk.Layer("TextLayer", n, get_position=["lon", "lat"], get_text="label", get_size=12,
                            get_color=th.hex_to_rgb(th.INK, 235), get_pixel_offset=[0, -16], font_weight=600,
                            font_family=FONT, outline_width=3, outline_color=[14, 16, 20, 255],
                            font_settings={"sdf": True}, character_set=CHARSET_AUTO))

    # ---- 7) Centros de acopio: ícono de almacén (blanco = abierto, gris = candidato sin abrir) ----
    c = res["candidates"].copy()
    c["open"] = c.hub_id.isin(open_hubs)
    hub_in = fh.groupby("hub_id").boxes.sum().to_dict() if len(fh) else {}
    c["icon_data"] = [_icon("warehouse_open.png") if o else _icon("warehouse_idle.png") for o in c.open]
    c["size"] = [34 if o else 20 for o in c.open]
    c["tip"] = [f"<b>{r.name}</b><br/>{'OPEN' if r.open else 'Candidate (not opened)'} · {r.source}<br/>"
                + (f"{hub_in.get(r.hub_id, 0):,.0f} of {r.capacity:,.0f} boxes / week · " if r.open else f"Capacity {r.capacity:,.0f} boxes / week · ")
                + f"fixed cost €{r.fixed_cost_week:,.0f} / week" for r in c.itertuples()]
    layers.append(pdk.Layer("IconLayer", c.sort_values("open"), get_position=["lon", "lat"], get_icon="icon_data", get_size="size",
                            size_units=PIXELS, pickable=True))

    return pdk.Deck(
        layers=layers, initial_view_state=view_state(focus, res["market"]), map_provider="carto", map_style="dark",
        tooltip={"html": "{tip}", "style": {"backgroundColor": "#22262f", "color": th.INK, "fontSize": "12px",
                                            "borderRadius": "8px", "border": "1px solid #383835", "padding": "8px 10px"}},
    )


# ----------------------------------------------------------------------------
# Mapa pequeño del mercado (página 1): todas las cadenas europeas posibles
# ----------------------------------------------------------------------------
def market_preview(data: dict, market: str) -> go.Figure:
    nodes = data["nodes"].set_index("node_id")
    chains = data["chains"][data["chains"].market == market]
    lanes = data["lanes"][data["lanes"].dest.isin(chains.arrival)]
    seg = segments_table(chains)
    fig = go.Figure()
    # Llegada de los lanes intercontinentales: una flecha corta desde el océano
    for l in lanes.drop_duplicates("dest").itertuples():
        pts = net.SEA_WAYPOINTS[l.lane_id][-5:] if l.mode == "sea" else None
        if pts is None:
            o, d = nodes.loc[l.origin], nodes.loc[l.dest]
            pts = net.great_circle_points(o.lat, o.lon, d.lat, d.lon, n=60)[-8:]
        fig.add_trace(go.Scattergeo(lat=[p[0] for p in pts], lon=[p[1] for p in pts], mode="lines", hoverinfo="skip",
                                    line=dict(color=th.MODE_COLOR[l.mode], width=2.2, dash="dot"), showlegend=False))
    # Tramos europeos, un color por modo (la leyenda muestra cada modo una sola vez)
    shown = set()
    for s in seg.itertuples():
        lon = [p[0] for p in s.path]
        lat = [p[1] for p in s.path]
        if s.mode == "flight":
            gc = net.great_circle_points(lat[0], lon[0], lat[-1], lon[-1], n=30)
            lat, lon = [p[0] for p in gc], [p[1] for p in gc]
        fig.add_trace(go.Scattergeo(
            lat=lat, lon=lon, mode="lines", line=dict(color=th.EU_MODE_COLOR[s.mode], width=2.4),
            name=th.EU_MODE_LABEL[s.mode], legendgroup=s.mode, showlegend=s.mode not in shown,
            hovertemplate=f"{th.EU_MODE_LABEL[s.mode]}: {th.NODE_LABEL.get(s.origin, s.origin)} → "
                          f"{th.NODE_LABEL.get(s.dest, s.dest)}<br>{s.km:,.0f} km · {s.hours:.0f} h<extra></extra>"))
        shown.add(s.mode)
    ids = set(chains.arrival) | set(chains.dest_id) | {v for x in chains.via for v in str(x).split()}
    n = data["nodes"][data["nodes"].node_id.isin(ids)]
    dest = n.type == "destination"
    fig.add_trace(go.Scattergeo(lat=n.lat, lon=n.lon, mode="markers+text", text=[s if d else "" for s, d in zip(n.short, dest)],
                                textposition="top center", textfont=dict(color=th.INK, size=11),
                                marker=dict(size=[10 if d else 7 for d in dest], color=[KIND_COLOR[k] for k in n.kind],
                                            line=dict(color=th.PAGE, width=1.2), symbol=["circle" if d else "diamond" for d in dest]),
                                hovertext=n.name, hoverinfo="text", showlegend=False))
    lat0, lon0, z = MARKET_VIEW[market]
    span = {"NL": (2.2, 3.2), "FR": (10.5, 13.0), "CH": (10.5, 13.0), "SE": (10.5, 18.0), "FI": (12.0, 24.0)}[market]
    fig.update_layout(
        height=360, paper_bgcolor=th.SURFACE, margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family='system-ui, "Segoe UI", sans-serif', color=th.INK2),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(23,26,33,0.8)",
                    font=dict(color=th.INK2, size=11)),
        hoverlabel=dict(bgcolor="#22262f", bordercolor=th.AXIS, font=dict(color=th.INK)),
        geo=dict(projection_type="mercator", bgcolor=th.SURFACE, showland=True, landcolor="#262b35", showocean=True,
                 oceancolor="#141820", showlakes=False, showcountries=True, countrycolor="#3d434e", coastlinecolor="#4b525e",
                 showframe=False, resolution=50,
                 lataxis=dict(range=[lat0 - span[0] / 2, lat0 + span[0] / 2]),
                 lonaxis=dict(range=[lon0 - span[1] / 2, lon0 + span[1] / 2])),
    )
    return fig


# ----------------------------------------------------------------------------
# Animación: el avión cruza en un día; el barco tarda dos semanas y media
# ----------------------------------------------------------------------------
def animation_figure(data: dict, res: dict, plan: dict, closed_nodes: set) -> go.Figure:
    nodes = data["nodes"].set_index("node_id")
    lanes = res["lanes"].set_index("lane_id")
    flow = plan["routes"].groupby("lane_id").boxes.sum() if len(plan["routes"]) else pd.Series(dtype=float)

    def pick(mode):
        ids = [l for l in lanes.index if lanes.loc[l, "mode"] == mode]
        used = [l for l in ids if flow.get(l, 0) > 0]
        return max(used, key=lambda l: flow[l]) if used else ids[0]

    air_id, sea_id = pick("air"), pick("sea")
    air, sea = lanes.loc[air_id], lanes.loc[sea_id]
    a0, a1 = nodes.loc[air.origin], nodes.loc[air.dest]
    air_days, sea_days = air.time_h / 24.0, sea.time_h / 24.0
    step = 0.25                                                   # cada cuadro es un cuarto de día
    n_frames = int(np.ceil(sea_days / step)) + 3
    air_path = net.great_circle_points(a0.lat, a0.lon, a1.lat, a1.lon, n=60)
    sea_path = net.interpolate_path(net.SEA_WAYPOINTS[sea_id], n=240)
    cycle = 2.0                                                   # sale un vuelo cada dos días
    sea_arrival = nodes.loc[sea.dest, "short"]

    def at(path, frac):
        return path[min(int(round(frac * (len(path) - 1))), len(path) - 1)]

    ids = set(lanes.origin) | set(lanes.dest)
    pts = data["nodes"][data["nodes"].node_id.isin(ids)]
    closed = pts[pts.node_id.isin(closed_nodes)]
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=[p[0] for p in air_path], lon=[p[1] for p in air_path], mode="lines",
                                line=dict(color=th.AIR, width=2), name=f"Air lane {a0.short} → {a1.short} ({air_days:.1f} days)", hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(lat=[p[0] for p in sea_path], lon=[p[1] for p in sea_path], mode="lines",
                                line=dict(color=th.SEA, width=2), name=f"Sea lane {nodes.loc[sea.origin, 'short']} → {sea_arrival} ({sea_days:.1f} days)",
                                hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(lat=pts.lat, lon=pts.lon, mode="markers+text", text=pts.node_id, textposition="top center",
                                textfont=dict(color=th.INK, size=11),
                                marker=dict(size=8, color=[th.CRITICAL if i in closed_nodes else th.INK for i in pts.node_id],
                                            line=dict(color=th.PAGE, width=1)),
                                hovertext=pts.name, hoverinfo="text", showlegend=False))
    # Trazas que se mueven: 3 = pulso de nodos cerrados, 4 = avión, 5 = barco
    fig.add_trace(go.Scattergeo(lat=closed.lat, lon=closed.lon, mode="markers", name="Closed node",
                                marker=dict(size=16, color="rgba(208,59,59,0.25)", line=dict(color=th.CRITICAL, width=2)),
                                hoverinfo="skip", showlegend=len(closed) > 0))
    p0, s0 = air_path[0], sea_path[0]
    fig.add_trace(go.Scattergeo(lat=[p0[0]], lon=[p0[1]], mode="text", text=["✈"], textfont=dict(size=26, color="#ffb08f"),
                                hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergeo(lat=[s0[0]], lon=[s0[1]], mode="text", text=["🚢"], textfont=dict(size=22),
                                hoverinfo="skip", showlegend=False))

    frames, steps = [], []
    for i in range(n_frames):
        t = i * step
        pa = at(air_path, min(1.0, (t % cycle) / air_days))
        ps = at(sea_path, min(1.0, t / sea_days))
        flights = int(t // cycle) + (1 if (t % cycle) >= air_days else 0)
        pulse = 14 + 12 * (i % 4) / 3.0                           # el anillo rojo crece y vuelve a empezar
        title = (f"Day {t:4.1f}  ·  flights landed: {flights}  ·  ship: "
                 + (f"arrived in {sea_arrival}" if t >= sea_days else f"{t / sea_days:.0%} of the crossing"))
        frames.append(go.Frame(name=str(i), traces=[3, 4, 5], data=[
            go.Scattergeo(lat=closed.lat, lon=closed.lon, marker=dict(size=pulse)),
            go.Scattergeo(lat=[pa[0]], lon=[pa[1]]),
            go.Scattergeo(lat=[ps[0]], lon=[ps[1]]),
        ], layout=go.Layout(title_text=title)))
        steps.append(dict(method="animate", label=f"{t:.0f}" if t == int(t) else "",
                          args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]))
    fig.frames = frames

    fig.update_layout(
        title=dict(text="Day  0.0  ·  press Play", font=dict(size=14, color=th.INK2), x=0.01, y=0.97),
        height=500, paper_bgcolor=th.SURFACE, margin=dict(l=10, r=10, t=36, b=10),
        font=dict(family='system-ui, "Segoe UI", sans-serif', color=th.INK2),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(23,26,33,0.75)", font=dict(color=th.INK2)),
        geo=dict(projection_type="natural earth", bgcolor=th.SURFACE, showland=True, landcolor="#2a303b", showocean=True,
                 oceancolor=th.SURFACE, showlakes=False, showcountries=True, countrycolor="#3d434e", coastlinecolor="#4b525e",
                 showframe=False, lataxis=dict(range=[-6, 64]), lonaxis=dict(range=[-98, 30]), domain=dict(x=[0, 1], y=[0.14, 1])),
        updatemenus=[dict(type="buttons", direction="left", x=0.01, y=0.0, xanchor="left", yanchor="bottom", showactive=False,
                          bgcolor="#22262f", bordercolor=th.AXIS, font=dict(color=th.INK),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=110, redraw=True), transition=dict(duration=0), fromcurrent=True)]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.20, len=0.78, y=0.0, yanchor="bottom", pad=dict(t=0, b=6), steps=steps,
                      currentvalue=dict(visible=False), font=dict(color=th.MUTED, size=10), bgcolor="#22262f",
                      bordercolor=th.AXIS, tickcolor=th.AXIS, activebgcolor=th.BRAND)],
    )
    return fig
