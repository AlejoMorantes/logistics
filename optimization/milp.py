"""
Modelo lineal entero mixto (MILP) de BloomChain, en PuLP con el solver CBC.

Red de cinco niveles (un transbordo con localización, como los casos de clase):

    finca -> centro de acopio -> aeropuerto o puerto -> llegada a Europa -> cadena europea -> cliente
          x                   \\__________________ ruta (h, l, e): w __________________/

La cadena europea e puede ser de un solo modo (camión) o multimodal (barcaza + camión,
feeder + tren, camión + ferry + camión, vuelo de conexión + camión). Cada cadena termina
en un cliente d(e) del mercado elegido (Países Bajos, Francia, Suiza, Suecia o Finlandia).

Decisiones
    y_h      binaria   abrir el centro de acopio h
    m_l      binaria   activar el lane l (avión BOG/MDE -> AMS/CDG o barco CTG/SMR -> RTM/LEH/FOS/GOA)
    x_fhp    continua  cajas de la flor p de la finca f al centro h
    w_hlep   continua  cajas de la flor p del centro h, por el lane l y la cadena europea e
    v_lp     continua  cajas de la flor p que viajan en el lane l
    u_ldp    continua  cajas de la flor p que salen de la llegada del lane l hacia el cliente d
    z        0/1       disponibilidad de un lane o de un arco acopio->puerto. No la decide el
                       optimizador: la fija XGBoost (z = 0 si P(disrupción) >= umbral).

La vida en florero depende del camino completo. Por eso el flujo después del centro de
acopio se modela por ruta (h, l, e): así se puede descartar de forma exacta cada ruta que
llegaría con menos vida que la mínima.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from optimization import network as net

UNMET_PENALTY = 300.0   # EUR por caja no entregada (mucho mayor que cualquier ruta)

# Qué cierra cada tipo de disrupción. Los lanes se escogen por regla (origen o modo),
# así sirven para cualquier mercado: la niebla cierra todos los vuelos que salen de Bogotá.
CLOSURES = {
    "fog_bog": dict(lane_rule=("origin", "BOG"), hub_gateway=set(), nodes=["BOG"]),
    "road_strike": dict(lane_rule=None, hub_gateway={("Sabana de Bogotá", "CTG"), ("Sabana de Bogotá", "SMR")}, nodes=[]),
    "port_congestion": dict(lane_rule=("origin", "CTG"), hub_gateway=set(), nodes=["CTG"]),
    "atlantic_storm": dict(lane_rule=("mode", "sea"), hub_gateway=set(), nodes=["CTG", "SMR"]),
}


@dataclass(frozen=True)
class Scenario:
    """Parámetros que el usuario mueve en la página 1."""
    n_min: int = 2                      # número mínimo de centros de acopio
    n_max: int = 5                      # número máximo de centros de acopio
    max_farm_hub_km: float = 30.0       # distancia máxima finca-acopio (km por carretera)
    min_hub_gap_km: float = 5.0         # distancia mínima entre dos centros abiertos (km en línea recta)
    max_hub_capacity: int = 5000        # capacidad máxima por centro (cajas por semana)
    carbon_price: float = 80.0          # EUR por tonelada de CO2
    min_vase_life: float = 9.0          # días mínimos de vida en florero al llegar
    quality_pct: float = 6.0            # % del valor de la caja que se pierde por cada día de florero perdido
    flowers: tuple = ("rose", "carnation", "chrysanthemum")
    candidate_source: str = "both"      # "sites", "medoids" o "both"
    market: str = "NL"                  # mercado de destino: NL, FR, CH, SE o FI


@dataclass
class Prepared:
    """Todo lo que el modelo necesita ya calculado (arcos, rutas, costos)."""
    sc: Scenario
    hubs: pd.DataFrame
    farm_hub: pd.DataFrame      # arcos finca -> acopio factibles
    routes: pd.DataFrame        # rutas acopio -> lane -> cadena europea -> cliente (una fila por ruta y flor)
    lanes: pd.DataFrame         # solo los lanes que llegan a alguna cadena del mercado
    chains: pd.DataFrame        # cadenas europeas del mercado
    flowers: pd.DataFrame
    supply: dict                # (finca, flor) -> cajas
    demand: dict                # (destino, flor) -> cajas
    conflicts: list = field(default_factory=list)   # pares de centros demasiado cerca
    farms_region: pd.DataFrame | None = None        # región de cada finca


# ----------------------------------------------------------------------------
# Preparación de la red
# ----------------------------------------------------------------------------
def candidate_hubs(hubs: pd.DataFrame, medoid_sites: pd.DataFrame, source: str) -> pd.DataFrame:
    """Conjunto de sitios candidatos: los 8 predefinidos, los medoides de k-medoids o ambos."""
    sites = hubs.assign(source="pre-screened site")
    if source == "sites" or medoid_sites is None or medoid_sites.empty:
        return sites.reset_index(drop=True)
    if source == "medoids":
        return medoid_sites.reset_index(drop=True)
    return pd.concat([sites, medoid_sites], ignore_index=True)


def prepare(data: dict, sc: Scenario, medoid_sites: pd.DataFrame | None = None, all_medoids: pd.DataFrame | None = None) -> Prepared:
    farms, nodes, lanes, flowers = data["farms"], data["nodes"], data["lanes"], data["flowers"]
    flowers = flowers[flowers.flower.isin(sc.flowers)].reset_index(drop=True)
    if sc.candidate_source == "medoids" and all_medoids is not None:
        medoid_sites = all_medoids
    hubs = candidate_hubs(data["hubs"], medoid_sites, sc.candidate_source).copy()
    hubs["capacity"] = np.minimum(hubs.capacity_boxes, sc.max_hub_capacity)
    node = nodes.set_index("node_id")

    # ---- Arcos finca -> acopio (misma región y dentro de la distancia máxima) ----
    rows = []
    for _, h in hubs.iterrows():
        f_reg = farms[farms.region == h.region]
        straight = net.haversine_km(f_reg.lat.to_numpy(), f_reg.lon.to_numpy(), h.lat, h.lon)
        for (_, f), km in zip(f_reg.iterrows(), straight):
            arc = net.road_arc(float(km), "farm_hub")
            if arc["distance_km"] <= sc.max_farm_hub_km:
                rows.append(dict(farm_id=f.farm_id, hub_id=h.hub_id, **arc))
    farm_hub = pd.DataFrame(rows, columns=["farm_id", "hub_id", "distance_km", "cost_per_box", "time_h", "co2_kg_per_box", "vase_loss_per_day"])
    farm_hub["vase_loss"] = net.vase_loss_days(farm_hub.time_h, farm_hub.vase_loss_per_day)

    # ---- Mercado: cadenas europeas y lanes que llegan a alguna de ellas ----
    chains = data["chains"][data["chains"].market == sc.market].reset_index(drop=True)
    lanes = lanes[lanes.dest.isin(chains.arrival)].reset_index(drop=True)

    # ---- Rutas acopio -> (aeropuerto o puerto) -> lane -> llegada -> cadena europea -> cliente ----
    worst_first_mile = farm_hub.groupby("hub_id").vase_loss.max().to_dict()   # peor caso de la primera milla
    rows = []
    for _, h in hubs.iterrows():
        for _, l in lanes.iterrows():
            g = node.loc[l.origin]
            leg1 = net.road_arc(float(net.haversine_km(h.lat, h.lon, g.lat, g.lon)), "hub_gateway")
            for e in chains[chains.arrival == l.dest].itertuples():
                loss = (net.HUB_DWELL_VASE_LOSS
                        + net.vase_loss_days(leg1["time_h"], leg1["vase_loss_per_day"])
                        + net.vase_loss_days(l.time_h, l.vase_loss_per_day)
                        + e.vase_loss)
                rows.append(dict(
                    hub_id=h.hub_id, hub_region=h.region, lane_id=l.lane_id, mode=l["mode"], gateway=l.origin,
                    arrival=l.dest, chain_id=e.chain_id, chain_label=e.label, eu_modes=e.modes, dest_id=e.dest_id,
                    cost_per_box=net.HUB_HANDLING_EUR_BOX + leg1["cost_per_box"] + l.cost_per_box + e.cost_per_box,
                    co2_kg_per_box=leg1["co2_kg_per_box"] + l.co2_kg_per_box + e.co2_kg_per_box,
                    time_h=net.HUB_DWELL_H + leg1["time_h"] + l.time_h + e.time_h,
                    vase_loss=loss, first_mile_loss=worst_first_mile.get(h.hub_id, 0.0),
                ))
    base = pd.DataFrame(rows)
    # Una fila por ruta y flor, con la vida en florero al llegar
    routes = base.merge(flowers[["flower", "vase_life_days", "box_value_eur"]], how="cross")
    routes["vase_at_arrival"] = routes.vase_life_days - routes.vase_loss - routes.first_mile_loss
    routes["quality_eur_day"] = sc.quality_pct / 100.0 * routes.box_value_eur

    supply = {(f.farm_id, fl): int(getattr(f, f"boxes_{fl}")) for f in farms.itertuples() for fl in flowers.flower}
    dem = data["demand"]
    dem = dem[(dem.market == sc.market) & dem.flower.isin(flowers.flower)]
    demand = {(r.dest_id, r.flower): int(r.boxes) for r in dem.itertuples()}

    # Pares de candidatos que quedan más cerca que la distancia mínima entre centros
    dmat = net.distance_matrix_km(hubs.lat, hubs.lon)
    ids = hubs.hub_id.tolist()
    conflicts = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)) if dmat[i, j] < sc.min_hub_gap_km]

    return Prepared(sc=sc, hubs=hubs, farm_hub=farm_hub, routes=routes, lanes=lanes, chains=chains, flowers=flowers,
                    supply=supply, demand=demand, conflicts=conflicts, farms_region=farms[["farm_id", "region"]])


def closed_lanes_for(key: str, lanes: pd.DataFrame) -> set:
    """Lanes que cierra una disrupción (por origen o por modo)."""
    rule = CLOSURES[key]["lane_rule"]
    if rule is None:
        return set()
    col, value = rule
    return set(lanes.loc[lanes[col] == value, "lane_id"])


def closures_for(active: set | frozenset, lanes: pd.DataFrame) -> tuple[set, set]:
    """Une los cierres de todas las disrupciones activas."""
    closed, hub_gw = set(), set()
    for key in active:
        closed |= closed_lanes_for(key, lanes)
        hub_gw |= CLOSURES[key]["hub_gateway"]
    return closed, hub_gw


def _solver():
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=20, gapRel=1e-5)


# ----------------------------------------------------------------------------
# El modelo
# ----------------------------------------------------------------------------
def solve(prep: Prepared, active: frozenset = frozenset(), fixed_hubs: set | None = None, air_only: bool = False) -> dict:
    """
    Resuelve el MILP.

    active      disrupciones activas (cierran lanes o arcos: z = 0)
    fixed_hubs  si se da, esos centros quedan abiertos y los demás cerrados (para evaluar ADD y DROP)
    air_only    escenario de referencia "todo en avión"
    """
    sc = prep.sc
    closed_lanes, closed_hub_gw = closures_for(active, prep.lanes)
    if air_only:
        closed_lanes = closed_lanes | set(prep.lanes.loc[prep.lanes["mode"] == "sea", "lane_id"])

    hubs = prep.hubs.set_index("hub_id")
    H = hubs.index.tolist()
    lanes = prep.lanes.set_index("lane_id")
    L = [l for l in lanes.index if l not in closed_lanes]                # z_l = 1
    P = prep.flowers.flower.tolist()
    D = sorted({d for d, _ in prep.demand})
    pen = {r.flower: sc.quality_pct / 100.0 * r.box_value_eur for r in prep.flowers.itertuples()}
    co2_eur = sc.carbon_price / 1000.0                                    # EUR por kg de CO2

    # Rutas disponibles: lane abierto, arco acopio->puerto abierto y vida en florero suficiente
    r = prep.routes
    ok = (r.lane_id.isin(L) & (r.vase_at_arrival >= sc.min_vase_life - 1e-9)
          & ~pd.Series(list(zip(r.hub_region, r.gateway)), index=r.index).isin(closed_hub_gw))
    R = r[ok]
    R = R[[(d, p) in prep.demand and prep.demand[(d, p)] > 0 for d, p in zip(R.dest_id, R.flower)]]
    FH = prep.farm_hub

    mdl = pulp.LpProblem("BloomChain", pulp.LpMinimize)

    # ---- Variables ----
    y = {h: pulp.LpVariable(f"y_{h}", cat="Binary") for h in H}
    m = {l: pulp.LpVariable(f"m_{l}", cat="Binary") for l in L}
    x = {(a.farm_id, a.hub_id, p): pulp.LpVariable(f"x_{a.farm_id}_{a.hub_id}_{p}", lowBound=0)
         for a in FH.itertuples() for p in P if prep.supply.get((a.farm_id, p), 0) > 0}
    w = {(a.hub_id, a.lane_id, a.chain_id, a.flower): pulp.LpVariable(f"w_{a.hub_id}_{a.lane_id}_{a.chain_id}_{a.flower}", lowBound=0)
         for a in R.itertuples()}
    v = {(l, p): pulp.LpVariable(f"v_{l}_{p}", lowBound=0) for l in L for p in P}
    u = {(l, d, p): pulp.LpVariable(f"u_{l}_{d}_{p}", lowBound=0) for l in L for d in D for p in P}
    s = {k: pulp.LpVariable(f"s_{k[0]}_{k[1]}", lowBound=0) for k in prep.demand}     # demanda no atendida

    fh = {(a.farm_id, a.hub_id): a for a in FH.itertuples()}
    rt = {(a.hub_id, a.lane_id, a.chain_id, a.flower): a for a in R.itertuples()}
    dest_of = {k: a.dest_id for k, a in rt.items()}                      # cliente d(e) al final de la cadena

    # ---- Función objetivo ----
    c_fixed = pulp.lpSum(hubs.loc[h, "fixed_cost_week"] * y[h] for h in H) + pulp.lpSum(lanes.loc[l, "fixed_cost_week"] * m[l] for l in L)
    c_transport = (pulp.lpSum(fh[f, h].cost_per_box * var for (f, h, p), var in x.items())
                   + pulp.lpSum(rt[k].cost_per_box * var for k, var in w.items()))
    kg_co2 = (pulp.lpSum(fh[f, h].co2_kg_per_box * var for (f, h, p), var in x.items())
              + pulp.lpSum(rt[k].co2_kg_per_box * var for k, var in w.items()))
    c_quality = (pulp.lpSum(pen[p] * fh[f, h].vase_loss * var for (f, h, p), var in x.items())
                 + pulp.lpSum(pen[k[3]] * rt[k].vase_loss * var for k, var in w.items()))
    c_unmet = UNMET_PENALTY * pulp.lpSum(s.values())
    mdl += c_fixed + c_transport + co2_eur * kg_co2 + c_quality + c_unmet

    # ---- Restricciones ----
    # Oferta de cada finca
    for (f, p), cap in prep.supply.items():
        terms = [x[f, h, p] for h in H if (f, h, p) in x]
        if terms:
            mdl += pulp.lpSum(terms) <= cap, f"supply_{f}_{p}"
    # Balance en cada centro de acopio: lo que entra = lo que sale
    for h in H:
        for p in P:
            inflow = [var for (f, hh, pp), var in x.items() if hh == h and pp == p]
            outflow = [var for (hh, l, e, pp), var in w.items() if hh == h and pp == p]
            if inflow or outflow:
                mdl += pulp.lpSum(inflow) == pulp.lpSum(outflow), f"balance_hub_{h}_{p}"
    # Capacidad del centro, solo si está abierto: x <= Cap * y
    for h in H:
        mdl += pulp.lpSum(var for (f, hh, p), var in x.items() if hh == h) <= hubs.loc[h, "capacity"] * y[h], f"cap_hub_{h}"
    # Balance en cada aeropuerto o puerto: lo que llega de los centros = lo que sale en el lane
    for l in L:
        for p in P:
            mdl += pulp.lpSum(var for (h, ll, e, pp), var in w.items() if ll == l and pp == p) == v[l, p], f"balance_gateway_{l}_{p}"
            # Balance en la llegada a Europa: lo que trae el lane = lo que sale por las cadenas europeas
            mdl += v[l, p] == pulp.lpSum(u[l, d, p] for d in D), f"balance_arrival_{l}_{p}"
            for d in D:
                mdl += u[l, d, p] == pulp.lpSum(var for k, var in w.items() if k[1] == l and k[3] == p and dest_of[k] == d), f"link_{l}_{d}_{p}"
    # Flujo solo por lanes activos: v <= Cap * m
    for l in L:
        mdl += pulp.lpSum(v[l, p] for p in P) <= lanes.loc[l, "capacity_boxes"] * m[l], f"cap_lane_{l}"
    # Demanda de cada destino (s = cajas que no se alcanzan a entregar)
    for (d, p), dem in prep.demand.items():
        mdl += pulp.lpSum(u[l, d, p] for l in L) + s[d, p] == dem, f"demand_{d}_{p}"
    # Centros abiertos
    if fixed_hubs is None:
        n_max = max(1, min(sc.n_max, len(H)))
        mdl += pulp.lpSum(y.values()) <= n_max, "max_hubs"
        mdl += pulp.lpSum(y.values()) >= min(sc.n_min, n_max), "min_hubs"
        for a, b in prep.conflicts:                      # distancia mínima entre centros
            mdl += y[a] + y[b] <= 1, f"gap_{a}_{b}"
    else:
        for h in H:
            mdl += y[h] == (1 if h in fixed_hubs else 0), f"fix_{h}"

    mdl.solve(_solver())
    status = pulp.LpStatus[mdl.status]

    # ---- Lectura de la solución ----
    val = lambda var: float(var.value() or 0.0)
    x_rows = [dict(farm_id=f, hub_id=h, flower=p, boxes=val(var), km=fh[f, h].distance_km,
                   cost=fh[f, h].cost_per_box, co2=fh[f, h].co2_kg_per_box, vase_loss=fh[f, h].vase_loss, time_h=fh[f, h].time_h)
              for (f, h, p), var in x.items() if val(var) > 0.5]
    xdf = pd.DataFrame(x_rows, columns=["farm_id", "hub_id", "flower", "boxes", "km", "cost", "co2", "vase_loss", "time_h"])
    # Primera milla promedio de cada centro (para reportar la vida en florero real, no la del peor caso)
    if len(xdf):
        fm = xdf.assign(l=xdf.vase_loss * xdf.boxes, t=xdf.time_h * xdf.boxes).groupby("hub_id")[["l", "t", "boxes"]].sum()
        fm_loss, fm_time = (fm.l / fm.boxes).to_dict(), (fm.t / fm.boxes).to_dict()
    else:
        fm_loss, fm_time = {}, {}
    w_rows = []
    for k, var in w.items():
        b = val(var)
        if b > 0.5:
            a = rt[k]
            w_rows.append(dict(
                hub_id=a.hub_id, gateway=a.gateway, lane_id=a.lane_id, mode=a.mode, arrival=a.arrival, chain_id=a.chain_id,
                chain_label=a.chain_label, eu_modes=a.eu_modes, dest_id=a.dest_id,
                flower=a.flower, boxes=b, cost_per_box=a.cost_per_box, co2_kg_per_box=a.co2_kg_per_box,
                transit_h=a.time_h + fm_time.get(a.hub_id, 0.0),
                vase_at_arrival=a.vase_life_days - a.vase_loss - fm_loss.get(a.hub_id, 0.0),
                vase_loss=a.vase_loss, quality_eur_day=pen[a.flower]))
    wdf = pd.DataFrame(w_rows, columns=["hub_id", "gateway", "lane_id", "mode", "arrival", "chain_id", "chain_label", "eu_modes", "dest_id", "flower", "boxes",
                                        "cost_per_box", "co2_kg_per_box", "transit_h", "vase_at_arrival", "vase_loss", "quality_eur_day"])

    open_hubs = [h for h in H if val(y[h]) > 0.5]
    active_lanes = [l for l in L if val(m[l]) > 0.5]
    delivered = float(wdf.boxes.sum())
    co2_total = float((xdf.co2 * xdf.boxes).sum() + (wdf.co2_kg_per_box * wdf.boxes).sum())
    cost = dict(
        hub_fixed=float(sum(hubs.loc[h, "fixed_cost_week"] for h in open_hubs)),
        lane_fixed=float(sum(lanes.loc[l, "fixed_cost_week"] for l in active_lanes)),
        transport=float((xdf.cost * xdf.boxes).sum() + (wdf.cost_per_box * wdf.boxes).sum()),
        carbon=co2_eur * co2_total,
        quality=float((xdf.vase_loss * xdf.boxes * xdf.flower.map(pen)).sum() + (wdf.vase_loss * wdf.boxes * wdf.quality_eur_day).sum()),
    )
    cost["total"] = sum(cost.values())
    unmet = float(sum(val(var) for var in s.values()))
    by_mode = wdf.groupby("mode").boxes.sum().to_dict()
    # Cajas cuyo tramo europeo no es solo camión (barcaza, tren, feeder, ferry o vuelo de conexión)
    multimodal = float(wdf.loc[wdf.eu_modes != "truck", "boxes"].sum())

    return dict(
        status=status, cost=cost, co2_kg=co2_total, delivered=delivered, unmet=unmet,
        demand_total=float(sum(prep.demand.values())),
        boxes_air=float(by_mode.get("air", 0.0)), boxes_sea=float(by_mode.get("sea", 0.0)),
        sea_share=float(by_mode.get("sea", 0.0) / delivered) if delivered else 0.0,
        multimodal_share=multimodal / delivered if delivered else 0.0,
        avg_vase_life=float((wdf.vase_at_arrival * wdf.boxes).sum() / delivered) if delivered else 0.0,
        avg_transit_h=float((wdf.transit_h * wdf.boxes).sum() / delivered) if delivered else 0.0,
        open_hubs=open_hubs, active_lanes=active_lanes, closed_lanes=sorted(closed_lanes),
        closed_hub_gateway=sorted(closed_hub_gw), farm_hub=xdf, routes=wdf,
        n_vars=len(mdl.variables()), n_cons=len(mdl.constraints),
    )


# ----------------------------------------------------------------------------
# Heurísticas ADD y DROP para escoger centros (versión de clase, con matriz de costos)
# ----------------------------------------------------------------------------
def heuristic_cost_matrix(prep: Prepared, active: frozenset = frozenset()):
    """
    Tabla G[h][(f, p)] = costo por caja de mandar la flor p de la finca f por el centro h hasta Europa
    (como la matriz h_j*d_ij de clase, pero en euros por caja: primera milla + la mejor salida del centro).

    Devuelve también cost_of(S): con los centros S abiertos, el pedido de cada flor se llena con las
    combinaciones (finca, centro) más baratas, respetando la oferta de cada finca y la capacidad de
    cada centro (un problema de transporte resuelto a lo codicioso). Suma costos fijos y una multa
    por las cajas que no alcanzan. Es una versión simplificada: no mira la capacidad de los lanes.
    """
    sc = prep.sc
    closed_lanes, closed_hub_gw = closures_for(active, prep.lanes)
    co2_eur = sc.carbon_price / 1000.0
    r = prep.routes
    ok = (~r.lane_id.isin(closed_lanes) & (r.vase_at_arrival >= sc.min_vase_life - 1e-9)
          & ~pd.Series(list(zip(r.hub_region, r.gateway)), index=r.index).isin(closed_hub_gw))
    r = r[ok].copy()
    r["gen_cost"] = r.cost_per_box + co2_eur * r.co2_kg_per_box + r.quality_eur_day * r.vase_loss
    r["dem"] = [prep.demand.get((d, p), 0) for d, p in zip(r.dest_id, r.flower)]
    # Costo de salida del centro h para la flor p: para cada cliente, la ruta más barata; luego promedio según el pedido
    best = r.groupby(["hub_id", "flower", "dest_id"]).agg(c=("gen_cost", "min"), dem=("dem", "first")).reset_index()
    out = best.assign(cd=best.c * best.dem).groupby(["hub_id", "flower"])[["cd", "dem"]].sum()
    out_cost = (out.cd / out.dem.replace(0, np.nan)).to_dict()

    hubs = prep.hubs.hub_id.tolist()
    pen = {x.flower: sc.quality_pct / 100.0 * x.box_value_eur for x in prep.flowers.itertuples()}
    tot_dem = {p: sum(v for (d, pp), v in prep.demand.items() if pp == p) for p in pen}
    keys = [k for k, v in prep.supply.items() if v > 0]                 # pares (finca, flor) con oferta
    fh = {(a.farm_id, a.hub_id): a for a in prep.farm_hub.itertuples()}

    G = pd.DataFrame(np.nan, index=hubs, columns=pd.MultiIndex.from_tuples(keys, names=["farm", "flower"]))
    for h in hubs:
        for f, p in keys:
            a, oc = fh.get((f, h)), out_cost.get((h, p), np.nan)
            if a is not None and not np.isnan(oc):
                G.loc[h, (f, p)] = a.cost_per_box + co2_eur * a.co2_kg_per_box + pen[p] * a.vase_loss + oc

    hub_info = prep.hubs.set_index("hub_id")
    fixed, cap = hub_info.fixed_cost_week.astype(float), hub_info.capacity.astype(float)
    # Todas las opciones (centro, finca, flor) en forma larga, ordenadas de la más barata a la más cara
    opts = G.stack(level=[0, 1], future_stack=True).dropna().rename("c").reset_index()
    opts.columns = ["hub", "farm", "flower", "c"]
    opts = opts.sort_values("c", kind="stable").reset_index(drop=True)

    def cost_of(S: list) -> float:
        """Reparto codicioso: la opción más barata primero, respetando pedido, oferta y capacidad de cada centro."""
        if not S:
            return float("inf")
        left = dict(tot_dem)
        sup = {k: float(prep.supply[k]) for k in keys}
        room = {h: float(cap[h]) for h in S}
        total = 0.0
        for o in opts[opts.hub.isin(S)].itertuples(index=False):
            take = min(left[o.flower], sup[(o.farm, o.flower)], room[o.hub])
            if take <= 0:
                continue
            total += take * o.c
            left[o.flower] -= take
            sup[(o.farm, o.flower)] -= take
            room[o.hub] -= take
        return float(total + fixed[S].sum() + UNMET_PENALTY * sum(left.values()))

    return G, cost_of


def _feasible_to_add(c, selected, conflicts):
    return not any((c == a and b in selected) or (c == b and a in selected) for a, b in conflicts)


def greedy_add(hubs: list, cost_of, n_min: int, n_max: int, conflicts: list) -> dict:
    """ADD: empieza sin centros y agrega el que más baja el costo total; para cuando abrir otro ya no ayuda."""
    selected, history, total_now = [], [], np.inf
    while len(selected) < min(n_max, len(hubs)):
        best_total, best_c = np.inf, None
        for c in hubs:
            if c in selected or not _feasible_to_add(c, selected, conflicts):
                continue
            total = cost_of(selected + [c])
            if total < best_total:
                best_total, best_c = total, c
        if best_c is None or (best_total >= total_now and len(selected) >= n_min):
            break                                           # abrir otro centro ya no reduce el costo
        selected.append(best_c)
        total_now = best_total
        history.append(dict(step=len(selected), hub=best_c, n_hubs=len(selected), est_cost=float(best_total)))
    return dict(hubs=selected, history=pd.DataFrame(history), est_cost=float(total_now))


def greedy_drop(hubs: list, cost_of, n_min: int, n_max: int, conflicts: list) -> dict:
    """DROP: empieza con todos abiertos y cierra el que más baja el costo total; para cuando cerrar otro ya no ayuda."""
    selected = list(hubs)
    n_conf = lambda S: sum(1 for a, b in conflicts if a in S and b in S)
    total_now = cost_of(selected)
    history = [dict(step=0, hub="(all open)", n_hubs=len(selected), est_cost=total_now)]
    while len(selected) > max(n_min, 1):
        must = len(selected) > n_max or n_conf(selected) > 0      # todavía incumple una regla: hay que cerrar
        best_total, best_c = np.inf, None
        for c in selected:
            S = [h for h in selected if h != c]
            # Si hay centros demasiado cerca, primero se cierra uno de los que está en conflicto
            if n_conf(selected) > 0 and n_conf(S) == n_conf(selected):
                continue
            total = cost_of(S)
            if total < best_total:
                best_total, best_c = total, c
        if best_c is None or (best_total >= total_now and not must):
            break
        selected.remove(best_c)
        total_now = best_total
        history.append(dict(step=len(history), hub=best_c, n_hubs=len(selected), est_cost=total_now))
    return dict(hubs=selected, history=pd.DataFrame(history), est_cost=total_now)


# ----------------------------------------------------------------------------
# Corrida completa de un escenario
# ----------------------------------------------------------------------------
def run_scenario(data: dict, sc: Scenario, active: frozenset, medoid_sites=None, all_medoids=None) -> dict:
    """
    Resuelve todo lo que muestra la app:
        after      plan reoptimizado con las disrupciones activas (z = 0)
        before     plan sin disrupciones
        all_air    referencia: todo en avión (la práctica actual)
        heuristics ADD y DROP contra el óptimo exacto
    """
    prep = prepare(data, sc, medoid_sites, all_medoids)
    before = solve(prep, frozenset())
    after = solve(prep, active) if active else before
    all_air = solve(prep, frozenset(), air_only=True)

    # Heurísticas de clase para escoger centros, evaluadas después con el modelo completo
    G, cost_of = heuristic_cost_matrix(prep, active)
    n_max = max(1, min(sc.n_max, len(G)))
    add = greedy_add(list(G.index), cost_of, min(sc.n_min, n_max), n_max, prep.conflicts)
    drop = greedy_drop(list(G.index), cost_of, min(sc.n_min, n_max), n_max, prep.conflicts)
    exact_set = set(after["open_hubs"])
    rows = [dict(method="Exact MILP (CBC)", hubs=", ".join(sorted(exact_set)), n_hubs=len(exact_set),
                 total_cost=after["cost"]["total"] + UNMET_PENALTY * after["unmet"], unmet=after["unmet"])]
    for name, h in (("ADD heuristic", add), ("DROP heuristic", drop)):
        S = set(h["hubs"])
        res = after if S == exact_set else solve(prep, active, fixed_hubs=S)
        rows.append(dict(method=name, hubs=", ".join(sorted(S)), n_hubs=len(S),
                         total_cost=res["cost"]["total"] + UNMET_PENALTY * res["unmet"], unmet=res["unmet"]))
    comp = pd.DataFrame(rows)
    comp["gap_pct"] = (comp.total_cost / comp.total_cost.iloc[0] - 1.0) * 100.0

    return dict(scenario=sc, market=sc.market, active=sorted(active), candidates=prep.hubs, conflicts=prep.conflicts,
                lanes=prep.lanes, chains=prep.chains,
                before=before, after=after, all_air=all_air,
                heuristics=dict(table=comp, add_history=add["history"], drop_history=drop["history"]))
