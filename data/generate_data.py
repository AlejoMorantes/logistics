"""
Genera todos los datos sintéticos de BloomChain (semilla fija, coordenadas reales aproximadas).

Uso:
    python data/generate_data.py

Crea estos CSV en data/:
    flowers.csv              tipos de flor (tallos por caja, vida en florero, valor de la caja)
    farms.csv                40 fincas con oferta semanal
    hubs.csv                 8 sitios candidatos para centro de acopio refrigerado
    nodes.csv                aeropuertos, puertos, llegadas a Europa, transbordos y clientes
    lanes.csv                tramos intercontinentales (avión o barco) con capacidad y costo fijo
    markets.csv              5 mercados europeos (NL, FR, CH, SE, FI) y su pedido semanal
    chains.csv               cadenas europeas multimodales: llegada -> (camión, tren, barcaza, feeder, ferry, vuelo) -> cliente
    demand.csv               demanda semanal por mercado, cliente y tipo de flor
    arcs.csv                 arcos en Colombia y tramo intercontinental: costo, tiempo, CO2 y vida en florero
    disruptions_history.csv  3 años diarios de disrupciones (para XGBoost)
    shipments_history.csv    12 meses de envíos (para el dashboard)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization import network as net  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
SEED = 42

HISTORY_START = "2023-11-01"
HISTORY_END = "2026-10-31"      # 3 años de historia
TEST_START = "2025-11-01"       # los últimos 12 meses se usan para probar el modelo y para el dashboard


# ----------------------------------------------------------------------------
# 1) Tipos de flor
# ----------------------------------------------------------------------------
def make_flowers() -> pd.DataFrame:
    return pd.DataFrame([
        # flor, etiqueta, tallos por caja, vida en florero al cortar (días), valor de la caja (EUR)
        dict(flower="rose", label="Rose", stems_per_box=300, vase_life_days=12.0, box_value_eur=105.0),
        dict(flower="carnation", label="Carnation", stems_per_box=500, vase_life_days=16.0, box_value_eur=80.0),
        dict(flower="chrysanthemum", label="Chrysanthemum", stems_per_box=240, vase_life_days=15.0, box_value_eur=72.0),
    ])


# ----------------------------------------------------------------------------
# 2) Fincas
# ----------------------------------------------------------------------------
# (municipio, lat, lon, número de fincas)
SABANA_TOWNS = [
    ("Madrid", 4.732, -74.264, 4), ("Facatativá", 4.813, -74.354, 3), ("Funza", 4.716, -74.212, 2),
    ("Bojacá", 4.733, -74.342, 1), ("El Rosal", 4.853, -74.263, 2), ("Subachoque", 4.928, -74.173, 1),
    ("Tenjo", 4.872, -74.144, 2), ("Tabio", 4.916, -74.098, 1), ("Chía", 4.858, -74.059, 2),
    ("Cajicá", 4.918, -74.028, 2), ("Zipaquirá", 5.022, -74.004, 1), ("Tocancipá", 4.965, -73.913, 3),
    ("Gachancipá", 4.991, -73.873, 1), ("Sopó", 4.908, -73.940, 1), ("Nemocón", 5.068, -73.878, 1),
    ("Suesca", 5.103, -73.798, 1),
]  # 28 fincas
ANTIOQUIA_TOWNS = [
    ("Rionegro", 6.155, -75.374, 3), ("La Ceja", 6.031, -75.433, 3), ("El Carmen de Viboral", 6.083, -75.335, 2),
    ("La Unión", 5.973, -75.361, 1), ("Marinilla", 6.174, -75.336, 1), ("Guarne", 6.280, -75.443, 1),
    ("El Retiro", 6.058, -75.503, 1),
]  # 12 fincas

FARM_NAMES = [
    "La Esperanza", "San Rafael", "El Jardín", "Santa Mónica", "Las Margaritas", "El Rocío", "La Primavera",
    "Buenavista", "San Isidro", "Los Arrayanes", "La Pradera", "El Encanto", "Santa Bárbara", "Las Palmas",
    "La Aurora", "El Paraíso", "San Marino", "La Gaitana", "Los Sauces", "El Trébol", "Santa Helena",
    "La Conchita", "El Cerezo", "Monteverde", "La Floresta", "San Valentín", "El Redil", "Altamira",
    "La Colina", "El Capiro", "Guadalupe", "Las Acacias", "La Montaña", "San Nicolás", "El Tambo",
    "Llanogrande", "La Selva", "Vista Hermosa", "El Vergel", "Santa Elena",
]


def make_farms(rng: np.random.Generator, flowers: pd.DataFrame) -> pd.DataFrame:
    stems_box = flowers.set_index("flower")["stems_per_box"]
    rows, k = [], 0
    for region, towns in (("Sabana de Bogotá", SABANA_TOWNS), ("Rionegro, Antioquia", ANTIOQUIA_TOWNS)):
        for town, lat, lon, n in towns:
            for _ in range(n):
                # Cada finca queda a unos 1-3 km del casco urbano de su municipio
                f_lat = lat + rng.normal(0, 0.013)
                f_lon = lon + rng.normal(0, 0.013)
                if region.startswith("Sabana"):
                    # La Sabana es sobre todo rosa y clavel
                    rose = rng.uniform(55_000, 100_000) if rng.random() < 0.93 else 0.0
                    carn = rng.uniform(50_000, 115_000) if rng.random() < 0.72 else 0.0
                    chry = rng.uniform(8_000, 20_000) if rng.random() < 0.20 else 0.0
                else:
                    # El oriente antioqueño es sobre todo crisantemo
                    rose = rng.uniform(15_000, 35_000) if rng.random() < 0.35 else 0.0
                    carn = rng.uniform(40_000, 80_000) if rng.random() < 0.30 else 0.0
                    chry = rng.uniform(45_000, 85_000)
                stems = dict(rose=rose, carnation=carn, chrysanthemum=chry)
                row = dict(farm_id=f"F{k + 1:02d}", name=f"Finca {FARM_NAMES[k]}", region=region,
                           municipality=town, lat=round(f_lat, 5), lon=round(f_lon, 5))
                for fl, s in stems.items():
                    s = int(round(s, -2))
                    row[f"stems_{fl}"] = s
                    row[f"boxes_{fl}"] = int(round(s / stems_box[fl]))
                rows.append(row)
                k += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3) Sitios candidatos para centros de acopio refrigerados
# ----------------------------------------------------------------------------
def make_hubs() -> pd.DataFrame:
    return pd.DataFrame([
        dict(hub_id="H1", name="Madrid cold hub", region="Sabana de Bogotá", lat=4.7350, lon=-74.2550, fixed_cost_week=5200, capacity_boxes=4500),
        dict(hub_id="H2", name="Facatativá cold hub", region="Sabana de Bogotá", lat=4.8100, lon=-74.3450, fixed_cost_week=4300, capacity_boxes=3500),
        dict(hub_id="H3", name="Funza – El Dorado cold hub", region="Sabana de Bogotá", lat=4.7150, lon=-74.2000, fixed_cost_week=6400, capacity_boxes=5000),
        dict(hub_id="H4", name="Tocancipá cold hub", region="Sabana de Bogotá", lat=4.9650, lon=-73.9150, fixed_cost_week=4800, capacity_boxes=4000),
        dict(hub_id="H5", name="Chía – Cajicá cold hub", region="Sabana de Bogotá", lat=4.8900, lon=-74.0400, fixed_cost_week=5000, capacity_boxes=3500),
        dict(hub_id="H6", name="Rionegro – JMC cold hub", region="Rionegro, Antioquia", lat=6.1650, lon=-75.4000, fixed_cost_week=5000, capacity_boxes=4000),
        dict(hub_id="H7", name="La Ceja cold hub", region="Rionegro, Antioquia", lat=6.0300, lon=-75.4300, fixed_cost_week=4000, capacity_boxes=3000),
        dict(hub_id="H8", name="El Carmen de Viboral cold hub", region="Rionegro, Antioquia", lat=6.0850, lon=-75.3350, fixed_cost_week=3500, capacity_boxes=2500),
    ])


# ----------------------------------------------------------------------------
# 4) Mercados de destino en Europa
# ----------------------------------------------------------------------------
# El pedido semanal sale del tamaño del mercado: población x gasto anual en flor cortada por persona.
# BOXES_PER_MEUR = cajas por semana que este grupo de fincas vende por cada millón de euros de gasto anual.
# Los Países Bajos son distintos: sus subastas compran para revender a toda Europa, así que piden
# el 88% de la oferta. Cifras redondeadas e ilustrativas.
BOXES_PER_MEUR = 2.6
MARKETS = [
    dict(market="NL", name="Netherlands", role="Dutch flower auctions, which re-export to the rest of Europe",
         population_m=17.9, spend_eur_capita=55, supply_share=0.88),
    dict(market="FR", name="France", role="Wholesale markets in Paris, Lyon and Marseille",
         population_m=68.4, spend_eur_capita=35, supply_share=None),
    dict(market="CH", name="Switzerland", role="Landlocked, high-spending market: Zurich, Basel and Geneva",
         population_m=9.0, spend_eur_capita=95, supply_share=None),
    dict(market="SE", name="Sweden", role="Wholesalers in Stockholm, Gothenburg and Malmö",
         population_m=10.6, spend_eur_capita=60, supply_share=None),
    dict(market="FI", name="Finland", role="Small, distant market across the Baltic: Helsinki, Tampere and Turku",
         population_m=5.6, spend_eur_capita=50, supply_share=None),
]
# Reparto del pedido entre las ciudades de cada mercado
DEST_SHARE = {
    "NL": {"AAL": 0.55, "NAA": 0.25, "RIJ": 0.20},
    "FR": {"RUN": 0.55, "LYS": 0.25, "MRS": 0.20},
    "CH": {"ZUR": 0.45, "BAS": 0.25, "GVA": 0.30},
    "SE": {"STO": 0.50, "GBG": 0.30, "MMA": 0.20},
    "FI": {"HKI": 0.60, "TMP": 0.22, "TKU": 0.18},
}


def make_markets(farms: pd.DataFrame) -> pd.DataFrame:
    supply = sum(int(farms[c].sum()) for c in farms.columns if c.startswith("boxes_"))
    rows = []
    for m in MARKETS:
        spend_meur = m["population_m"] * m["spend_eur_capita"]
        if m["supply_share"]:
            weekly = supply * m["supply_share"]
        else:
            weekly = BOXES_PER_MEUR * spend_meur
        rows.append(dict(market=m["market"], name=m["name"], role=m["role"], population_m=m["population_m"],
                         spend_eur_capita=m["spend_eur_capita"], spend_meur_year=round(spend_meur),
                         weekly_boxes=int(round(weekly, -1))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5) Nodos: salida en Colombia, llegada a Europa, transbordos y clientes
# ----------------------------------------------------------------------------
def make_nodes() -> pd.DataFrame:
    N = lambda i, name, short, typ, kind, country, market, lat, lon: dict(
        node_id=i, name=name, short=short, type=typ, kind=kind, country=country, market=market, lat=lat, lon=lon)
    return pd.DataFrame([
        # Salida desde Colombia
        N("BOG", "El Dorado Airport (Bogotá)", "Bogotá airport", "airport", "airport", "Colombia", "", 4.7016, -74.1469),
        N("MDE", "José María Córdova Airport (Rionegro)", "Rionegro airport", "airport", "airport", "Colombia", "", 6.1645, -75.4231),
        N("CTG", "Port of Cartagena", "Cartagena port", "port", "seaport", "Colombia", "", 10.4000, -75.5300),
        N("SMR", "Port of Santa Marta", "Santa Marta port", "port", "seaport", "Colombia", "", 11.2500, -74.2100),
        # Llegada a Europa (fin del tramo intercontinental)
        N("AMS", "Amsterdam Schiphol Airport", "Schiphol", "arrival_airport", "airport", "Netherlands", "", 52.3105, 4.7683),
        N("CDG", "Paris Charles de Gaulle Airport", "Paris-CDG", "arrival_airport", "airport", "France", "", 49.0097, 2.5479),
        N("RTM", "Port of Rotterdam (Maasvlakte)", "Rotterdam port", "arrival_port", "seaport", "Netherlands", "", 51.9500, 4.0500),
        N("LEH", "Port of Le Havre", "Le Havre port", "arrival_port", "seaport", "France", "", 49.4800, 0.1100),
        N("FOS", "Port of Marseille-Fos", "Fos port", "arrival_port", "seaport", "France", "", 43.4200, 4.8800),
        N("GOA", "Port of Genoa", "Genoa port", "arrival_port", "seaport", "Italy", "", 44.4000, 8.9000),
        # Transbordos dentro de Europa
        N("ARN", "Stockholm Arlanda Airport", "Arlanda", "transfer", "airport", "Sweden", "", 59.6519, 17.9186),
        N("HEL", "Helsinki-Vantaa Airport", "Helsinki airport", "transfer", "airport", "Finland", "", 60.3172, 24.9633),
        N("ZRH", "Zurich Airport", "Zurich airport", "transfer", "airport", "Switzerland", "", 47.4582, 8.5555),
        N("GOT", "Port of Gothenburg", "Gothenburg port", "transfer", "seaport", "Sweden", "", 57.6900, 11.8300),
        N("VUO", "Port of Helsinki (Vuosaari)", "Vuosaari port", "transfer", "seaport", "Finland", "", 60.2100, 25.1900),
        N("TRV", "Travemünde ferry terminal", "Travemünde", "transfer", "ferry_port", "Germany", "", 53.9600, 10.8700),
        N("ALP", "Alphen aan den Rijn barge terminal", "Alphen barge terminal", "transfer", "river_port", "Netherlands", "", 52.1300, 4.6600),
        N("BSL", "Basel Rhine ports", "Basel Rhine port", "transfer", "river_port", "Switzerland", "", 47.5900, 7.5900),
        N("GEN", "Paris-Gennevilliers river port", "Gennevilliers port", "transfer", "river_port", "France", "", 48.9300, 2.2800),
        N("LYP", "Lyon Édouard Herriot river port", "Lyon river port", "transfer", "river_port", "France", "", 45.7300, 4.8200),
        # Clientes por mercado
        N("AAL", "Aalsmeer flower auction", "Aalsmeer", "destination", "destination", "Netherlands", "NL", 52.2580, 4.7900),
        N("NAA", "Naaldwijk flower auction", "Naaldwijk", "destination", "destination", "Netherlands", "NL", 51.9930, 4.2080),
        N("RIJ", "Rijnsburg flower auction", "Rijnsburg", "destination", "destination", "Netherlands", "NL", 52.1990, 4.4470),
        N("RUN", "Rungis wholesale market (Paris)", "Paris-Rungis", "destination", "destination", "France", "FR", 48.7590, 2.3520),
        N("LYS", "Lyon-Corbas wholesale market", "Lyon", "destination", "destination", "France", "FR", 45.6660, 4.9060),
        N("MRS", "Marseille wholesale market", "Marseille", "destination", "destination", "France", "FR", 43.3310, 5.3830),
        N("ZUR", "Wholesale customer, Zurich", "Zurich", "destination", "destination", "Switzerland", "CH", 47.4000, 8.4800),
        N("BAS", "Wholesale customer, Basel", "Basel", "destination", "destination", "Switzerland", "CH", 47.5600, 7.6000),
        N("GVA", "Wholesale customer, Geneva", "Geneva", "destination", "destination", "Switzerland", "CH", 46.2000, 6.1400),
        N("STO", "Wholesale customer, Stockholm", "Stockholm", "destination", "destination", "Sweden", "SE", 59.3000, 18.0500),
        N("GBG", "Wholesale customer, Gothenburg", "Gothenburg", "destination", "destination", "Sweden", "SE", 57.7100, 11.9700),
        N("MMA", "Wholesale customer, Malmö", "Malmö", "destination", "destination", "Sweden", "SE", 55.6000, 13.0000),
        N("HKI", "Wholesale customer, Helsinki", "Helsinki", "destination", "destination", "Finland", "FI", 60.2900, 25.0400),
        N("TMP", "Wholesale customer, Tampere", "Tampere", "destination", "destination", "Finland", "FI", 61.5000, 23.7600),
        N("TKU", "Wholesale customer, Turku", "Turku", "destination", "destination", "Finland", "FI", 60.4500, 22.2700),
    ])


# ----------------------------------------------------------------------------
# 6) Tramos intercontinentales (lanes): avión o contenedor refrigerado por barco
# ----------------------------------------------------------------------------
def make_lanes(nodes: pd.DataFrame) -> pd.DataFrame:
    n = nodes.set_index("node_id")
    specs = [
        # id, modo, origen, destino, capacidad semanal (cajas), costo fijo semanal (EUR)
        ("AIR_BOG_AMS", "air", "BOG", "AMS", 14_000, 2_000),
        ("AIR_MDE_AMS", "air", "MDE", "AMS", 7_000, 2_000),
        ("AIR_BOG_CDG", "air", "BOG", "CDG", 6_000, 2_000),
        ("SEA_CTG_RTM", "sea", "CTG", "RTM", 4_500, 3_500),
        ("SEA_SMR_RTM", "sea", "SMR", "RTM", 3_000, 3_500),
        ("SEA_CTG_LEH", "sea", "CTG", "LEH", 3_500, 3_500),
        ("SEA_SMR_FOS", "sea", "SMR", "FOS", 2_500, 3_500),
        ("SEA_SMR_GOA", "sea", "SMR", "GOA", 2_500, 3_500),
    ]
    rows = []
    for lane_id, mode, o, d, cap, fixed in specs:
        if mode == "air":
            km = float(net.haversine_km(n.loc[o, "lat"], n.loc[o, "lon"], n.loc[d, "lat"], n.loc[d, "lon"]))
        else:
            km = net.path_length_km(net.SEA_WAYPOINTS[lane_id])
        rows.append(dict(lane_id=lane_id, mode=mode, origin=o, dest=d, **net.lane_arc(km, mode),
                         capacity_boxes=cap, fixed_cost_week=fixed))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 7) Cadenas europeas: de la llegada al cliente, con uno o varios modos
# ----------------------------------------------------------------------------
# Cada cadena es una lista de tramos (desde, hasta, modo). El último "hasta" es el cliente.
T, R, B, F, X, A = "truck", "rail", "barge", "feeder", "ferry", "flight"
CHAIN_SPECS = {
    "NL": [
        [("AMS", "AAL", T)], [("AMS", "NAA", T)], [("AMS", "RIJ", T)],
        [("RTM", "AAL", T)], [("RTM", "NAA", T)], [("RTM", "RIJ", T)],
        [("RTM", "ALP", B), ("ALP", "AAL", T)], [("RTM", "ALP", B), ("ALP", "RIJ", T)],
    ],
    "FR": [
        [("AMS", "RUN", T)], [("AMS", "LYS", T)], [("AMS", "MRS", T)],
        [("CDG", "RUN", T)], [("CDG", "LYS", T)], [("CDG", "MRS", T)],
        [("RTM", "RUN", T)], [("RTM", "LYS", T)],
        [("LEH", "RUN", T)], [("LEH", "GEN", B), ("GEN", "RUN", T)], [("LEH", "LYS", R)],
        [("FOS", "MRS", T)], [("FOS", "LYS", R)], [("FOS", "LYP", B), ("LYP", "LYS", T)], [("FOS", "RUN", R)],
    ],
    "CH": [
        [("AMS", "ZUR", T)], [("AMS", "BAS", T)], [("AMS", "GVA", T)],
        [("AMS", "ZRH", A), ("ZRH", "ZUR", T)], [("AMS", "ZRH", A), ("ZRH", "BAS", T)], [("AMS", "ZRH", A), ("ZRH", "GVA", T)],
        [("CDG", "BAS", T)], [("CDG", "GVA", T)],
        [("RTM", "ZUR", T)], [("RTM", "BAS", T)],
        [("RTM", "BSL", B), ("BSL", "BAS", T)], [("RTM", "BSL", B), ("BSL", "ZUR", T)],
        [("GOA", "ZUR", R)], [("GOA", "BAS", R)], [("GOA", "GVA", T)],
        [("FOS", "GVA", R)],
    ],
    "SE": [
        [("AMS", "STO", T)], [("AMS", "GBG", T)], [("AMS", "MMA", T)],
        [("AMS", "ARN", A), ("ARN", "STO", T)],
        [("RTM", "STO", T)], [("RTM", "GBG", T)], [("RTM", "MMA", T)],
        [("RTM", "GOT", F), ("GOT", "STO", R)], [("RTM", "GOT", F), ("GOT", "GBG", T)], [("RTM", "GOT", F), ("GOT", "MMA", T)],
    ],
    "FI": [
        [("AMS", "HEL", A), ("HEL", "HKI", T)], [("AMS", "HEL", A), ("HEL", "TMP", T)], [("AMS", "HEL", A), ("HEL", "TKU", T)],
        [("AMS", "TRV", T), ("TRV", "VUO", X), ("VUO", "HKI", T)],
        [("AMS", "TRV", T), ("TRV", "VUO", X), ("VUO", "TMP", T)],
        [("AMS", "TRV", T), ("TRV", "VUO", X), ("VUO", "TKU", T)],
        [("RTM", "VUO", F), ("VUO", "HKI", T)], [("RTM", "VUO", F), ("VUO", "TMP", T)], [("RTM", "VUO", F), ("VUO", "TKU", T)],
        [("RTM", "TRV", T), ("TRV", "VUO", X), ("VUO", "HKI", T)],
    ],
}


def make_chains(nodes: pd.DataFrame) -> pd.DataFrame:
    n = nodes.set_index("node_id")
    rows = []
    for market, specs in CHAIN_SPECS.items():
        for i, spec in enumerate(specs, start=1):
            arrival = spec[0][0]
            unit = "pallet" if n.loc[arrival, "type"] == "arrival_airport" else "container"
            segs = []
            for a, b, mode in spec:
                pa, pb = (a, n.loc[a, "lat"], n.loc[a, "lon"]), (b, n.loc[b, "lat"], n.loc[b, "lon"])
                segs.append(net.eu_segment(pa, pb, mode, unit, net.EU_PATHS.get((a, b))))
            modes = [s["mode"] for s in segs]
            label = " + ".join(dict.fromkeys(net.EU_MODE[m]["label"] for m in modes))
            # Cada transbordo intermedio suma su manejo (ya incluido en fixed_h de cada tramo)
            rows.append(dict(
                chain_id=f"{market}_{i:02d}", market=market, arrival=arrival, dest_id=spec[-1][1], unit=unit,
                modes="+".join(modes), label=label, via=" ".join(s["dest"] for s in segs[:-1]),
                distance_km=round(sum(s["distance_km"] for s in segs), 1),
                cost_per_box=round(sum(s["cost_per_box"] for s in segs), 3),
                time_h=round(sum(s["time_h"] for s in segs), 2),
                co2_kg_per_box=round(sum(s["co2_kg_per_box"] for s in segs), 3),
                vase_loss=round(sum(s["vase_loss"] for s in segs), 4),
                segments=json.dumps([dict(origin=s["origin"], dest=s["dest"], mode=s["mode"], path=s["path"],
                                          distance_km=s["distance_km"], time_h=s["time_h"]) for s in segs]),
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 8) Demanda semanal por mercado, cliente y flor
# ----------------------------------------------------------------------------
def make_demand(farms: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """El pedido de cada mercado se reparte por flor igual que la oferta, y por ciudad según DEST_SHARE."""
    supply = {fl: farms[f"boxes_{fl}"].sum() for fl in ("rose", "carnation", "chrysanthemum")}
    total = sum(supply.values())
    rows = []
    for m in markets.itertuples():
        for fl, s in supply.items():
            for d, share in DEST_SHARE[m.market].items():
                rows.append(dict(market=m.market, dest_id=d, flower=fl, boxes=int(round(m.weekly_boxes * s / total * share))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 9) Tabla de arcos (Colombia y tramo intercontinental)
# ----------------------------------------------------------------------------
def make_arcs(farms, hubs, nodes, lanes) -> pd.DataFrame:
    rows = []
    # Finca -> centro de acopio (solo dentro de la misma región)
    for _, f in farms.iterrows():
        for _, h in hubs[hubs.region == f.region].iterrows():
            km = float(net.haversine_km(f.lat, f.lon, h.lat, h.lon))
            rows.append(dict(origin=f.farm_id, dest=h.hub_id, arc_type="farm_hub", mode="truck", **net.road_arc(km, "farm_hub")))
    # Centro de acopio -> aeropuerto o puerto
    gateways = nodes[nodes.type.isin(["airport", "port"])]
    for _, h in hubs.iterrows():
        for _, g in gateways.iterrows():
            km = float(net.haversine_km(h.lat, h.lon, g.lat, g.lon))
            rows.append(dict(origin=h.hub_id, dest=g.node_id, arc_type="hub_gateway", mode="reefer_truck", **net.road_arc(km, "hub_gateway")))
    # Tramo principal
    for _, l in lanes.iterrows():
        rows.append(dict(origin=l.origin, dest=l.dest, arc_type="lane", mode=l["mode"], distance_km=l.distance_km,
                         cost_per_box=l.cost_per_box, time_h=l.time_h, co2_kg_per_box=l.co2_kg_per_box,
                         vase_loss_per_day=l.vase_loss_per_day))
    # El tramo europeo está en chains.csv (cada cadena con todos sus tramos)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 10) Histórico diario de disrupciones (3 años)
# ----------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ar1(rng, n, phi, sigma):
    """Serie autorregresiva simple: hoy se parece a ayer."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return x


def make_disruptions(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(HISTORY_START, HISTORY_END, freq="D")
    n = len(dates)
    month = dates.month.to_numpy()
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()

    # Temporada alta: San Valentín (20 ene - 10 feb) y Día de la Madre (20 abr - 8 may)
    md = month * 100 + dates.day.to_numpy()
    peak = (((md >= 120) & (md <= 210)) | ((md >= 420) & (md <= 508))).astype(int)

    # --- Clima en Bogotá: dos temporadas de lluvia (abr-may y oct-nov) ---
    rainy_bog = np.isin(month, [4, 5, 10, 11]).astype(float)
    rain_bog = rng.gamma(0.8, 4.0, n) * (0.5 + 1.6 * rainy_bog)
    humidity = np.clip(70 + 0.9 * rain_bog + 6 * rainy_bog + rng.normal(0, 5, n), 45, 99)
    temp_min = np.clip(8.2 - 1.6 * np.isin(month, [12, 1, 2]) + 0.8 * rainy_bog + rng.normal(0, 1.3, n), 1, 13)
    wind_bog = np.clip(rng.gamma(3.0, 3.0, n), 1, 35)

    # --- Paros: se anuncian y duran varios días seguidos ---
    announced = np.zeros(n, int)
    t = 0
    while t < n:
        if rng.random() < 0.012:  # empieza un episodio de paro
            length = int(rng.integers(2, 7))
            announced[t:t + length] = 1
            t += length + int(rng.integers(5, 20))
        else:
            t += 1
    fuel_change = np.round(rng.normal(0.2, 1.0, n) + 3.5 * (rng.random(n) < 0.03), 2)

    # --- Puerto de Cartagena ---
    occupancy = np.clip(72 + 9 * _ar1(rng, n, 0.85, 0.55) + 7 * peak + rng.normal(0, 2, n), 40, 99)
    vessels_wait = rng.poisson(np.clip((occupancy - 55) / 6.5, 0.3, None))
    rain_car = rng.gamma(0.7, 5.0, n) * (0.4 + 1.5 * np.isin(month, [5, 6, 9, 10, 11]))

    # --- Atlántico: temporada de huracanes de junio a noviembre ---
    hurricane = np.isin(month, [6, 7, 8, 9, 10, 11]).astype(float)
    storm_drive = np.clip(_ar1(rng, n, 0.80, 0.60) + 0.9 * hurricane * np.sin(np.pi * (doy - 150) / 185).clip(0), -3, 4)
    wave = np.clip(1.7 + 0.75 * storm_drive + rng.normal(0, 0.35, n), 0.4, 8)
    wind_car = np.clip(22 + 7 * storm_drive + 5 * np.isin(month, [12, 1, 2, 3]) + rng.normal(0, 4, n), 5, 110)
    pressure = np.round(-3.2 * storm_drive.clip(0) + rng.normal(0, 1.5, n), 1)

    # --- Etiquetas: probabilidad logística + efecto de "ayer también pasó" ---
    fog = np.zeros(n, int); strike = np.zeros(n, int); cong = np.zeros(n, int); storm = np.zeros(n, int)
    for t in range(n):
        f_prev = fog[t - 1] if t else 0
        s_prev = strike[t - 1] if t else 0
        c_prev = cong[t - 1] if t else 0
        st_prev = storm[t - 1] if t else 0
        # Niebla en BOG: humedad alta, madrugada fría y poco viento
        z = -2.5 + 0.17 * (humidity[t] - 78) - 0.60 * (temp_min[t] - 8) - 0.20 * (wind_bog[t] - 9) + 0.9 * f_prev
        fog[t] = rng.random() < _sigmoid(z)
        # Paro en carreteras: anuncio previo, alza del combustible, entre semana
        z = -4.4 + 3.6 * announced[t] + 0.35 * fuel_change[t] + 0.5 * (dow[t] < 5) + 1.6 * s_prev
        strike[t] = rng.random() < _sigmoid(z)
        # Congestión en Cartagena: patio lleno, barcos esperando, temporada alta
        z = -3.6 + 0.16 * (occupancy[t] - 75) + 0.30 * (vessels_wait[t] - 3) + 0.6 * peak[t] + 0.02 * rain_car[t] + 1.0 * c_prev
        cong[t] = rng.random() < _sigmoid(z)
        # Tormenta en el Atlántico: oleaje, viento y baja presión en temporada de huracanes
        z = -4.6 + 1.60 * (wave[t] - 2.0) + 0.045 * (wind_car[t] - 25) - 0.10 * pressure[t] + 0.5 * hurricane[t] + 0.8 * st_prev
        storm[t] = rng.random() < _sigmoid(z)

    # --- Retrasos observados (sirven como variable "retraso del día anterior") ---
    delay_air = np.round(np.clip(rng.gamma(2.0, 18, n) + fog * rng.uniform(300, 1300, n) + strike * rng.uniform(60, 400, n), 0, None))
    delay_sea = np.round(np.clip(rng.gamma(2.0, 2.5, n) + cong * rng.uniform(26, 70, n) + storm * rng.uniform(28, 90, n) + strike * rng.uniform(4, 30, n), 0, None), 1)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "month": month, "day_of_week": dow, "peak_season": peak,
        "rain_bog_mm": np.round(rain_bog, 1), "humidity_bog_pct": np.round(humidity, 1),
        "temp_min_bog_c": np.round(temp_min, 1), "wind_bog_kmh": np.round(wind_bog, 1),
        "strike_announced": announced, "fuel_price_change_pct": fuel_change,
        "port_yard_occupancy_pct": np.round(occupancy, 1), "vessels_waiting_ctg": vessels_wait,
        "rain_caribbean_mm": np.round(rain_car, 1), "wind_caribbean_kmh": np.round(wind_car, 1),
        "wave_height_atl_m": np.round(wave, 2), "pressure_anomaly_hpa": pressure,
        # Lo que pasó ayer (se conoce al planear el envío de mañana)
        "delay_prev_air_min": np.concatenate([[0], delay_air[:-1]]),
        "delay_prev_sea_h": np.concatenate([[0], delay_sea[:-1]]),
        "fog_prev_day": np.concatenate([[0], fog[:-1]]),
        "strike_prev_day": np.concatenate([[0], strike[:-1]]),
        "congestion_prev_day": np.concatenate([[0], cong[:-1]]),
        "storm_prev_day": np.concatenate([[0], storm[:-1]]),
        # Etiquetas binarias: 1 = el nodo o arco cierra o se retrasa más de 24 h
        "disrupted_fog_bog": fog, "disrupted_road_strike": strike,
        "disrupted_port_congestion": cong, "disrupted_atlantic_storm": storm,
        # Retrasos del propio día (solo para el dashboard; no se usan como variables del modelo)
        "delay_air_min": delay_air, "delay_sea_h": delay_sea,
    })
    return df


# ----------------------------------------------------------------------------
# 11) Histórico de envíos (12 meses) para el dashboard
# ----------------------------------------------------------------------------
# Práctica actual en cada mercado: casi todo vuela. (lane, parte de rosa y clavel, parte de crisantemo)
# El crisantemo sale sobre todo de Rionegro (MDE); rosa y clavel, de la Sabana (BOG).
AIR_PRACTICE = {
    "NL": [("AIR_BOG_AMS", 0.80, 0.25), ("AIR_MDE_AMS", 0.20, 0.75)],
    "FR": [("AIR_BOG_CDG", 0.55, 0.15), ("AIR_BOG_AMS", 0.25, 0.10), ("AIR_MDE_AMS", 0.20, 0.75)],
    "CH": [("AIR_BOG_AMS", 0.80, 0.25), ("AIR_MDE_AMS", 0.20, 0.75)],
    "SE": [("AIR_BOG_AMS", 0.80, 0.25), ("AIR_MDE_AMS", 0.20, 0.75)],
    "FI": [("AIR_BOG_AMS", 0.80, 0.25), ("AIR_MDE_AMS", 0.20, 0.75)],
}
# Piloto marítimo pequeño (martes y viernes): (lane, probabilidad de usarlo)
SEA_PILOT = {
    "NL": [("SEA_CTG_RTM", 0.65), ("SEA_SMR_RTM", 0.35)],
    "FR": [("SEA_CTG_LEH", 0.60), ("SEA_SMR_FOS", 0.40)],
    "CH": [("SEA_CTG_RTM", 0.55), ("SEA_SMR_GOA", 0.45)],
    "SE": [("SEA_CTG_RTM", 0.65), ("SEA_SMR_RTM", 0.35)],
    "FI": [("SEA_CTG_RTM", 0.65), ("SEA_SMR_RTM", 0.35)],
}


def make_shipments(rng, disruptions, demand, lanes, chains) -> pd.DataFrame:
    """Hoy casi todo va en avión; el barco es un piloto pequeño que sale martes y viernes."""
    hist = disruptions[disruptions.date >= TEST_START].reset_index(drop=True)
    lane = lanes.set_index("lane_id")
    # Tramo europeo de hoy: después del avión la cadena más rápida; después del barco la más barata
    fastest = chains.sort_values("time_h").drop_duplicates(["market", "arrival", "dest_id"]).set_index(["market", "arrival", "dest_id"])
    cheapest = chains.sort_values("cost_per_box").drop_duplicates(["market", "arrival", "dest_id"]).set_index(["market", "arrival", "dest_id"])
    rows = []
    for _, day in hist.iterrows():
        d = pd.Timestamp(day.date)
        season = 1.0 + 0.9 * day.peak_season + 0.08 * np.sin(2 * np.pi * d.dayofyear / 365)
        sea_day = d.dayofweek in (1, 4)
        for _, dem in demand.iterrows():
            daily = dem.boxes / 7.0 * season * rng.uniform(0.75, 1.25)
            options = [(lid, s_chry if dem.flower == "chrysanthemum" else s_other) for lid, s_other, s_chry in AIR_PRACTICE[dem.market]]
            plan = [(lid, share * daily * (0.86 if sea_day and dem.flower != "rose" else 1.0)) for lid, share in options]
            if sea_day and dem.flower != "rose":
                pilots = SEA_PILOT[dem.market]
                pick = pilots[0][0] if rng.random() < pilots[0][1] else pilots[1][0]
                plan.append((pick, daily * 0.14 * 3.5))
            for lid, boxes in plan:
                boxes = int(round(boxes))
                if boxes <= 0:
                    continue
                l = lane.loc[lid]
                table = fastest if l["mode"] == "air" else cheapest
                key = (dem.market, l.dest, dem.dest_id)
                if key not in table.index:
                    continue
                chain = table.loc[key]
                planned_h = 8.0 + l.time_h + chain.time_h
                if l["mode"] == "air":
                    delay_min = rng.gamma(2.0, 20)
                    if l.origin == "BOG" and day.disrupted_fog_bog:
                        delay_min += rng.uniform(400, 1500)
                    if day.disrupted_road_strike:
                        delay_min += rng.uniform(60, 420)
                    loss_rate = 0.004 + 0.0045 * delay_min / 60.0 / 6.0
                else:
                    delay_min = rng.gamma(2.0, 150)
                    if l.origin == "CTG" and day.disrupted_port_congestion:
                        delay_min += rng.uniform(1500, 4200)
                    if day.disrupted_atlantic_storm:
                        delay_min += rng.uniform(1600, 5200)
                    if day.disrupted_road_strike:
                        delay_min += rng.uniform(300, 1800)
                    loss_rate = 0.009 + 0.0035 * delay_min / 60.0 / 12.0
                if dem.flower == "rose":
                    loss_rate *= 1.5  # la rosa es la más delicada
                transit_min = planned_h * 60 * rng.uniform(0.97, 1.04) + delay_min
                rows.append(dict(
                    date=day.date, market=dem.market, flower=dem.flower, gateway=l.origin, mode=l["mode"], lane_id=lid,
                    arrival=l.dest, eu_leg=chain.label, destination=dem.dest_id, boxes=boxes,
                    transit_minutes=int(round(transit_min)), delay_minutes=int(round(delay_min)),
                    boxes_lost_quality=int(rng.binomial(boxes, min(loss_rate, 0.6))),
                ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)
    flowers = make_flowers()
    farms = make_farms(rng, flowers)
    hubs = make_hubs()
    nodes = make_nodes()
    lanes = make_lanes(nodes)
    markets = make_markets(farms)
    chains = make_chains(nodes)
    demand = make_demand(farms, markets)
    arcs = make_arcs(farms, hubs, nodes, lanes)
    disruptions = make_disruptions(rng)
    shipments = make_shipments(rng, disruptions, demand, lanes, chains)

    out = dict(flowers=flowers, farms=farms, hubs=hubs, nodes=nodes, lanes=lanes, markets=markets, chains=chains,
               demand=demand, arcs=arcs, disruptions_history=disruptions, shipments_history=shipments)
    for name, df in out.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8")
        print(f"{name + '.csv':28s} {len(df):6d} filas")

    print("\nOferta semanal (cajas):", {fl: int(farms[f'boxes_{fl}'].sum()) for fl in flowers.flower})
    print("Pedido semanal por mercado (cajas):", demand.groupby("market").boxes.sum().to_dict())
    labels = [c for c in disruptions.columns if c.startswith("disrupted_")]
    print("Frecuencia de disrupciones:", disruptions[labels].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
