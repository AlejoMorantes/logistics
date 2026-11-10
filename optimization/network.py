"""
Parámetros físicos y económicos de la red (un solo lugar para todos los supuestos).

Todo el proyecto usa estas mismas funciones para calcular distancia, costo, tiempo,
CO2 y pérdida de vida en florero de cada arco, tanto al generar los CSV como al
crear arcos nuevos dentro de la app (por ejemplo, hacia un medoide de k-medoids).

Unidad de trabajo: una caja estándar de flores de 15 kg.
Los valores son supuestos ilustrativos de orden de magnitud, no datos oficiales.
"""
from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------
# Supuestos generales
# ----------------------------------------------------------------------------
BOX_WEIGHT_T = 0.015  # 15 kg por caja, en toneladas

# Parámetros por tipo de tramo:
#   circuity   = factor para pasar de línea recta (Haversine) a distancia por carretera
#   speed_kmh  = velocidad media
#   eur_box_km = costo de transporte por caja y por km
#   kg_co2_tkm = factor de emisión por tonelada-km
#   vase_loss  = días de vida en florero que se pierden por cada día de tránsito
LEG = {
    # Finca -> centro de acopio: camión pequeño, vías rurales, sin frío
    "farm_hub": dict(circuity=1.30, speed_kmh=35.0, eur_box_km=0.0060, kg_co2_tkm=0.25, vase_loss=1.50, fixed_h=0.5),
    # Centro de acopio -> aeropuerto o puerto: tractomula refrigerada por los Andes
    "hub_gateway": dict(circuity=1.50, speed_kmh=45.0, eur_box_km=0.0025, kg_co2_tkm=0.11, vase_loss=0.40, fixed_h=1.0),
    # Llegada a Europa -> cliente: camión refrigerado por autopista
    "arrival_dest": dict(circuity=1.25, speed_kmh=65.0, eur_box_km=0.0030, kg_co2_tkm=0.09, vase_loss=0.40, fixed_h=2.0),
}

# Tramo principal (lane) por modo
MODE = {
    # Avión: rápido y caro; la cadena de frío se rompe en plataforma y bodega
    "air": dict(eur_box_fixed=6.0, eur_box_km=0.00237, kg_co2_tkm=0.60, vase_loss=0.90, speed_kmh=820.0, handling_h=15.0),
    # Barco: contenedor refrigerado con atmósfera controlada, lento y barato
    "sea": dict(eur_box_fixed=1.6, eur_box_km=0.00042, kg_co2_tkm=0.025, vase_loss=0.20, speed_kmh=26.0, handling_h=60.0),
}

HUB_HANDLING_EUR_BOX = 0.30   # preenfriado y manejo en el centro de acopio
HUB_DWELL_H = 6.0             # horas que la caja pasa en el centro de acopio
HUB_DWELL_VASE_LOSS = 0.20    # días de vida perdidos en el acopio (incluye preenfriado)

# ----------------------------------------------------------------------------
# Rutas marítimas aproximadas (lat, lon), una por lane, para dibujar el barco y medir la distancia
# ----------------------------------------------------------------------------
_FROM_CTG = [(10.40, -75.53), (12.80, -74.20)]
_FROM_SMR = [(11.25, -74.21), (13.20, -73.20)]
_CARIBBEAN = [(15.80, -70.60), (18.40, -67.70), (23.00, -62.50), (28.00, -57.00), (33.00, -49.50)]
_NORTH_ATLANTIC = [(37.00, -42.00), (40.00, -35.00), (43.50, -25.00), (46.00, -17.00), (48.20, -9.00), (49.40, -5.00)]
_TO_RTM = [(50.10, -2.00), (51.20, 1.90), (51.95, 4.05)]
_TO_LEH = [(49.90, -2.50), (49.75, -0.50), (49.48, 0.11)]
# Estrecho de Gibraltar y Mediterráneo occidental
_TO_GIBRALTAR = [(35.00, -35.00), (36.00, -20.00), (36.00, -9.50), (35.95, -5.60), (36.30, -3.50), (37.00, -1.00)]
_TO_FOS = [(37.80, 0.30), (38.60, 0.90), (40.00, 2.00), (41.60, 3.40), (42.70, 4.00), (43.25, 4.85), (43.42, 4.88)]
_TO_GOA = [(38.20, 2.50), (39.50, 5.50), (41.50, 7.50), (43.30, 8.60), (44.40, 8.90)]

SEA_WAYPOINTS = {
    "SEA_CTG_RTM": _FROM_CTG + _CARIBBEAN + _NORTH_ATLANTIC + _TO_RTM,
    "SEA_SMR_RTM": _FROM_SMR + _CARIBBEAN + _NORTH_ATLANTIC + _TO_RTM,
    "SEA_CTG_LEH": _FROM_CTG + _CARIBBEAN + _NORTH_ATLANTIC + _TO_LEH,
    "SEA_SMR_FOS": _FROM_SMR + _CARIBBEAN + _TO_GIBRALTAR + _TO_FOS,
    "SEA_SMR_GOA": _FROM_SMR + _CARIBBEAN + _TO_GIBRALTAR + _TO_GOA,
}


# ----------------------------------------------------------------------------
# Tramo europeo: de la llegada (aeropuerto o puerto) hasta el cliente
# ----------------------------------------------------------------------------
# Después del avión las flores viajan en estibas dentro de un camión refrigerado ("pallet").
# Después del barco siguen dentro del mismo contenedor refrigerado con atmósfera controlada,
# sellado hasta el cliente ("container"): por eso pierden menos vida en florero por día.
VASE_RATE = {"pallet": 0.40, "container": 0.20, "flight": 0.90}

# speed_kmh = velocidad media, circuity = factor sobre la línea recta (o sobre la ruta dibujada),
# eur_box_km = costo por caja y km, eur_box_fixed = manejo en la terminal por caja,
# kg_co2_tkm = emisiones por tonelada-km, fixed_h = horas de terminal o transbordo
EU_MODE = {
    # Camión refrigerado. Un contenedor completo por camión cuesta más por caja que estibas consolidadas.
    "truck": dict(label="Reefer truck", speed_kmh=62.0, circuity=1.25, eur_box_km={"pallet": 0.0030, "container": 0.0044},
                  eur_box_fixed=0.0, kg_co2_tkm=0.090, fixed_h=2.0),
    # Tren de contenedores: barato y limpio, con espera en la terminal
    "rail": dict(label="Rail", speed_kmh=48.0, circuity=1.15, eur_box_km=0.0015, eur_box_fixed=0.60, kg_co2_tkm=0.025, fixed_h=8.0),
    # Barcaza fluvial (Rin, Sena, Ródano): muy barata y lenta, sobre todo río arriba
    "barge": dict(label="River barge", speed_kmh=10.0, circuity=1.0, eur_box_km=0.0011, eur_box_fixed=0.50, kg_co2_tkm=0.035, fixed_h=12.0),
    # Buque feeder: barco pequeño que lleva contenedores de Rotterdam a puertos del norte
    "feeder": dict(label="Feeder ship", speed_kmh=28.0, circuity=1.0, eur_box_km=0.0010, eur_box_fixed=0.70, kg_co2_tkm=0.030, fixed_h=18.0),
    # Ferry ro-ro: el remolque refrigerado cruza el Báltico a bordo
    "ferry": dict(label="Ro-ro ferry", speed_kmh=38.0, circuity=1.0, eur_box_km=0.0018, eur_box_fixed=0.40, kg_co2_tkm=0.060, fixed_h=4.0),
    # Vuelo de conexión dentro de Europa (transbordo en Schiphol)
    "flight": dict(label="Connecting flight", speed_kmh=700.0, circuity=1.0, eur_box_km=0.0045, eur_box_fixed=3.00, kg_co2_tkm=1.100, fixed_h=10.0),
}

# Rutas dibujadas de los tramos europeos que no van en línea recta (lat, lon)
EU_PATHS = {
    # Barcazas
    ("RTM", "ALP"): [(51.95, 4.05), (51.90, 4.30), (51.91, 4.50), (51.95, 4.60), (52.01, 4.71), (52.07, 4.70), (52.13, 4.66)],
    ("RTM", "BSL"): [(51.95, 4.05), (51.90, 4.40), (51.82, 4.75), (51.83, 5.10), (51.85, 5.87), (51.83, 6.25), (51.43, 6.76),
                     (51.23, 6.77), (50.94, 6.96), (50.73, 7.10), (50.36, 7.60), (50.15, 7.70), (49.97, 7.90), (50.00, 8.27),
                     (49.49, 8.46), (49.03, 8.31), (48.58, 7.80), (48.00, 7.58), (47.59, 7.59)],
    ("LEH", "GEN"): [(49.48, 0.11), (49.43, 0.50), (49.40, 0.80), (49.44, 1.09), (49.30, 1.25), (49.22, 1.40), (49.08, 1.55),
                     (49.00, 1.85), (48.98, 2.10), (48.93, 2.28)],
    ("FOS", "LYP"): [(43.42, 4.88), (43.68, 4.63), (43.95, 4.80), (44.25, 4.65), (44.55, 4.75), (44.93, 4.89), (45.25, 4.80),
                     (45.52, 4.85), (45.73, 4.82)],
    # Buques feeder y ferry
    ("RTM", "GOT"): [(51.95, 4.05), (52.60, 4.00), (54.00, 5.50), (55.80, 7.00), (57.30, 8.00), (57.90, 10.00), (57.95, 10.75),
                     (57.70, 11.60), (57.69, 11.83)],
    ("RTM", "VUO"): [(51.95, 4.05), (52.60, 4.00), (53.60, 6.00), (53.90, 8.00), (53.89, 9.14), (54.20, 9.60), (54.36, 10.15),
                     (54.55, 10.60), (54.58, 11.25), (54.60, 12.50), (55.25, 14.30), (55.90, 16.80), (57.50, 19.60),
                     (58.80, 21.00), (59.75, 24.00), (60.10, 25.00), (60.21, 25.19)],
    ("TRV", "VUO"): [(53.96, 10.87), (54.20, 11.10), (54.45, 11.60), (54.60, 12.50), (55.25, 14.30), (55.90, 16.80),
                     (57.50, 19.60), (58.80, 21.00), (59.75, 24.00), (60.10, 25.00), (60.21, 25.19)],
    # Tren por el túnel de base del Gotardo y por el valle del Ródano
    ("GOA", "ZUR"): [(44.41, 8.93), (44.91, 8.61), (45.48, 9.20), (45.83, 9.03), (46.19, 9.02), (46.60, 8.75), (46.82, 8.65),
                     (47.05, 8.55), (47.40, 8.48)],
    ("GOA", "BAS"): [(44.41, 8.93), (44.91, 8.61), (45.48, 9.20), (45.83, 9.03), (46.19, 9.02), (46.60, 8.75), (46.82, 8.65),
                     (47.05, 8.55), (47.35, 7.90), (47.56, 7.60)],
    ("FOS", "GVA"): [(43.42, 4.88), (43.95, 4.80), (44.93, 4.89), (45.75, 4.85), (45.96, 5.36), (46.11, 5.83), (46.20, 6.14)],
    ("FOS", "LYS"): [(43.42, 4.88), (43.95, 4.80), (44.93, 4.89), (45.66, 4.90)],
    # Camiones por los puentes daneses hacia Suecia
    ("AMS", "STO"): [(52.31, 4.77), (53.55, 10.00), (55.68, 12.57), (55.60, 13.00), (59.30, 18.05)],
    ("AMS", "GBG"): [(52.31, 4.77), (53.55, 10.00), (55.68, 12.57), (55.60, 13.00), (57.71, 11.97)],
    ("AMS", "MMA"): [(52.31, 4.77), (53.55, 10.00), (55.68, 12.57), (55.60, 13.00)],
    ("RTM", "STO"): [(51.95, 4.05), (53.55, 10.00), (55.68, 12.57), (55.60, 13.00), (59.30, 18.05)],
    ("RTM", "GBG"): [(51.95, 4.05), (53.55, 10.00), (55.68, 12.57), (55.60, 13.00), (57.71, 11.97)],
    ("RTM", "MMA"): [(51.95, 4.05), (53.55, 10.00), (55.68, 12.57), (55.60, 13.00)],
}


# ----------------------------------------------------------------------------
# Distancias
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en línea recta sobre la esfera (km). Acepta escalares o arreglos numpy."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def distance_matrix_km(lats_a, lons_a, lats_b=None, lons_b=None):
    """Matriz de distancias Haversine entre dos conjuntos de puntos (o de un conjunto consigo mismo)."""
    lats_a, lons_a = np.asarray(lats_a, float), np.asarray(lons_a, float)
    if lats_b is None:
        lats_b, lons_b = lats_a, lons_a
    lats_b, lons_b = np.asarray(lats_b, float), np.asarray(lons_b, float)
    return haversine_km(lats_a[:, None], lons_a[:, None], lats_b[None, :], lons_b[None, :])


def path_length_km(points):
    """Longitud de una ruta definida por puntos (lat, lon)."""
    return float(sum(haversine_km(a[0], a[1], b[0], b[1]) for a, b in zip(points[:-1], points[1:])))


def great_circle_points(lat1, lon1, lat2, lon2, n=40):
    """Puntos intermedios de un círculo máximo (para dibujar y animar el vuelo)."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d = 2 * math.asin(math.sqrt(math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2))
    pts = []
    for f in np.linspace(0, 1, n):
        a = math.sin((1 - f) * d) / math.sin(d)
        b = math.sin(f * d) / math.sin(d)
        x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
        y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
        z = a * math.sin(p1) + b * math.sin(p2)
        pts.append((math.degrees(math.atan2(z, math.sqrt(x * x + y * y))), math.degrees(math.atan2(y, x))))
    return pts


def interpolate_path(points, n=60):
    """Reparte n puntos a distancias iguales a lo largo de una ruta (para animar el barco)."""
    seg = [haversine_km(a[0], a[1], b[0], b[1]) for a, b in zip(points[:-1], points[1:])]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for s in np.linspace(0, cum[-1], n):
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
        f = 0.0 if seg[i] == 0 else (s - cum[i]) / seg[i]
        out.append((points[i][0] + f * (points[i + 1][0] - points[i][0]),
                    points[i][1] + f * (points[i + 1][1] - points[i][1])))
    return out


# ----------------------------------------------------------------------------
# Atributos de cada arco (costo, tiempo, CO2, pérdida de vida en florero)
# ----------------------------------------------------------------------------
def road_arc(straight_km: float, leg: str) -> dict:
    """Arco por carretera. Parte de la distancia Haversine y aplica el factor de carretera."""
    p = LEG[leg]
    km = straight_km * p["circuity"]
    time_h = km / p["speed_kmh"] + p["fixed_h"]
    return dict(
        distance_km=round(km, 1),
        cost_per_box=round(km * p["eur_box_km"], 3),
        time_h=round(time_h, 2),
        co2_kg_per_box=round(km * BOX_WEIGHT_T * p["kg_co2_tkm"], 3),
        vase_loss_per_day=p["vase_loss"],
    )


def lane_arc(distance_km: float, mode: str) -> dict:
    """Tramo principal: vuelo o travesía marítima, con manejo en origen y destino."""
    p = MODE[mode]
    time_h = distance_km / p["speed_kmh"] + p["handling_h"]
    return dict(
        distance_km=round(distance_km, 0),
        cost_per_box=round(p["eur_box_fixed"] + distance_km * p["eur_box_km"], 2),
        time_h=round(time_h, 1),
        co2_kg_per_box=round(distance_km * BOX_WEIGHT_T * p["kg_co2_tkm"], 2),
        vase_loss_per_day=p["vase_loss"],
    )


def vase_loss_days(time_h, vase_loss_per_day):
    """Días de vida en florero perdidos en un arco = (horas / 24) x tasa de pérdida por día."""
    return time_h / 24.0 * vase_loss_per_day


def eu_segment(a: tuple, b: tuple, mode: str, unit: str, path: list | None = None) -> dict:
    """
    Un tramo europeo entre dos nodos a=(id, lat, lon) y b=(id, lat, lon).

    Si hay una ruta dibujada (río, mar, puentes), la distancia es su longitud x circuity;
    si no, es la línea recta x circuity. unit = "pallet" (venía en avión) o "container" (venía en barco).
    """
    p = EU_MODE[mode]
    if path is None:
        path = [(a[1], a[2]), (b[1], b[2])]
        km = float(haversine_km(a[1], a[2], b[1], b[2])) * p["circuity"]
    else:
        km = path_length_km(path) * (1.10 if mode == "truck" else p["circuity"])
    rate = p["eur_box_km"][unit] if isinstance(p["eur_box_km"], dict) else p["eur_box_km"]
    time_h = km / p["speed_kmh"] + p["fixed_h"]
    vase_rate = VASE_RATE["flight"] if mode == "flight" else VASE_RATE[unit]
    return dict(
        origin=a[0], dest=b[0], mode=mode, distance_km=round(km, 1),
        cost_per_box=round(p["eur_box_fixed"] + km * rate, 3), time_h=round(time_h, 2),
        co2_kg_per_box=round(km * BOX_WEIGHT_T * p["kg_co2_tkm"], 3),
        vase_loss=round(vase_loss_days(time_h, vase_rate), 4),
        path=[[round(lo, 4), round(la, 4)] for la, lo in path],   # [lon, lat] para pydeck
    )
