"""
Agrupación de fincas con k-medoids (PAM) programado a mano.

¿Por qué k-medoids y no k-means?
- El centro de un grupo de k-means es un promedio: puede caer en un potrero o en un cerro.
- El medoide de k-medoids es una finca que existe, así que sirve directamente como
  sitio candidato para un centro de acopio (es el mismo criterio de las p-medianas).

Igual que en mi proyecto de clase (Yopal y Villavicencio por separado), cada región se
agrupa por separado y el número de grupos k se escoge con el coeficiente de silueta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from optimization.network import distance_matrix_km, haversine_km


# ----------------------------------------------------------------------------
# PAM: Partitioning Around Medoids
# ----------------------------------------------------------------------------
def pam(dist: np.ndarray, k: int, max_iter: int = 100) -> tuple[list[int], np.ndarray, float]:
    """
    k-medoids clásico sobre una matriz de distancias.

    Fase BUILD: escoge los medoides iniciales uno por uno (codicioso, como el ADD de clase):
        el primero es el punto con menor suma de distancias a todos; cada uno de los
        siguientes es el que más reduce el costo total.
    Fase SWAP: prueba cambiar cada medoide por cada punto que no lo es y aplica el mejor
        cambio mientras el costo baje (búsqueda local, la misma idea de 2-opt).

    Devuelve (índices de los medoides, etiqueta de cada punto, costo total).
    """
    n = dist.shape[0]
    k = min(k, n)

    # ---- BUILD ----
    medoids = [int(np.argmin(dist.sum(axis=1)))]
    nearest = dist[:, medoids[0]].copy()          # distancia de cada punto a su medoide más cercano
    while len(medoids) < k:
        best_gain, best_c = -1.0, None
        for c in range(n):
            if c in medoids:
                continue
            gain = np.maximum(nearest - dist[:, c], 0).sum()   # cuánto baja el costo si c es medoide
            if gain > best_gain:
                best_gain, best_c = gain, c
        medoids.append(int(best_c))
        nearest = np.minimum(nearest, dist[:, best_c])

    # ---- SWAP ----
    cost = float(dist[:, medoids].min(axis=1).sum())
    for _ in range(max_iter):
        best_cost, best_swap = cost, None
        for i in range(k):
            for c in range(n):
                if c in medoids:
                    continue
                trial = medoids.copy()
                trial[i] = c
                trial_cost = float(dist[:, trial].min(axis=1).sum())
                if trial_cost < best_cost - 1e-9:
                    best_cost, best_swap = trial_cost, (i, c)
        if best_swap is None:
            break                                   # ningún cambio mejora: óptimo local
        medoids[best_swap[0]] = best_swap[1]
        cost = best_cost

    labels = np.argmin(dist[:, medoids], axis=1)
    return medoids, labels, cost


# ----------------------------------------------------------------------------
# Elegir k por silueta y comparar con k-means
# ----------------------------------------------------------------------------
def _to_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Proyección plana sencilla (km) para poder correr k-means en una zona pequeña."""
    lat0 = lat.mean()
    return np.column_stack([(lon - lon.mean()) * 111.32 * np.cos(np.radians(lat0)), (lat - lat0) * 110.57])


def compare_k(lat, lon, k_values=range(2, 8), seed: int = 42) -> pd.DataFrame:
    """Para cada k: silueta y costo de k-medoids (PAM) y de k-means, con la misma matriz de distancias."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dist = distance_matrix_km(lat, lon)
    xy = _to_km(lat, lon)
    rows = []
    for k in k_values:
        if k >= len(lat):
            break
        medoids, lab_pam, cost_pam = pam(dist, k)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xy)
        # Distancia de cada centroide de k-means a la finca más cercana: qué tan "en el aire" queda
        cen_gap = [float(np.min(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]))) for c in km.cluster_centers_]
        rows.append(dict(
            k=k,
            silhouette_pam=float(silhouette_score(dist, lab_pam, metric="precomputed")),
            silhouette_kmeans=float(silhouette_score(dist, km.labels_, metric="precomputed")),
            avg_km_to_medoid=cost_pam / len(lat),
            avg_km_to_centroid=float(np.hypot(*(xy - km.cluster_centers_[km.labels_]).T).mean()),
            centroid_to_nearest_farm_km=float(np.mean(cen_gap)),
        ))
    return pd.DataFrame(rows)


def cluster_farms(farms: pd.DataFrame, k_values=range(2, 8)) -> dict:
    """
    Agrupa las fincas de cada región con PAM y escoge k por silueta.

    Devuelve un diccionario con:
        farms    -> las fincas con columnas 'cluster' (id global) e 'is_medoid'
        medoids  -> una fila por medoide (la finca real que representa al grupo)
        scores   -> tabla de silueta por región y k, PAM contra k-means
        best_k   -> k escogido en cada región
    """
    farms = farms.copy()
    farms["cluster"] = -1
    farms["is_medoid"] = False
    scores, medoid_rows, best_k, offset = [], [], {}, 0

    for region, grp in farms.groupby("region", sort=False):
        lat, lon = grp.lat.to_numpy(), grp.lon.to_numpy()
        table = compare_k(lat, lon, k_values)
        table.insert(0, "region", region)
        scores.append(table)
        k = int(table.loc[table.silhouette_pam.idxmax(), "k"])   # mayor silueta
        best_k[region] = k

        dist = distance_matrix_km(lat, lon)
        medoids, labels, _ = pam(dist, k)
        farms.loc[grp.index, "cluster"] = labels + offset
        farms.loc[grp.index[medoids], "is_medoid"] = True
        for j, m in enumerate(medoids):
            f = grp.iloc[m]
            members = grp.iloc[np.where(labels == j)[0]]
            medoid_rows.append(dict(
                cluster=j + offset, farm_id=f.farm_id, name=f["name"], municipality=f.municipality, region=region,
                lat=f.lat, lon=f.lon, n_farms=len(members),
                boxes=int(members[[c for c in farms.columns if c.startswith("boxes_")]].sum().sum()),
                max_km=float(dist[np.where(labels == j)[0], m].max()),
            ))
        offset += k

    return dict(farms=farms, medoids=pd.DataFrame(medoid_rows), scores=pd.concat(scores, ignore_index=True), best_k=best_k)


def medoids_as_hub_candidates(medoids: pd.DataFrame, hubs: pd.DataFrame, min_gap_km: float = 3.0) -> pd.DataFrame:
    """
    Convierte los medoides en sitios candidatos adicionales para centro de acopio.

    Un medoide es una finca real, así que el costo fijo se toma como el promedio de los
    sitios de su región más un 10% (hay que adecuar un cuarto frío en la finca) y la
    capacidad como el promedio de la región. Si ya hay un sitio a menos de min_gap_km,
    no se duplica.
    """
    rows = []
    for _, m in medoids.iterrows():
        near = haversine_km(m.lat, m.lon, hubs.lat.to_numpy(), hubs.lon.to_numpy())
        if len(near) and near.min() < min_gap_km:
            continue
        reg = hubs[hubs.region == m.region]
        rows.append(dict(
            hub_id=f"M{int(m.cluster) + 1}", name=f"Medoid site – {m['name']} ({m.municipality})", region=m.region,
            lat=m.lat, lon=m.lon,
            fixed_cost_week=int(round(reg.fixed_cost_week.mean() * 1.10, -2)),
            capacity_boxes=int(round(reg.capacity_boxes.mean(), -2)),
            source="k-medoids",
        ))
    return pd.DataFrame(rows)
