"""
Paleta, estilos CSS y piezas de interfaz que se repiten en todas las páginas.

Regla de color: cada color tiene un solo trabajo en toda la app.
    naranja = avión, azul = barco, rojo = alerta o cierre, y un color fijo por tipo de flor.
"""
from __future__ import annotations

import streamlit as st

# ---- Superficies y texto (tema oscuro) ----
PAGE = "#0e1014"
SURFACE = "#171a21"
BORDER = "rgba(255,255,255,0.10)"
INK = "#f0efec"
INK2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

# ---- Colores con significado ----
AIR = "#d95926"       # avión
SEA = "#3987e5"       # barco
TRUCK = "#898781"     # camión (neutro)
BRAND = "#d55181"     # rosa de la marca
MODE_COLOR = {"air": AIR, "sea": SEA}
MODE_LABEL = {"air": "Air", "sea": "Sea"}
FLOWER_COLOR = {"rose": "#d55181", "carnation": "#9085e9", "chrysanthemum": "#c98500"}
FLOWER_LABEL = {"rose": "Rose", "carnation": "Carnation", "chrysanthemum": "Chrysanthemum"}

# Estados (reservados: nunca se usan para una serie)
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# Paleta categórica en orden fijo (para grupos de fincas, destinos, etc.)
CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# Tramo europeo: un color por modo. La familia del barco (azules) y la del avión (naranjas) se mantienen.
EU_MODE_COLOR = {"truck": "#a9a79e", "rail": "#199e70", "barge": "#4fb0bf", "feeder": "#8fb8f2",
                 "ferry": "#9085e9", "flight": "#f2a07b"}
EU_MODE_LABEL = {"truck": "Reefer truck", "rail": "Rail", "barge": "River barge", "feeder": "Feeder ship",
                 "ferry": "Ro-ro ferry", "flight": "Connecting flight"}
EU_MODE_ICON = {"truck": "🚚", "rail": "🚆", "barge": "⛴️", "feeder": "🚢", "ferry": "⛴️", "flight": "✈️"}


def main_eu_mode(modes: str) -> str:
    """Modo que define una cadena europea: el primero que no sea camión (o camión si solo hay camión)."""
    parts = str(modes).split("+")
    return next((m for m in parts if m != "truck"), "truck")


# Banderas en SVG (los emojis de bandera no se ven en Windows)
_FLAG_SVG = {
    "NL": '<rect width="30" height="7" fill="#AE1C28"/><rect y="7" width="30" height="6" fill="#fff"/><rect y="13" width="30" height="7" fill="#21468B"/>',
    "FR": '<rect width="10" height="20" fill="#0055A4"/><rect x="10" width="10" height="20" fill="#fff"/><rect x="20" width="10" height="20" fill="#EF4135"/>',
    "CH": '<rect width="30" height="20" fill="#DA291C"/><rect x="13" y="4" width="4" height="12" fill="#fff"/><rect x="9" y="8" width="12" height="4" fill="#fff"/>',
    "SE": '<rect width="30" height="20" fill="#006AA7"/><rect x="9" width="4" height="20" fill="#FECC00"/><rect y="8" width="30" height="4" fill="#FECC00"/>',
    "FI": '<rect width="30" height="20" fill="#fff"/><rect x="8" width="5" height="20" fill="#002F6C"/><rect y="7.5" width="30" height="5" fill="#002F6C"/>',
}


def flag(code: str, height: int = 20) -> str:
    w = int(height * 1.5)
    return (f'<svg width="{w}" height="{height}" viewBox="0 0 30 20" style="border-radius:3px;vertical-align:middle;'
            f'box-shadow:0 0 0 1px rgba(255,255,255,0.18)">{_FLAG_SVG.get(code, "")}</svg>')


# Etiquetas cortas de los nodos: se llenan desde nodes.csv al cargar los datos (loaders.load_data)
NODE_LABEL: dict[str, str] = {}
DEST_LABEL: dict[str, str] = {}


def hex_to_rgb(h: str, alpha: int | None = None) -> list[int]:
    h = h.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return rgb + [alpha] if alpha is not None else rgb


# Colores de los grupos de fincas: sin azul ni naranja, que ya significan barco y avión
CLUSTER_COLORS = ["#199e70", "#c98500", "#9085e9", "#d55181", "#4fb0bf", "#7a9a2e"]


def cluster_color(i: int) -> str:
    """Color fijo por grupo; después del sexto no se inventan colores nuevos."""
    return CLUSTER_COLORS[i] if 0 <= i < len(CLUSTER_COLORS) else MUTED


CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}

.hero {{ border: 1px solid {BORDER}; border-radius: 16px; padding: 26px 30px;
        background: linear-gradient(120deg, #1b1f2a 0%, #171a21 55%, #241a22 100%); margin-bottom: 18px; }}
.hero .eyebrow {{ color: {BRAND}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; }}
.hero h1 {{ font-size: 2.1rem; margin: 6px 0 8px 0; padding: 0; color: {INK}; }}
.hero p {{ color: {INK2}; font-size: 1.02rem; margin: 0; max-width: 980px; line-height: 1.5; }}

.kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px 12px 16px; height: 100%; min-height: 132px; }}
.kpi .label {{ color: {MUTED}; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; }}
.kpi .value {{ color: {INK}; font-size: 1.75rem; font-weight: 650; line-height: 1.25; margin-top: 4px; white-space: nowrap; }}
.kpi .value small {{ font-size: 0.95rem; color: {INK2}; font-weight: 500; }}
.kpi .sub {{ color: {INK2}; font-size: 0.82rem; margin-top: 2px; }}
.kpi .good {{ color: {GOOD}; font-weight: 600; }}
.kpi .bad {{ color: #e66767; font-weight: 600; }}
.kpi .bar {{ display: flex; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 8px; gap: 2px; }}

.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px; }}
.card h4 {{ margin: 0 0 6px 0; font-size: 1rem; color: {INK}; }}
.card p {{ color: {INK2}; font-size: 0.9rem; margin: 0; line-height: 1.45; }}
.step-num {{ display: inline-block; width: 26px; height: 26px; border-radius: 50%; background: {BRAND}; color: #fff;
            text-align: center; line-height: 26px; font-weight: 700; font-size: 0.85rem; margin-right: 8px; }}

.alert {{ border: 1px solid {CRITICAL}; background: rgba(208,59,59,0.12); border-radius: 12px; padding: 12px 16px;
         color: {INK}; margin-bottom: 14px; display: flex; gap: 12px; align-items: flex-start; }}
.alert .dot {{ width: 12px; height: 12px; border-radius: 50%; background: {CRITICAL}; margin-top: 5px; flex: none;
              animation: pulse 1.4s infinite; }}
.alert b {{ color: #ff8a8a; }}
.ok-banner {{ border: 1px solid rgba(12,163,12,0.6); background: rgba(12,163,12,0.10); border-radius: 12px;
             padding: 10px 16px; color: {INK}; margin-bottom: 14px; }}
@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(208,59,59,0.75); }} 70% {{ box-shadow: 0 0 0 12px rgba(208,59,59,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(208,59,59,0); }} }}

.chip {{ display: inline-block; border: 1px solid {BORDER}; border-radius: 999px; padding: 2px 10px; margin: 0 6px 6px 0;
        font-size: 0.8rem; color: {INK2}; background: {SURFACE}; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: {INK2}; font-size: 0.82rem; margin: 4px 0 2px 0; }}
.legend span.sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.section-title {{ font-size: 1.05rem; font-weight: 650; color: {INK}; margin: 6px 0 2px 0; }}
.section-note {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 8px; }}
div[data-testid="stButton"] button[kind="primary"] {{ font-size: 1.1rem; font-weight: 700; padding: 0.75rem 1.2rem; border-radius: 12px; }}

.market {{ background: linear-gradient(135deg, #1b1f2a 0%, {SURFACE} 60%); border: 1px solid {BORDER}; border-radius: 16px;
          padding: 18px 20px; height: 100%; }}
.market .head {{ display: flex; align-items: center; gap: 12px; }}
.market .head .name {{ font-size: 1.35rem; font-weight: 700; color: {INK}; }}
.market .role {{ color: {INK2}; font-size: 0.92rem; margin: 6px 0 12px 0; line-height: 1.45; }}
.market .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }}
.market .stat {{ background: rgba(255,255,255,0.03); border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 10px; }}
.market .stat .v {{ color: {INK}; font-size: 1.15rem; font-weight: 650; }}
.market .stat .l {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.market .sub {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; margin: 10px 0 6px 0; }}
.mode-pill {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 3px 10px 3px 8px; margin: 0 6px 6px 0;
             font-size: 0.8rem; color: {INK}; background: rgba(255,255,255,0.04); border: 1px solid {BORDER}; }}
.mode-pill i {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.chain-row {{ display: flex; justify-content: space-between; gap: 10px; color: {INK2}; font-size: 0.84rem; padding: 5px 0;
             border-bottom: 1px dashed rgba(255,255,255,0.07); }}
.chain-row b {{ color: {INK}; font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", bar: list[tuple[str, float]] | None = None) -> str:
    """Tarjeta KPI. bar = lista de (color, proporción) para una barra partida debajo del número."""
    bar_html = ""
    if bar:
        segs = "".join(f'<div style="background:{c};flex:{max(v, 0.0001)}"></div>' for c, v in bar if v > 0)
        bar_html = f'<div class="bar">{segs}</div>'
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div>{bar_html}</div>'


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>' + (f'<div class="section-note">{note}</div>' if note else ""), unsafe_allow_html=True)


def legend(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="legend">' + "".join(f'<div><span class="sw" style="background:{c}"></span>{t}</div>' for c, t in items) + "</div>", unsafe_allow_html=True)


def delta_html(pct: float, lower_is_better: bool = True, suffix: str = "") -> str:
    """Texto de variación con flecha; verde si mejora, rojo si empeora."""
    if abs(pct) < 0.05:
        return f"no change{suffix}"
    good = (pct < 0) == lower_is_better
    arrow = "▼" if pct < 0 else "▲"
    return f'<span class="{"good" if good else "bad"}">{arrow} {abs(pct):.1f}%</span>{suffix}'


def fmt_eur(v: float) -> str:
    return f"€{v / 1000:,.0f}k" if abs(v) >= 10_000 else f"€{v:,.0f}"


def style_fig(fig, height: int = 340, legend_top: bool = True):
    """Estilo común de los gráficos de plotly: fondo de la tarjeta, rejilla tenue, texto neutro."""
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=12),
        margin=dict(l=12, r=12, t=40 if legend_top else 16, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="#22262f", bordercolor=AXIS, font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    return fig


PLOTLY_CONFIG = {"displayModeBar": False}
