"""Dibuja los íconos de almacén que usa el mapa (se corre una sola vez: python assets/make_icons.py)."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent


def warehouse(fill, outline, door, name):
    s = 4                                   # se dibuja 4 veces más grande y luego se reduce (bordes suaves)
    img = Image.new("RGBA", (128 * s, 128 * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    p = lambda pts: [(x * s, y * s) for x, y in pts]
    # Techo y cuerpo del almacén
    d.polygon(p([(64, 10), (120, 48), (120, 118), (8, 118), (8, 48)]), fill=fill, outline=outline, width=5 * s)
    # Puerta enrollable con tres líneas
    d.rectangle(p([(36, 66), (92, 118)]), fill=door)
    for y in (78, 91, 104):
        d.line(p([(36, y), (92, y)]), fill=fill, width=3 * s)
    img.resize((128, 128), Image.LANCZOS).save(OUT / name)


if __name__ == "__main__":
    warehouse((245, 239, 236, 255), (14, 16, 20, 255), (213, 81, 129, 255), "warehouse_open.png")     # centro abierto
    warehouse((90, 94, 102, 235), (14, 16, 20, 255), (40, 43, 50, 255), "warehouse_idle.png")         # candidato sin abrir
    print("iconos creados")
