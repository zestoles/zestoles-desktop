"""Draw the JARVIS icon, with nothing but the standard library.

A desktop application that inherits the generic script icon looks like a script.
Pillow would draw this in four lines and would also be the first third-party
dependency in a project that has none, on a machine where "pip install" is not
part of opening JARVIS. So the PNG is assembled by hand -- zlib is in the
standard library, and a PNG is a handful of length-prefixed chunks.

The image is the core from the interface: a blue sphere with a soft halo on a
dark ground, drawn at four sizes so Windows has a real one to pick for the
taskbar, the desktop and the alt-tab list instead of scaling one.

Run once (or after changing the colours):

    python tools/launcher/make-icon.py
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "ui" / "jarvis.ico"

SIZES = (16, 32, 48, 256)

BACKGROUND = (7, 9, 13)
HALO = (29, 106, 153)
CORE_EDGE = (26, 110, 163)
CORE_MID = (77, 184, 255)
CORE_LIGHT = (191, 230, 255)


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def pixels(size: int) -> bytes:
    """One RGBA image, drawn analytically so it stays crisp at every size."""
    centre = (size - 1) / 2
    radius = size * 0.34
    halo_radius = size * 0.48
    # Light source up and to the left, the same place the CSS gradient puts it.
    light = (centre - radius * 0.36, centre - radius * 0.40)

    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            dx, dy = x - centre, y - centre
            distance = math.hypot(dx, dy)

            if distance <= radius:
                # Distance from the highlight, normalised over the sphere.
                lit = math.hypot(x - light[0], y - light[1]) / (radius * 1.9)
                colour = mix(CORE_LIGHT, CORE_MID, lit * 1.5)
                colour = mix(colour, CORE_EDGE, max(0.0, (distance / radius - 0.55) * 2.2))
                alpha = 255
                edge = radius - distance
                if edge < 1.0:
                    alpha = round(255 * max(0.0, edge))
            elif distance <= halo_radius:
                # The ring, fading out. Keeps the icon readable on a light
                # taskbar without painting a hard square of background.
                falloff = 1.0 - (distance - radius) / (halo_radius - radius)
                colour = mix(BACKGROUND, HALO, falloff * 0.9)
                alpha = round(210 * falloff ** 1.4)
            else:
                colour = BACKGROUND
                alpha = 0

            row += bytes((*colour, alpha))
        rows.append(bytes(row))
    return b"".join(b"\x00" + row for row in rows)


def png(size: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(pixels(size), 9))
            + chunk(b"IEND", b""))


def ico(images: dict[int, bytes]) -> bytes:
    """Wrap PNGs in an ICO directory. Windows has read PNG-in-ICO since Vista."""
    count = len(images)
    out = struct.pack("<HHH", 0, 1, count)
    offset = 6 + count * 16
    entries, payloads = [], []
    for size, data in sorted(images.items()):
        # 0 means 256 in the directory; anything larger will not fit the byte.
        dimension = 0 if size >= 256 else size
        entries.append(struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32,
                                   len(data), offset))
        payloads.append(data)
        offset += len(data)
    return out + b"".join(entries) + b"".join(payloads)


def main() -> int:
    images = {size: png(size) for size in SIZES}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(ico(images))
    print(f"yazildi: {TARGET} ({TARGET.stat().st_size} bayt, "
          f"{len(SIZES)} boyut: {', '.join(str(s) for s in SIZES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
