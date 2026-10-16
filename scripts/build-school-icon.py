"""Render the school SVG mark into a multi-resolution Windows ICO."""

from __future__ import annotations

import binascii
import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "school-management" / "assets" / "school-mark.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)
SCALE = 4
COLORS = {
    "ink": (24, 58, 77, 255),
    "paper": (244, 242, 233, 255),
    "clay": (197, 123, 83, 255),
    "gold": (233, 183, 123, 255),
}


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body))


def _png(size: int, pixels: bytes) -> bytes:
    rows = b"".join(
        b"\0" + pixels[row * size * 4 : (row + 1) * size * 4] for row in range(size)
    )
    header = struct.pack(">2I5B", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _paint(size: int) -> bytes:
    scale = SCALE
    side = size * scale
    pixels = bytearray(side * side * 4)

    def fill_row(y: int, left: int, right: int, color: tuple[int, ...]) -> None:
        if left >= right:
            return
        start = (y * side + left) * 4
        end = (y * side + right) * 4
        pixels[start:end] = bytes(color) * (right - left)

    def rounded_background() -> None:
        radius = 56 * size / 256 * scale
        for y in range(side):
            if y < radius:
                dy = radius - y - 0.5
            elif y >= side - radius:
                dy = y - (side - radius) + 0.5
            else:
                dy = 0
            inset = radius - math.sqrt(max(0.0, radius * radius - dy * dy)) if dy else 0
            left = max(0, math.ceil(inset))
            right = min(side, math.floor(side - inset))
            fill_row(y, left, right, COLORS["ink"])

    def rectangle(
        x1: float, y1: float, x2: float, y2: float, color: tuple[int, ...]
    ) -> None:
        left = max(0, round(x1 * size / 256 * scale))
        top = max(0, round(y1 * size / 256 * scale))
        right = min(side, round(x2 * size / 256 * scale))
        bottom = min(side, round(y2 * size / 256 * scale))
        for y in range(top, bottom):
            fill_row(y, left, right, color)

    def polygon(points: tuple[tuple[int, int], ...], color: tuple[int, ...]) -> None:
        scaled = [(x * size / 256 * scale, y * size / 256 * scale) for x, y in points]
        top = max(0, math.floor(min(y for _x, y in scaled)))
        bottom = min(side, math.ceil(max(y for _x, y in scaled)))
        for y in range(top, bottom):
            scan_y = y + 0.5
            crossings = []
            for index, (x1, y1) in enumerate(scaled):
                x2, y2 = scaled[(index + 1) % len(scaled)]
                if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
                    crossings.append(x1 + (scan_y - y1) * (x2 - x1) / (y2 - y1))
            crossings.sort()
            for index in range(0, len(crossings), 2):
                left = max(0, math.ceil(crossings[index] - 0.5))
                right = min(side, math.ceil(crossings[index + 1] - 0.5))
                fill_row(y, left, right, color)

    def round_line(
        x1: int, y1: int, x2: int, y2: int, width: int, color: tuple[int, ...]
    ) -> None:
        radius = width * size / 256 * scale / 2
        ax, ay = x1 * size / 256 * scale, y1 * size / 256 * scale
        bx, by = x2 * size / 256 * scale, y2 * size / 256 * scale
        left = max(0, math.floor(min(ax, bx) - radius))
        right = min(side, math.ceil(max(ax, bx) + radius))
        top = max(0, math.floor(min(ay, by) - radius))
        bottom = min(side, math.ceil(max(ay, by) + radius))
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        for y in range(top, bottom):
            for x in range(left, right):
                px, py = x + 0.5, y + 0.5
                projection = (
                    max(
                        0.0,
                        min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared),
                    )
                    if length_squared
                    else 0.0
                )
                distance = math.hypot(
                    px - (ax + projection * dx), py - (ay + projection * dy)
                )
                if distance <= radius:
                    offset = (y * side + x) * 4
                    pixels[offset : offset + 4] = bytes(color)

    rounded_background()
    polygon(
        (
            (37, 111),
            (128, 39),
            (219, 111),
            (201, 111),
            (201, 206),
            (55, 206),
            (55, 111),
        ),
        COLORS["paper"],
    )
    polygon(
        ((35, 111), (128, 37), (221, 111), (209, 126), (128, 62), (47, 126)),
        COLORS["clay"],
    )
    rectangle(67, 117, 189, 208, COLORS["paper"])
    for left in (80, 117, 154):
        rectangle(left, 132, left + 22, 156, COLORS["ink"])
    rectangle(111, 169, 145, 208, COLORS["clay"])
    round_line(57, 215, 199, 215, 9, COLORS["gold"])
    round_line(91, 229, 165, 229, 7, COLORS["paper"])

    output = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            channels = [0, 0, 0, 0]
            for sub_y in range(scale):
                for sub_x in range(scale):
                    offset = ((y * scale + sub_y) * side + x * scale + sub_x) * 4
                    for channel in range(4):
                        channels[channel] += pixels[offset + channel]
            target = (y * size + x) * 4
            output[target : target + 4] = bytes(
                value // (scale * scale) for value in channels
            )
    return bytes(output)


def main() -> None:
    frames = [(size, _png(size, _paint(size))) for size in SIZES]
    offset = 6 + len(frames) * 16
    directory = bytearray(struct.pack("<HHH", 0, 1, len(frames)))
    payloads = bytearray()
    for size, png in frames:
        directory.extend(
            struct.pack(
                "<BBBBHHII",
                size if size < 256 else 0,
                size if size < 256 else 0,
                0,
                0,
                1,
                32,
                len(png),
                offset,
            )
        )
        payloads.extend(png)
        offset += len(png)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    ASSET.write_bytes(directory + payloads)
    print(f"Created {ASSET} ({len(frames)} sizes)")


if __name__ == "__main__":
    main()
