from __future__ import annotations

from html import escape

from grasp_agent_middleware.schemas import BoundingBox, CoordinateSpace, ObjectTable


COLOR_MAP = {
    "red": "#d94841",
    "blue": "#3371c2",
    "green": "#2f8a57",
    "yellow": "#d0a12e",
    "white": "#f7f7f7",
    "black": "#20242c",
    "orange": "#df7d32",
    "purple": "#7c5cc4",
    "gray": "#8b95a5",
    "grey": "#8b95a5",
}


def object_table_overlay_svg(
    object_table: ObjectTable,
    *,
    width: int = 960,
    height: int = 540,
    image_href: str | None = None,
) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Object table overlay">',
        "<defs>",
        "<pattern id=\"grid\" width=\"32\" height=\"32\" patternUnits=\"userSpaceOnUse\">",
        "<path d=\"M 32 0 L 0 0 0 32\" fill=\"none\" stroke=\"#d8dee8\" stroke-width=\"1\" opacity=\"0.45\"/>",
        "</pattern>",
        "</defs>",
    ]
    if image_href:
        parts.append(
            f'<image href="{escape(image_href)}" x="0" y="0" width="{width}" height="{height}" '
            'preserveAspectRatio="xMidYMid meet" opacity="0.72"/>'
        )
        parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="rgba(255,255,255,0.10)"/>')
    else:
        parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>')
        parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#grid)"/>')
        parts.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            'font-family="Arial" font-size="20" fill="#8b95a5">visual workspace</text>'
        )

    for obj in object_table.objects:
        if obj.bbox is None:
            continue
        x, y, w, h = _bbox_to_pixels(obj.bbox, width, height)
        color = COLOR_MAP.get((obj.attributes.color or "").lower(), "#176b87")
        label = escape(f"{obj.object_id} | {obj.label} | {obj.confidence:.2f}")
        fill = color if color != "#f7f7f7" else "#d8dee8"
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'fill="{fill}" fill-opacity="0.10" stroke="{color}" stroke-width="4" rx="6"/>',
                f'<rect x="{x:.1f}" y="{max(y - 30, 0):.1f}" width="{min(max(len(label) * 8.5, 170), width - x):.1f}" '
                f'height="26" fill="{color}" rx="5"/>',
                f'<text x="{x + 9:.1f}" y="{max(y - 12, 18):.1f}" font-family="Arial" '
                f'font-size="14" font-weight="700" fill="white">{label}</text>',
            ]
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _bbox_to_pixels(bbox: BoundingBox, width: int, height: int) -> tuple[float, float, float, float]:
    if bbox.coordinate_space == CoordinateSpace.pixel:
        x1, y1, x2, y2 = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
    else:
        x1, y1, x2, y2 = (
            bbox.x_min * width,
            bbox.y_min * height,
            bbox.x_max * width,
            bbox.y_max * height,
        )
    return x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)

