# -*- coding: utf-8 -*-
"""
Word-region sorting and line grouping.

Detected word-level bounding boxes are sorted top→bottom, then left→right,
and grouped into logical lines / paragraphs based on vertical proximity.
"""

import config


def group_regions_by_lines(regions, line_threshold=None):
    """
    Group word regions into lines/paragraphs based on vertical proximity.

    Args:
        regions: List of dicts, each with at least
                 ``text``, ``center_x``, ``center_y``, ``bbox``.
        line_threshold: Maximum vertical distance (px) to consider two
                        words as being on the same line.

    Returns:
        list[dict]: Each dict has ``text``, ``y_position``, ``bbox``.
    """
    if line_threshold is None:
        line_threshold = config.LINE_THRESHOLD

    if not regions:
        return []

    # Sort by y-coordinate (top to bottom)
    regions_sorted = sorted(regions, key=lambda r: r["center_y"])

    grouped_texts = []
    current_group = [regions_sorted[0]]
    current_y = regions_sorted[0]["center_y"]

    for region in regions_sorted[1:]:
        if abs(region["center_y"] - current_y) > line_threshold:
            # Finalise current group
            text = _combine_line_regions(current_group)
            if text:
                grouped_texts.append(
                    {
                        "text": text,
                        "y_position": current_y,
                        "bbox": _get_combined_bbox(current_group),
                    }
                )
            current_group = [region]
            current_y = region["center_y"]
        else:
            current_group.append(region)

    # Last group
    if current_group:
        text = _combine_line_regions(current_group)
        if text:
            grouped_texts.append(
                {
                    "text": text,
                    "y_position": current_y,
                    "bbox": _get_combined_bbox(current_group),
                }
            )

    return grouped_texts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _combine_line_regions(regions):
    """Sort words left→right and join into a single string."""
    regions_sorted = sorted(regions, key=lambda r: r["center_x"])
    texts = [r["text"] for r in regions_sorted if r["text"]]
    return " ".join(texts).strip()


def _get_combined_bbox(regions):
    """Return the enclosing bounding box for a group of regions."""
    if not regions:
        return [0, 0, 0, 0]

    x1 = min(r["bbox"][0] for r in regions)
    y1 = min(r["bbox"][1] for r in regions)
    x2 = max(r["bbox"][2] for r in regions)
    y2 = max(r["bbox"][3] for r in regions)

    return [x1, y1, x2, y2]
