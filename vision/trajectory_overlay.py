from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OverlayStyle:
    ref_color: tuple = (20, 109, 91)
    ref_thickness: int = 2
    trace_color: tuple = (0, 0, 255)
    trace_thickness: int = 1


def blend_masked_overlay(frame_bgr, overlay_bgr, overlay_mask, alpha=1.0):
    out = frame_bgr.copy()
    if overlay_bgr is None or overlay_mask is None:
        return out
    mask = overlay_mask > 0
    if not np.any(mask):
        return out
    base = out[mask].astype(np.float32)
    over = overlay_bgr[mask].astype(np.float32)
    mixed = (1.0 - float(alpha)) * base + float(alpha) * over
    out[mask] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def append_trace_segment(trace_overlay_bgr, trace_overlay_mask, p0_px, p1_px, color, thickness=1):
    p0 = (int(p0_px[0]), int(p0_px[1]))
    p1 = (int(p1_px[0]), int(p1_px[1]))
    cv2.line(trace_overlay_bgr, p0, p1, color, thickness=thickness)
    cv2.line(trace_overlay_mask, p0, p1, 255, thickness=max(1, thickness))
