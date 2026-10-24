"""Calibrated camera display transforms; never used by detection/control."""
import cv2
import numpy as np


def matched_navigation_views(top, side, *, scene, top_origin, top_px_per_cm,
                             side_detector, width, height, side_frame_shape=None):
    """Use one pixels/cm for top XY and side YZ, including their backgrounds.

    Mapping the entire annotated frame preserves image/overlay registration.
    Both views share their horizontal Y centre and direction. Raw camera and
    debug images, calibration parameters and measured positions are untouched.
    """
    values = (top_px_per_cm, side_detector.side_y_px_per_cm,
              side_detector.side_px_per_cm)
    if any(v is None or not np.isfinite(v) or v <= 0 for v in values):
        return top, side
    origin = (side_detector.y_zero_center_x_px, side_detector.zero_center_y_px)
    if any(v is None or not np.isfinite(v) for v in origin):
        return top, side
    bounds = np.asarray(scene.bounds, float)
    centre = bounds.mean(axis=0)
    extent = np.maximum(np.ptp(bounds, axis=0), .1)
    scale = min((width-48)/extent[1], (height-72)/max(extent[0], extent[2]))
    # Fitting only the virtual scene zooms/crops the physical camera image.
    # Cap the COMMON metric scale so both original sensor rectangles fit.
    # Alignment stays in physical coordinates; all padding remains display-only.
    def full_frame_limit(shape, u0, v0, x_scale, y_scale, x_centre, y_centre):
        h, w = shape[:2]
        dx = np.array([0., w])/x_scale-u0/x_scale-x_centre
        dy = np.array([0., h])/y_scale-v0/y_scale-y_centre
        return min((width/2-16)/max(np.max(np.abs(dx)), 1e-9),
                   (height/2-28)/max(np.max(np.abs(dy)), 1e-9))

    if top is not None:
        scale = min(scale, full_frame_limit(top.shape, *top_origin,
                    top_px_per_cm, top_px_per_cm, centre[1], centre[0]))
    side_shape = side.shape if side is not None else side_frame_shape
    if side_shape is not None:
        scale = min(scale, full_frame_limit(side_shape, *origin,
                    side_detector.y_sign*side_detector.side_y_px_per_cm,
                    side_detector.side_px_per_cm, centre[1], -centre[2]))
    if not np.isfinite(scale) or scale <= 0:
        return top, side
    top_ratio = scale/top_px_per_cm
    side_ratio_y = scale/(side_detector.y_sign*side_detector.side_y_px_per_cm)
    side_ratio_z = scale/side_detector.side_px_per_cm
    top_matrix = np.array([
        [top_ratio, 0., width/2-scale*centre[1]-top_ratio*top_origin[0]],
        [0., top_ratio, height/2-scale*centre[0]-top_ratio*top_origin[1]],
    ])
    side_matrix = np.array([
        [side_ratio_y, 0., width/2-scale*centre[1]-side_ratio_y*origin[0]],
        [0., side_ratio_z, height/2+scale*centre[2]-side_ratio_z*origin[1]],
    ])

    def transform(frame, matrix):
        if frame is None:
            return None
        return cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

    return transform(top, top_matrix), transform(side, side_matrix)
