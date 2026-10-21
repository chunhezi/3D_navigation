"""上视、侧视共用的暗磁珠预处理与腐蚀前轮廓关联。"""

import cv2


# ============== 本次视觉修改【两路公用】：预处理开始 ==============
# 作用：统一平滑、反向二值化和可选腐蚀，同时返回标定测径、小窗显示所需的图。
def preprocess_bead(gray, threshold, kernel_size, erode_iterations):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    if erode_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded = cv2.erode(binary, kernel, iterations=erode_iterations)
    else:
        eroded = binary.copy()
    return binary, eroded
# ============== 本次视觉修改【两路公用】：预处理结束 ==============


# ============== 本次视觉修改【两路公用】：原始轮廓关联开始 ==============
# 作用：关联同一目标的腐蚀前轮廓，防止腐蚀改变标定直径；0 次时直接复用轮廓。
def get_calibration_contour(binary, selected_contour, roi_mask, erode_iterations):
    contour = selected_contour
    if erode_iterations > 0:
        original_roi = cv2.bitwise_and(binary, roi_mask)
        originals, _ = cv2.findContours(
            original_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        seed = tuple(float(value) for value in selected_contour[0, 0])
        matches = [c for c in originals if cv2.pointPolygonTest(c, seed, False) >= 0]
        if len(matches) != 1:
            return None
        contour = matches[0]

    # 原始轮廓必须完整；触及 ROI 裁剪边界时只拒绝标定观测。
    x, y, width, height = cv2.boundingRect(roi_mask)
    cx, cy, cw, ch = cv2.boundingRect(contour)
    if (cv2.contourArea(contour) <= 0 or cx <= x or cy <= y
            or cx + cw >= x + width or cy + ch >= y + height):
        return None
    return contour
# ============== 本次视觉修改【两路公用】：原始轮廓关联结束 ==============
