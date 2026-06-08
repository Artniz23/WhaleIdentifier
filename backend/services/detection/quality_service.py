import cv2
import math
import numpy as np


def clip_bbox(bbox, frame_shape):
    """
    Ограничивает координаты bbox границами изображения.

    Гарантирует:
    - координаты находятся внутри кадра;
    - ширина и высота bbox не меньше 1 пикселя.
    """

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1:
        x2 = min(x1 + 1, w - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, h - 1)

    return x1, y1, x2, y2


def normalize_score(value, min_val, max_val):
    """
    Нормализует значение в диапазон [0, 1].

    Значения ниже min_val становятся 0,
    значения выше max_val становятся 1.
    """

    if value is None or max_val <= min_val:
        return 0.0
    value = max(min_val, min(value, max_val))
    return float((value - min_val) / (max_val - min_val))


def corners_to_bbox(corners, frame_shape=None):
    """
    Строит axis-aligned bbox по вершинам OBB.

    При необходимости дополнительно ограничивает bbox
    границами кадра.
    """

    corners = np.array(corners, dtype=np.float32)

    x1 = float(np.min(corners[:, 0]))
    y1 = float(np.min(corners[:, 1]))
    x2 = float(np.max(corners[:, 0]))
    y2 = float(np.max(corners[:, 1]))

    bbox = (x1, y1, x2, y2)
    if frame_shape is not None:
        bbox = clip_bbox(bbox, frame_shape)
    return bbox


def polygon_area(corners):
    """
    Вычисляет площадь полигона по его вершинам.

    Используется для расчёта площади OBB.
    """

    corners = np.array(corners, dtype=np.float32)
    return float(cv2.contourArea(corners.astype(np.float32)))


def order_quad_points(pts):
    """
    Приводит вершины четырёхугольника к порядку:

        top-left
        top-right
        bottom-right
        bottom-left

    Необходимо для корректного perspective transform.
    """

    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def crop_rotated_obb(frame, corners):
    """
    Вырезает объект по ориентированному bounding box (OBB).

    Выполняет perspective transform и возвращает
    выровненный прямоугольный crop объекта.
    """

    corners = order_quad_points(corners)
    tl, tr, br, bl = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    crop_w = int(round(max(width_top, width_bottom)))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    crop_h = int(round(max(height_right, height_left)))

    crop_w = max(crop_w, 1)
    crop_h = max(crop_h, 1)

    dst = np.array([
        [0, 0],
        [crop_w - 1, 0],
        [crop_w - 1, crop_h - 1],
        [0, crop_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (crop_w, crop_h))

    return warped


def compute_border_touch(bbox, frame_shape, margin_px=20, margin_ratio=0.01):
    """
    Проверяет касается ли объект границ кадра.

    Объекты возле краёв часто оказываются частично
    обрезанными и имеют более низкое качество.
    """

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, frame_shape)

    mx = max(margin_px, int(w * margin_ratio))
    my = max(margin_px, int(h * margin_ratio))

    return {
        "border_touch": bool(
            x1 <= mx or y1 <= my or x2 >= (w - mx) or y2 >= (h - my)
        )
    }


def compute_blur(crop):
    """
    Оценивает резкость изображения через дисперсию Лапласиана.

    Чем выше значение, тем более резким считается изображение.
    """

    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness_contrast(crop):
    """
    Возвращает:

    - среднюю яркость изображения;
    - стандартное отклонение яркости (контраст).
    """

    if crop is None or crop.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def compute_glare_ratio(crop, v_thresh=240, s_thresh=60):
    """
    Оценивает долю пересвеченных областей.

    Блики определяются как очень яркие пиксели
    с низкой насыщенностью.
    """

    if crop is None or crop.size == 0:
        return 1.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    glare_mask = (v >= v_thresh) & (s <= s_thresh)
    return float(np.mean(glare_mask))


def compute_splash_ratio(crop, bright_thresh=220, max_component_area=150):
    """
    Оценивает количество мелких ярких пятен.

    Используется как эвристика для поиска бликов,
    брызг воды и шумовых артефактов.
    """

    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright_mask = (gray >= bright_thresh).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

    small_area_sum = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 1 <= area <= max_component_area:
            small_area_sum += area

    total_area = crop.shape[0] * crop.shape[1]
    return float(small_area_sum / total_area) if total_area > 0 else 0.0


def compute_geometry_features_obb(detection, frame_shape):
    """
    Вычисляет геометрические характеристики объекта.

    Возвращает:
    - размеры bbox и OBB;
    - площади;
    - aspect ratio;
    - положение центра;
    - угол поворота;
    - относительные размеры объекта в кадре.
    """

    h, w = frame_shape[:2]
    bbox = detection["bbox"]
    corners = np.array(detection["corners"], dtype=np.float32)
    obb_area = detection["obb_area"]

    x1, y1, x2, y2 = clip_bbox(bbox, frame_shape)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = bbox_w * bbox_h
    frame_area = h * w

    xywhr = detection.get("xywhr")
    if xywhr is not None:
        cx, cy, obb_w, obb_h, angle_rad = xywhr
        aspect_ratio = max(obb_w, obb_h) / max(1e-6, min(obb_w, obb_h))
        angle_deg = math.degrees(angle_rad)
    else:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        obb_w = bbox_w
        obb_h = bbox_h
        aspect_ratio = bbox_w / max(1, bbox_h)
        angle_deg = 0.0

    return {
        "bbox": [x1, y1, x2, y2],
        "bbox_area": int(bbox_area),
        "bbox_area_ratio": float(bbox_area / frame_area),
        "obb_area": float(obb_area),
        "obb_area_ratio": float(obb_area / frame_area),
        "obb_fill_ratio": float(obb_area / bbox_area),
        "obb_aspect_ratio": float(aspect_ratio),
        "bbox_center_x_norm": float(cx / w),
        "bbox_center_y_norm": float(cy / h),
        "angle_deg": float(angle_deg),
        "corners": corners.tolist(),
        "obb_width": float(obb_w),
        "obb_height": float(obb_h),
    }


def compute_quality_obb(frame, detection):
    """
    Рассчитывает набор признаков качества объекта
    и итоговый quality score.

    Возвращает:
    - features: измеренные характеристики объекта;
    - scores: нормализованные оценки качества;
    - crop: выровненный crop объекта.
    """

    conf = detection.get("confidence", 0.0) or 0.0
    # Вырезаем и выравниваем объект по OBB.
    crop = crop_rotated_obb(frame, detection["corners"])

    # Геометрические признаки объекта.
    geom = compute_geometry_features_obb(detection, frame.shape)
    # Проверяем находится ли объект возле границы кадра.
    border_touch = compute_border_touch(detection["bbox"], frame.shape)["border_touch"]
    # Метрики качества изображения.
    blur = compute_blur(crop)
    brightness, contrast = compute_brightness_contrast(crop)
    glare = compute_glare_ratio(crop)
    splash = compute_splash_ratio(crop)

    # Нормализация признаков в диапазон [0, 1].
    area_score = normalize_score(geom["obb_area_ratio"], 0.01, 0.10)
    fill_score = normalize_score(geom["obb_fill_ratio"], 0.20, 0.80)
    blur_score = normalize_score(blur, 30, 120)
    contrast_score = normalize_score(contrast, 12, 28)

    # Эвристическая оценка корректности экспозиции.
    if 80 <= brightness <= 180:
        brightness_score = 1.0
    elif 60 <= brightness < 80 or 180 < brightness <= 210:
        brightness_score = 0.5
    else:
        brightness_score = 0.1

    # Штраф за пересветы и блики.
    glare_score = 1.0 - normalize_score(glare, 0.01, 0.08)
    # Штраф за мелкие яркие артефакты.
    splash_score = 1.0 - normalize_score(splash, 0.002, 0.02)
    # Штраф за близость к краям кадра.
    border_score = 0.0 if border_touch else 1.0

    # Эвристическая оценка формы объекта.
    ar = geom["obb_aspect_ratio"]
    # Предпочитаем объекты ближе к центру кадра.
    cy = geom["bbox_center_y_norm"]

    if 1.2 <= ar <= 6.0:
        shape_score = 1.0
    elif 0.9 <= ar < 1.2 or 6.0 < ar <= 8.0:
        shape_score = 0.5
    else:
        shape_score = 0.1

    position_score = 1.0 if 0.20 <= cy <= 0.80 else 0.5

    # Насколько хорошо виден сам объект.
    visible_body_score = (
            0.30 * area_score +
            0.30 * fill_score +
            0.15 * border_score +
            0.10 * shape_score +
            0.05 * position_score +
            0.10 * conf
    )

    # Финальный quality score.
    #
    # Комбинирует:
    # - уверенность детектора;
    # - размер объекта;
    # - заполненность bbox;
    # - резкость;
    # - контраст;
    # - яркость;
    # - наличие бликов;
    # - положение в кадре.
    final_score = (
            0.10 * conf +
            0.15 * area_score +
            0.10 * fill_score +
            0.15 * blur_score +
            0.10 * contrast_score +
            0.10 * brightness_score +
            0.10 * glare_score +
            0.05 * splash_score +
            0.05 * border_score +
            0.10 * visible_body_score
    )

    # Полный набор измеренных характеристик объекта.
    features = {
        **geom,
        "confidence": float(conf),
        "border_touch": bool(border_touch),
        "blur": float(blur),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "glare": float(glare),
        "splash": float(splash),
    }

    # Нормализованные оценки качества.
    scores = {
        "final": float(final_score),
        "area": float(area_score),
        "fill": float(fill_score),
        "blur": float(blur_score),
        "contrast": float(contrast_score),
        "brightness": float(brightness_score),
        "glare": float(glare_score),
        "splash": float(splash_score),
        "border": float(border_score),
        "visible_body": float(visible_body_score),
    }

    return features, scores, crop
