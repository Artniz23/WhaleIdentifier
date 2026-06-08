from pathlib import Path
import math
import xml.etree.ElementTree as ET
import argparse

from dataset.detection.data_classes import BoundingBox

EPSILON = 1e-8

DEFAULT_XML_PATH = (
    "detection_dataset/annotations/annotations.xml"
)

DEFAULT_OUTPUT_DIR = (
    "detection_dataset/labels_obb"
)

DEFAULT_CLASSES = {
    "Whale": 0,
}


def rotate_points(
        points: list[tuple[float, float]],
        cx: float,
        cy: float,
        angle_deg: float,
) -> list[tuple[float, float]]:
    """Повернуть точки вокруг центра."""
    angle_rad = math.radians(angle_deg)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = []

    for x, y in points:
        dx = x - cx
        dy = y - cy

        rotated.append(
            (
                dx * cos_a - dy * sin_a + cx,
                dx * sin_a + dy * cos_a + cy,
            )
        )

    return rotated


def clip_point(
        x: float,
        y: float,
        img_w: float,
        img_h: float,
) -> tuple[float, float]:
    """Ограничить координату границами изображения."""
    return (
        max(0.0, min(x, img_w)),
        max(0.0, min(y, img_h)),
    )


def parse_box(box_tag: ET.Element) -> BoundingBox:
    """Преобразовать XML box в BoundingBox."""
    return BoundingBox(
        xtl=float(box_tag.attrib["xtl"]),
        ytl=float(box_tag.attrib["ytl"]),
        xbr=float(box_tag.attrib["xbr"]),
        ybr=float(box_tag.attrib["ybr"]),
        rotation=float(
            box_tag.attrib.get("rotation", 0.0)
        ),
    )


def cvat_box_to_obb_corners(
        bbox: BoundingBox,
) -> list[tuple[float, float]]:
    """
    Получить 4 угла OBB.

    Порядок:
    top-left,
    top-right,
    bottom-right,
    bottom-left.
    """
    cx = (bbox.xtl + bbox.xbr) / 2.0
    cy = (bbox.ytl + bbox.ybr) / 2.0

    corners = [
        (bbox.xtl, bbox.ytl),
        (bbox.xbr, bbox.ytl),
        (bbox.xbr, bbox.ybr),
        (bbox.xtl, bbox.ybr),
    ]

    if abs(bbox.rotation) < EPSILON:
        return corners

    return rotate_points(
        corners,
        cx,
        cy,
        bbox.rotation,
    )


def format_yolo_obb_line(
        corners: list[tuple[float, float]],
        img_w: float,
        img_h: float,
        class_id: int,
) -> str:
    """Сформировать строку YOLO OBB."""
    normalized = []

    for x, y in corners:
        x, y = clip_point(
            x,
            y,
            img_w,
            img_h,
        )

        normalized.extend(
            [
                x / img_w,
                y / img_h,
            ]
        )

    coords = " ".join(
        f"{value:.6f}"
        for value in normalized
    )

    return f"{class_id} {coords}"


def process_image(
        image_tag: ET.Element,
        output_dir: Path,
        class_name_to_id: dict[str, int],
) -> Path:
    """Обработать один image tag."""

    image_name = image_tag.attrib["name"]

    img_w = float(
        image_tag.attrib["width"]
    )

    img_h = float(
        image_tag.attrib["height"]
    )

    txt_path = (
            output_dir /
            f"{Path(image_name).stem}.txt"
    )

    lines = []

    for box_tag in image_tag.findall("box"):
        label = box_tag.attrib.get(
            "label",
            "",
        )

        if label not in class_name_to_id:
            continue

        bbox = parse_box(box_tag)

        if (
                bbox.xbr <= bbox.xtl
                or bbox.ybr <= bbox.ytl
        ):
            continue

        corners = cvat_box_to_obb_corners(
            bbox
        )

        lines.append(
            format_yolo_obb_line(
                corners=corners,
                img_w=img_w,
                img_h=img_h,
                class_id=class_name_to_id[
                    label
                ],
            )
        )

    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return txt_path


def process_xml(
        xml_path: str | Path,
        output_dir: str | Path,
        class_name_to_id: dict[str, int] | None = None,
) -> list[Path]:
    """Конвертировать CVAT XML в YOLO OBB."""

    if class_name_to_id is None:
        class_name_to_id = DEFAULT_CLASSES

    xml_path = Path(xml_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    root = ET.parse(
        xml_path
    ).getroot()

    return [
        process_image(
            image_tag=image_tag,
            output_dir=output_dir,
            class_name_to_id=class_name_to_id,
        )
        for image_tag in root.findall("image")
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML to YOLO OBB"
    )

    parser.add_argument(
        "--xml",
        default=DEFAULT_XML_PATH,
        help="Путь к annotations.xml",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для txt-аннотаций",
    )

    args = parser.parse_args()

    created = process_xml(
        xml_path=args.xml,
        output_dir=args.output_dir,
    )

    print(
        f"Создано файлов: {len(created)}"
    )


if __name__ == "__main__":
    main()
