from pathlib import Path
import math
import xml.etree.ElementTree as ET

def rotate_points(points, cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = []
    for x, y in points:
        dx = x - cx
        dy = y - cy

        rx = dx * cos_a - dy * sin_a + cx
        ry = dx * sin_a + dy * cos_a + cy
        rotated.append((rx, ry))

    return rotated

def clip_point(x, y, img_w, img_h):
    x = max(0.0, min(float(x), float(img_w)))
    y = max(0.0, min(float(y), float(img_h)))
    return x, y

def cvat_box_to_obb_corners(xtl, ytl, xbr, ybr, rotation_deg=0.0):
    """
    Возвращает 4 угла rotated bbox в порядке:
    top-left, top-right, bottom-right, bottom-left
    """
    cx = (xtl + xbr) / 2.0
    cy = (ytl + ybr) / 2.0

    corners = [
        (xtl, ytl),  # top-left
        (xbr, ytl),  # top-right
        (xbr, ybr),  # bottom-right
        (xtl, ybr),  # bottom-left
    ]

    if abs(rotation_deg) < 1e-8:
        return corners

    return rotate_points(corners, cx, cy, rotation_deg)

def corners_to_yolo_obb_line(corners, img_w, img_h, class_id=0):
    norm_coords = []

    for x, y in corners:
        x, y = clip_point(x, y, img_w, img_h)
        xn = x / img_w
        yn = y / img_h
        norm_coords.extend([xn, yn])

    coords_str = " ".join(f"{v:.6f}" for v in norm_coords)
    return f"{class_id} {coords_str}"

def process_xml(xml_path, output_dir, class_name_to_id=None):
    if class_name_to_id is None:
        class_name_to_id = {"Whale": 0}

    xml_path = Path(xml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    created_files = []

    for image_tag in root.findall("image"):
        image_name = image_tag.attrib["name"]
        img_w = float(image_tag.attrib["width"])
        img_h = float(image_tag.attrib["height"])

        txt_name = Path(image_name).stem + ".txt"
        txt_path = output_dir / txt_name

        lines = []

        for box_tag in image_tag.findall("box"):
            label = box_tag.attrib.get("label", "")
            if label not in class_name_to_id:
                continue

            class_id = class_name_to_id[label]

            xtl = float(box_tag.attrib["xtl"])
            ytl = float(box_tag.attrib["ytl"])
            xbr = float(box_tag.attrib["xbr"])
            ybr = float(box_tag.attrib["ybr"])
            rotation = float(box_tag.attrib.get("rotation", 0.0))

            if xbr <= xtl or ybr <= ytl:
                continue

            corners = cvat_box_to_obb_corners(
                xtl=xtl,
                ytl=ytl,
                xbr=xbr,
                ybr=ybr,
                rotation_deg=rotation
            )

            line = corners_to_yolo_obb_line(
                corners=corners,
                img_w=img_w,
                img_h=img_h,
                class_id=class_id
            )
            lines.append(line)

        txt_path.write_text("\n".join(lines), encoding="utf-8")
        created_files.append(txt_path)

    return created_files

def main():
    process_xml(
        xml_path="detection_dataset/annotations/annotations.xml",
        output_dir="detection_dataset/labels_obb"
    )

if __name__ == "__main__":
    main()