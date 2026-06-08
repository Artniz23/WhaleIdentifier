from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetDirs:
    train_img: Path
    val_img: Path
    train_lbl: Path
    val_lbl: Path

@dataclass
class DatasetPairs:
    pairs: list[tuple[Path, Path]]
    missing_images: list[str]
    missing_labels: list[str]

@dataclass
class DatasetSummary:
    output_dir: str
    yaml_path: str
    total_used: int
    train_count: int
    val_count: int
    train_first: str | None
    train_last: str | None
    val_first: str | None
    val_last: str | None
    missing_images: list[str]
    missing_labels: list[str]

@dataclass
class BoundingBox:
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    rotation: float = 0.0

@dataclass
class Point:
    x: float
    y: float