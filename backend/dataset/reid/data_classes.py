from dataclasses import dataclass, field

@dataclass
class WhaleInfo:
    source_folder: str
    count: int
    sizes: list[tuple[int, int]]
    image_paths: list[str]


@dataclass
class DatasetStats:
    total_whales: int = 0
    total_images: int = 0
    whales: dict[str, WhaleInfo] = field(default_factory=dict)