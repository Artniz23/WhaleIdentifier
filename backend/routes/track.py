from ultralytics import YOLO
from dotenv import load_dotenv
import argparse
from core.env import get_required_env
from services.detection.tracker_service import process_video_obb_tracking, filter_tracks, rank_frames_within_tracks, \
    save_top_results_per_track


def load_model() -> YOLO:
    """
    Загружает YOLO OBB модель.
    """
    ckpt_path = get_required_env("YOLO_CKPT_PATH")
    return YOLO(ckpt_path)


def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Run OBB tracking on video."
    )

    parser.add_argument(
        "--video-path",
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--output-dir",
        default="tracked_results_obb",
        help="Directory for saved results.",
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker config.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="Detection IoU threshold.",
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Process every N-th frame.",
    )

    parser.add_argument(
        "--min-conf-for-save",
        type=float,
        default=0.05,
        help="Minimum confidence for saving detection.",
    )

    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.001,
        help="Minimum object area ratio.",
    )

    parser.add_argument(
        "--min-track-length",
        type=int,
        default=5,
        help="Minimum track length.",
    )

    parser.add_argument(
        "--top-tracks",
        type=int,
        default=10,
        help="Number of tracks to export.",
    )

    parser.add_argument(
        "--top-frames-per-track",
        type=int,
        default=5,
        help="Number of frames per track.",
    )

    parser.add_argument(
        "--save-untracked",
        action="store_true",
        help="Save detections without track id.",
    )

    return parser.parse_args()


def main():
    """
    Точка входа в пайплайн OBB-трекинга.
    """

    load_dotenv()

    args = parse_args()

    model = load_model()

    result = process_video_obb_tracking(
        video_path=args.video_path,
        model=model,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        frame_step=args.frame_step,
        min_conf_for_save=args.min_conf_for_save,
        min_area_ratio=args.min_area_ratio,
        save_untracked=args.save_untracked,
    )

    tracks = result["tracks"]

    tracks = filter_tracks(
        tracks,
        min_track_length=args.min_track_length,
    )

    tracks = rank_frames_within_tracks(tracks)

    save_top_results_per_track(
        video_path=args.video_path,
        tracks=tracks,
        output_dir=args.output_dir,
        top_tracks=args.top_tracks,
        top_frames_per_track=args.top_frames_per_track,
    )


if __name__ == "__main__":
    main()
