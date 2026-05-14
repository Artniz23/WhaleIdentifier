from ultralytics import YOLO

from services.tracker_service import process_video_obb_tracking, filter_tracks, rank_frames_within_tracks, \
    save_top_results_per_track


def main():
    model = YOLO("models/detection/obb/weights/best.pt")

    result = process_video_obb_tracking(
        video_path="DJI_0022.MP4",
        model=model,
        tracker="bytetrack.yaml",
        conf=0.10,
        iou=0.50,
        frame_step=10,
        min_conf_for_save=0.05,
        min_area_ratio=0.001,
        save_untracked=False
    )

    tracks = result["tracks"]

    tracks = filter_tracks(tracks, min_track_length=5)

    tracks = rank_frames_within_tracks(tracks)

    save_top_results_per_track(
        video_path="DJI_0022.MP4",
        tracks=tracks,
        output_dir="tracked_results_obb",
        top_tracks=10,
        top_frames_per_track=5
    )


if __name__ == "__main__":
    main()
