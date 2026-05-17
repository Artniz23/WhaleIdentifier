import cv2
import numpy as np
from pathlib import Path

from services.detection.quality_service import corners_to_bbox, polygon_area, compute_quality_obb, compute_geometry_features_obb, \
    crop_rotated_obb


def extract_tracked_obb_instances(result, frame_shape):
    instances = []

    if result.obb is None or len(result.obb) == 0:
        return instances

    cls_arr = result.obb.cls.cpu().numpy() if result.obb.cls is not None else None
    conf_arr = result.obb.conf.cpu().numpy() if result.obb.conf is not None else None
    xywhr_arr = result.obb.xywhr.cpu().numpy() if result.obb.xywhr is not None else None
    corners_arr = result.obb.xyxyxyxy.cpu().numpy() if result.obb.xyxyxyxy is not None else None

    ids_arr = None
    if hasattr(result.obb, "id") and result.obb.id is not None:
        ids_arr = result.obb.id.cpu().numpy()

    for i in range(len(result.obb)):
        corners = corners_arr[i]
        bbox = corners_to_bbox(corners, frame_shape)
        obb_area = polygon_area(corners)

        instances.append({
            "track_id": int(ids_arr[i]) if ids_arr is not None else None,
            "is_tracked": ids_arr is not None,
            "bbox": bbox,
            "corners": corners.tolist(),
            "obb_area": obb_area,
            "confidence": float(conf_arr[i]) if conf_arr is not None else 0.0,
            "class_id": int(cls_arr[i]) if cls_arr is not None else None,
            "xywhr": xywhr_arr[i].tolist() if xywhr_arr is not None else None,
        })

    return instances

def is_valid_detection_for_tracking_save(
    detection,
    frame_shape,
    min_conf=0.05,
    min_area_ratio=0.001
):
    geom = compute_geometry_features_obb(detection, frame_shape)

    if detection.get("confidence", 0.0) < min_conf:
        return False

    if geom["obb_area_ratio"] < min_area_ratio:
        return False

    return True

def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None

def draw_detection_obb(frame, detection):
    img = frame.copy()
    pts = np.array(detection["corners"], dtype=np.int32)

    score = detection["scores"]["final"]
    conf = detection["confidence"]
    track_id = detection.get("track_id")

    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=4)

    x, y = pts[0]
    label = f"id={track_id} | q={score:.3f} | conf={conf:.3f}"
    cv2.putText(
        img,
        label,
        (int(x), max(30, int(y) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    return img

def process_video_obb_tracking(
    video_path,
    model,
    tracker="bytetrack.yaml",
    conf=0.10,
    iou=0.50,
    frame_step=1,
    min_conf_for_save=0.05,
    min_area_ratio=0.001,
    save_untracked=False
):
    video_path = str(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"Видео: {video_path}")
    print(f"FPS: {fps}")
    print(f"Всего кадров: {total_frames}")
    print(f"Длительность: {duration_sec:.2f} сек")
    print(f"Трекинг: {tracker}")
    print(f"frame_step={frame_step}, conf={conf}, iou={iou}")

    samples = []
    tracks_dict = {}
    processed_frame_counter = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_step > 1 and (frame_index % frame_step != 0):
            frame_index += 1
            continue

        time_sec = frame_index / fps if fps > 0 else 0.0

        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            verbose=False
        )
        result = results[0]

        detections = extract_tracked_obb_instances(result, frame.shape)

        processed_detections = []
        for det in detections:
            if det["track_id"] is None and not save_untracked:
                continue

            if not is_valid_detection_for_tracking_save(
                det,
                frame.shape,
                min_conf=min_conf_for_save,
                min_area_ratio=min_area_ratio
            ):
                continue

            features, scores, crop = compute_quality_obb(frame, det)

            processed_det = {
                # старые поля
                "bbox": features["bbox"],
                "corners": features["corners"],
                "confidence": features["confidence"],
                "obb_area": features["obb_area"],
                "obb_area_ratio": features["obb_area_ratio"],
                "features": features,
                "scores": scores,
                "crop": crop,

                # новые поля трекинга
                "track_id": det["track_id"],
                "is_tracked": det["is_tracked"],
                "class_id": det["class_id"],
                "xywhr": det["xywhr"],
                "frame_index": frame_index,
                "time_sec": time_sec,
                "frame_name": f"frame_{processed_frame_counter:06d}_idx_{frame_index:06d}_t_{time_sec:.2f}s.jpg",

                # полный сырой detection
                "raw_detection": {
                    **det
                }
            }

            processed_detections.append(processed_det)

            track_id = det["track_id"]
            if track_id is not None:
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = {
                        "track_id": track_id,
                        "start_frame": frame_index,
                        "end_frame": frame_index,
                        "start_time_sec": time_sec,
                        "end_time_sec": time_sec,
                        "frames": [],
                    }

                tracks_dict[track_id]["end_frame"] = frame_index
                tracks_dict[track_id]["end_time_sec"] = time_sec
                tracks_dict[track_id]["frames"].append(processed_det)

        processed_detections = sorted(
            processed_detections,
            key=lambda x: x["scores"]["final"],
            reverse=True
        )

        samples.append({
            "frame_index": frame_index,
            "frame_name": f"frame_{processed_frame_counter:06d}_idx_{frame_index:06d}_t_{time_sec:.2f}s.jpg",
            "time_sec": time_sec,
            "detections": processed_detections
        })

        print(
            f"[TRACK] frame={frame_index} "
            f"t={time_sec:.2f}s "
            f"detections={len(detections)} "
            f"saved={len(processed_detections)}"
        )

        processed_frame_counter += 1
        frame_index += 1

    cap.release()

    tracks = list(tracks_dict.values())

    print(f"\nГотово.")
    print(f"Обработано кадров: {processed_frame_counter}")
    print(f"Кадров с сохранёнными detections: {len(samples)}")
    print(f"Всего треков: {len(tracks)}")

    return {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "samples": samples,
        "tracks": tracks,
    }

def filter_tracks(tracks, min_track_length=5):
    filtered = []
    for track in tracks:
        if len(track["frames"]) >= min_track_length:
            filtered.append(track)
    return filtered

def rank_frames_within_tracks(tracks):
    ranked_tracks = []

    for track in tracks:
        frames_sorted = sorted(
            track["frames"],
            key=lambda x: x["scores"]["final"],
            reverse=True
        )

        enriched_frames = []
        for rank_idx, item in enumerate(frames_sorted, start=1):
            new_item = {
                **item,
                "track_rank": rank_idx,
                "track_length": len(frames_sorted),
            }
            enriched_frames.append(new_item)

        ranked_tracks.append({
            **track,
            "frames": enriched_frames,
            "best_detection": enriched_frames[0] if enriched_frames else None,
            "best_score": enriched_frames[0]["scores"]["final"] if enriched_frames else None,
            "track_length": len(enriched_frames),
        })

    ranked_tracks = sorted(
        ranked_tracks,
        key=lambda x: x["best_score"] if x["best_score"] is not None else -1,
        reverse=True
    )

    return ranked_tracks

def summarize_tracks(tracks):
    summaries = []

    for track in tracks:
        best = track.get("best_detection")
        if best is None:
            continue

        summary = {
            "track_id": track["track_id"],
            "track_length": track["track_length"],
            "start_frame": track["start_frame"],
            "end_frame": track["end_frame"],
            "start_time_sec": track["start_time_sec"],
            "end_time_sec": track["end_time_sec"],
            "duration_sec": track["end_time_sec"] - track["start_time_sec"],
            "best_score": best["scores"]["final"],
            "best_confidence": best["confidence"],
            "best_frame_index": best["frame_index"],
            "best_time_sec": best["time_sec"],
            "best_bbox": best["bbox"],
            "best_corners": best["corners"],
            "best_features": best["features"],
            "best_scores": best["scores"],
        }
        summaries.append(summary)

    summaries = sorted(summaries, key=lambda x: x["best_score"], reverse=True)
    return summaries

def save_top_results_per_track(
    video_path,
    tracks,
    output_dir="tracked_results",
    top_tracks=10,
    top_frames_per_track=3
):
    from pathlib import Path
    import cv2

    output_dir = Path(output_dir)
    full_dir = output_dir / "full"
    ann_dir = output_dir / "annotated"
    crop_dir = output_dir / "crops"

    full_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    saved_tracks = []

    for track_rank, track in enumerate(tracks[:top_tracks], start=1):
        track_id = track.get("track_id")
        frames = track.get("frames", [])[:top_frames_per_track]

        track_result = {
            "track_id": track_id,
            "track_rank": track_rank,
            "start_frame": track.get("start_frame"),
            "end_frame": track.get("end_frame"),
            "start_time_sec": track.get("start_time_sec"),
            "end_time_sec": track.get("end_time_sec"),
            "frames": []
        }

        for frame_rank, item in enumerate(frames, start=1):
            frame = read_frame(video_path, item.get("frame_index"))
            if frame is None:
                print(f"[WARN] Не удалось прочитать кадр {item.get('frame_index')}")
                continue

            det = item
            crop = crop_rotated_obb(frame, det.get("corners"))
            ann = draw_detection_obb(frame, det)

            # score и timestamp могут быть в разных полях — возьмём доступныe варианты
            score = None
            if isinstance(det.get("scores"), dict):
                score = det["scores"].get("final")
            score = score if score is not None else det.get("score")

            timestamp = det.get("time_sec", det.get("timestamp"))

            base_name = (
                f"track_{track_id:03d}"
                f"_trackRank_{track_rank:02d}"
                f"_frameRank_{frame_rank:02d}"
                f"_t_{(timestamp if timestamp is not None else 0):.2f}s"
                f"_score_{(score if score is not None else 0):.4f}.jpg"
            )

            full_path = full_dir / base_name
            ann_path = ann_dir / base_name
            crop_path = crop_dir / base_name

            cv2.imwrite(str(full_path), frame)
            cv2.imwrite(str(ann_path), ann)
            cv2.imwrite(str(crop_path), crop)

            frame_entry = {
                "id": det.get("id", f"{track_id}_{frame_rank}"),
                "rank": frame_rank,
                "score": score,
                "timestamp": timestamp,
                "confidence": det.get("confidence"),
                "bbox": det.get("bbox"),
                "corners": det.get("corners"),
                "frame_index": det.get("frame_index"),
                "full_path": str(full_path),
                "annotated_path": str(ann_path),
                "crop_path": str(crop_path),
            }

            track_result["frames"].append(frame_entry)

            print(f"[SAVED] {base_name}")

        # Добавляем трек только если есть сохранённые кадры
        if track_result["frames"]:
            saved_tracks.append(track_result)

    return saved_tracks