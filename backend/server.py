import torch
from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid
import os
from ultralytics import YOLO
from pathlib import Path
import asyncio
import shutil

from services.catalog.identify import identify_whale_group_pgvector
from services.detection.tracker_service import process_video_obb_tracking, rank_frames_within_tracks, filter_tracks, \
    save_top_results_per_track
from services.reid.classifier import SphereClassifier

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

detectionModel = YOLO("models/detection/obb/weights/best.pt")

ckpt_path = "models/reid/b7/0/best.ckpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reidModel = SphereClassifier.load_from_checkpoint(ckpt_path, map_location=device,
                                              id_class_nums=None,
                                              backbone_pretrained=False)

reidModel = reidModel.to(device)
reidModel.eval()
reidModel.freeze()

db_config = {
    "host": "localhost",
    "port": 5432,
    "dbname": "whale_reid_db",
    "user": "whale_app",
    "password": "3110"
}

jobs = {}

UPLOAD_DIR = "uploads"
RESPONSES_DIR = "responses"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSES_DIR, exist_ok=True)

BASE_DIR = Path(".")
VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(RESULTS_DIR)), name="static")


def process_video_job_sync(job_id: str, video_path: str):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 5
        jobs[job_id]["stage"] = "tracking"

        result = process_video_obb_tracking(
            video_path=video_path,
            model=detectionModel,
            tracker="bytetrack.yaml",
            conf=0.10,
            iou=0.50,
            frame_step=10,
            min_conf_for_save=0.05,
            min_area_ratio=0.001,
            save_untracked=False
        )

        jobs[job_id]["progress"] = 40
        jobs[job_id]["stage"] = "filtering"

        tracks = result["tracks"]
        tracks = filter_tracks(tracks, min_track_length=5)

        jobs[job_id]["progress"] = 60
        jobs[job_id]["stage"] = "ranking"

        tracks = rank_frames_within_tracks(tracks)

        jobs[job_id]["progress"] = 80
        jobs[job_id]["stage"] = "saving"

        job_result_dir = RESULTS_DIR / job_id

        saved_tracks = save_top_results_per_track(
            video_path=video_path,
            tracks=tracks,
            output_dir=job_result_dir,
            top_tracks=10,
            top_frames_per_track=5
        )

        tracks_response = []

        for track in saved_tracks:
            track_frames = []

            for item in track["frames"]:
                full_rel = Path(item["full_path"]).relative_to(RESULTS_DIR).as_posix()
                ann_rel = Path(item["annotated_path"]).relative_to(RESULTS_DIR).as_posix()
                crop_rel = Path(item["crop_path"]).relative_to(RESULTS_DIR).as_posix()

                track_frames.append({
                    "id": f"{job_id}_track_{track['track_id']}_{item['id']}",
                    "rank": item["rank"],
                    "score": item["score"],
                    "timestamp": item["timestamp"],
                    "confidence": item.get("confidence"),
                    "bbox": item.get("bbox"),
                    "corners": item.get("corners"),
                    "frame_index": item.get("frame_index"),
                    "url": f"/static/{full_rel}",
                    "annotated_url": f"/static/{ann_rel}",
                    "crop_url": f"/static/{crop_rel}",
                    "full_path": item["full_path"],
                    "annotated_path": item["annotated_path"],
                    "crop_path": item["crop_path"],
                })

            tracks_response.append({
                "track_id": track["track_id"],
                "start_frame": track.get("start_frame"),
                "end_frame": track.get("end_frame"),
                "start_time_sec": track.get("start_time_sec"),
                "end_time_sec": track.get("end_time_sec"),
                "frames": track_frames
            })

        jobs[job_id]["tracks"] = tracks_response
        jobs[job_id]["frames"] = []  # можно оставить для совместимости
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "completed"

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["stage"] = "failed"
        jobs[job_id]["message"] = str(e)


async def process_video_job(job_id: str, video_path: str):
    await asyncio.to_thread(process_video_job_sync, job_id, video_path)


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    job_id = f"job_{uuid.uuid4().hex[:12]}"

    ext = Path(video.filename).suffix if video.filename else ".mp4"
    video_path = VIDEO_DIR / f"{job_id}{ext}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "stage": "queued",
        "video_path": str(video_path),
        "tracks": [],
        "frames": []
    }

    asyncio.create_task(process_video_job(job_id, str(video_path)))

    return {"job_id": job_id}


@app.get("/api/job/{job_id}/status")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        return {
            "status": "failed",
            "message": "Job not found"
        }

    job = jobs[job_id]

    return {
        "status": job["status"],
        "progress": job["progress"],
        "stage": job.get("stage", ""),
        "tracks_count": len(job.get("tracks", [])) if job["status"] == "completed" else 0,
        "message": {
            "queued": "Задача поставлена в очередь…",
            "processing": "Обработка видео…",
            "completed": "Готово!",
            "failed": job.get("message", "Ошибка обработки")
        }.get(job["status"], "")
    }


@app.get("/api/job/{job_id}/tracks")
async def get_tracks(job_id: str):
    if job_id not in jobs:
        return {"tracks": []}

    if jobs[job_id]["status"] != "completed":
        return {"tracks": []}

    return {"tracks": jobs[job_id].get("tracks", [])}


identify_jobs = {}


def process_identification_job_sync(
    identify_job_id: str,
    source_job_id: str,
    selections: list[dict]
):
    try:
        identify_jobs[identify_job_id]["status"] = "processing"
        identify_jobs[identify_job_id]["progress"] = 5

        source_job = jobs.get(source_job_id)
        if not source_job or source_job.get("status") != "completed":
            raise ValueError("Source job not found or not completed")

        available_tracks = source_job.get("tracks", [])
        track_map = {track["track_id"]: track for track in available_tracks}

        if not selections:
            raise ValueError("No selections provided")

        results = []
        total = len(selections)

        for idx, selection in enumerate(selections, start=1):
            track_id = selection.get("track_id")
            frame_ids = selection.get("frame_ids", [])

            if track_id not in track_map:
                continue

            track = track_map[track_id]
            track_frames = track.get("frames", [])

            selected_frames = [
                frame for frame in track_frames
                if frame["id"] in frame_ids
            ]

            if not selected_frames:
                continue

            image_paths = [
                frame["crop_path"]
                for frame in selected_frames
                if frame.get("crop_path") and Path(frame["crop_path"]).exists()
            ]

            if not image_paths:
                continue

            group_result = identify_whale_group_pgvector(
                model=reidModel,
                image_paths=image_paths,
                device=device,
                db_config=db_config,
                image_size=(600, 600),
                model_name="EfficientNetB7",
                model_version="efficientnet_b7_v1",
                top_k=5,
                search_k=50,
                threshold=1.2,
            )

            matches = []
            for match in group_result.get("top_k", []):
                matches.append({
                    "whale_id": match.get("whale_uid"),
                    "score": match.get("score"),
                    "best_distance": match.get("best_distance"),
                    "best_match_image": match.get("best_match_image"),
                    "name": None,
                    "thumbnail_url": None
                })

            results.append({
                "track_id": track_id,
                "selected_frame_ids": frame_ids,
                "frames": [
                    {
                        "frame_id": frame["id"],
                        "frame_url": frame.get("url"),
                        "crop_url": frame.get("crop_url"),
                        "annotated_url": frame.get("annotated_url"),
                    }
                    for frame in selected_frames
                ],
                "status": group_result.get("status"),
                "is_new": group_result.get("status") == "new_whale",
                "aggregated_distance": group_result.get("aggregated_distance"),
                "best_whale": group_result.get("best_whale"),
                "nearest_known_whale": group_result.get("nearest_known_whale"),
                "nearest_known_image": group_result.get("nearest_known_image"),
                "matches": matches,
                "per_image_results": group_result.get("per_image_results", [])
            })

            identify_jobs[identify_job_id]["progress"] = int(idx / total * 100)

        identify_jobs[identify_job_id]["status"] = "completed"
        identify_jobs[identify_job_id]["progress"] = 100
        identify_jobs[identify_job_id]["results"] = results

    except Exception as e:
        identify_jobs[identify_job_id]["status"] = "failed"
        identify_jobs[identify_job_id]["progress"] = 0
        identify_jobs[identify_job_id]["message"] = str(e)


async def process_identification_job(
    identify_job_id: str,
    source_job_id: str,
    selections: list[dict]
):
    await asyncio.to_thread(
        process_identification_job_sync,
        identify_job_id,
        source_job_id,
        selections
    )


@app.post("/api/identify")
async def identify_whales(data: dict):
    source_job_id = data.get("job_id", "")
    selections = data.get("selections", [])

    identify_job_id = f"identify_{uuid.uuid4().hex[:12]}"

    identify_jobs[identify_job_id] = {
        "status": "queued",
        "progress": 0,
        "results": []
    }

    asyncio.create_task(
        process_identification_job(
            identify_job_id=identify_job_id,
            source_job_id=source_job_id,
            selections=selections
        )
    )

    return {"job_id": identify_job_id}


@app.get("/api/identify/{job_id}/status")
async def get_identify_status(job_id: str):
    if job_id not in identify_jobs:
        return {
            "status": "failed",
            "message": "Identify job not found"
        }

    job = identify_jobs[job_id]

    if job["status"] == "completed":
        return {
            "status": "completed",
            "results": job.get("results", [])
        }

    return {
        "status": job["status"],
        "progress": job["progress"],
        "message": {
            "queued": "Задача поставлена в очередь…",
            "processing": "Идентификация китов…",
            "failed": job.get("message", "Ошибка идентификации")
        }.get(job["status"], "")
    }


app.include_router(router, prefix="/api", tags=["Images"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8005, reload=True)
