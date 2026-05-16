// ========================
// API Response Models
// ========================

export interface UploadResponse {
  job_id: string;
}

export interface JobStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  message?: string;
  stage?: string; // e.g. "detection", "extraction"
  results?: IdentificationGroupResult[];
}

export type TrackId = string | number;

export interface Track {
  track_id: TrackId;
  frames: Frame[];
}

export interface Frame {
  id: string;
  url: string;
  annotated_url: string;
  score: number;       // quality score 0-1
  timestamp?: number;  // seconds in video
  thumbnail_url?: string;
}

export interface WhaleMatch {
  whale_id: string;
  score: number;  // 0-1
  name?: string;
  thumbnail_url?: string;
}

export interface IdentificationResult {
  frame_id?: string;
  frame_url?: string;
  matches: WhaleMatch[];
  is_new: boolean;
  approved?: boolean | null; // null = not decided yet
}

export interface IdentifySelection {
  track_id: TrackId;
  frame_ids: string[];
}

export interface GroupedResultFrame {
  frame_id: string;
  frame_url?: string;
  crop_url?: string;
  annotated_url?: string;
}

export interface IdentificationGroupResult {
  track_id: TrackId;
  selected_frame_ids: string[];
  frames: GroupedResultFrame[];
  status?: string;
  is_new: boolean;
  aggregated_distance?: number;
  best_whale?: WhaleMatch | null;
  nearest_known_whale?: WhaleMatch | null;
  nearest_known_image?: string | null;
  matches: WhaleMatch[];
  per_image_results?: unknown[];
  approved?: boolean | null; // null = not decided yet
}

export interface IdentifyResponse {
  job_id: string;
  results?: IdentificationGroupResult[];
}

// ========================
// UI State Models
// ========================

export type AppStage =
  | 'upload'
  | 'uploading'
  | 'processing'
  | 'frames'
  | 'identifying'
  | 'results';

