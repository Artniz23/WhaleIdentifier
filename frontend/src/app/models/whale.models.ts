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
  results?: IdentificationResult[];
}

export interface Track {
  track_id: string;
  frames: Frame[];
}

export interface Frame {
  id: string;
  url: string;
  score: number;       // quality score 0-1
  timestamp?: number;  // seconds in video
  thumbnail_url?: string;
}

export interface WhaleMatch {
  whale_id: string;
  confidence: number;  // 0-1
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

export interface IdentifyResponse {
  job_id: string;
  results?: IdentificationResult[];
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

