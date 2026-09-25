/**
 * TypeScript mirrors of the pydantic schemas in `backend/app/schemas.py`.
 * Keep both files in sync when the API changes.
 */

export type JobState = 'queued' | 'running' | 'done' | 'failed';
export type JobKind = 'video';

/** A single detected object, in absolute pixel coordinates. */
export interface Detection {
  /** 0 = smoke, 1 = fire. */
  class_id: number;
  class_name: string;
  /** Detector confidence in [0, 1]. */
  confidence: number;
  /** [x1, y1, x2, y2]. */
  xyxy: [number, number, number, number] | number[];
}

/** Detections for one sampled video frame. */
export interface FrameDetection {
  frame_index: number;
  timestamp_seconds: number;
  detections: Detection[];
}

/** Aggregated statistics for a single class across a clip. */
export interface ClassSummary {
  class_name: string;
  frames: number;
  detections: number;
  peak_confidence: number;
  first_seen_seconds: number | null;
}

/** Everything needed to describe a processed video at a glance. */
export interface DetectionSummary {
  total_frames: number;
  processed_frames: number;
  frames_with_detections: number;
  frames_with_fire: number;
  frames_with_smoke: number;
  total_detections: number;
  peak_confidence: number;
  peak_confidence_class: string | null;
  first_detection_seconds: number | null;
  fps: number;
  width: number;
  height: number;
  duration_seconds: number;
  frame_stride: number;
  per_class: ClassSummary[];
  timeline: FrameDetection[];
  timeline_truncated: boolean;
  /** 'h264' (plays in browsers) or 'mp4v' (fallback when ffmpeg is unavailable). */
  codec?: string;
}

/** Response of `POST /api/detect/image`. */
export interface ImageDetectionResponse {
  filename: string;
  width: number;
  height: number;
  detections: Detection[];
  class_counts: Record<string, number>;
  conf_threshold: number;
  iou_threshold: number;
  inference_ms: number;
  /** Annotated image as a `data:image/jpeg;base64,...` URI. */
  annotated_image: string;
}

/** 202 response of `POST /api/detect/video`. */
export interface JobCreated {
  job_id: string;
  state: JobState;
  kind: JobKind;
  filename: string;
  created_at: string;
  status_url: string;
  stream_url: string;
}

/** Response of `GET /api/jobs/{id}`. */
export interface JobStatus {
  job_id: string;
  kind: JobKind;
  state: JobState;
  progress: number;
  message: string | null;
  error: string | null;
  filename: string;
  created_at: string;
  updated_at: string;
  expires_at: string | null;
  download_url: string | null;
  summary: DetectionSummary | null;
}

/** Response of `GET /api/health`. */
export interface HealthResponse {
  status: 'ok' | 'degraded';
  version: string;
  model_loaded: boolean;
  model_path: string;
  model_source: string | null;
  device: string;
  class_names: string[];
  detail: string | null;
}

/** Response of `DELETE /api/jobs/{id}`. */
export interface JobDeleted {
  job_id: string;
  deleted: boolean;
}

/** Uniform error envelope returned by every non-2xx response. */
export interface ErrorResponse {
  error: string;
  message: string;
  detail: Record<string, unknown>;
}
