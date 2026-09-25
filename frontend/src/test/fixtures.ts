import type { DetectionSummary, HealthResponse, JobStatus } from '../types';

export function makeSummary(overrides: Partial<DetectionSummary> = {}): DetectionSummary {
  return {
    total_frames: 120,
    processed_frames: 40,
    frames_with_detections: 3,
    frames_with_fire: 2,
    frames_with_smoke: 1,
    total_detections: 4,
    peak_confidence: 0.91,
    peak_confidence_class: 'fire',
    first_detection_seconds: 1.2,
    fps: 10,
    width: 640,
    height: 480,
    duration_seconds: 12,
    frame_stride: 3,
    per_class: [
      { class_name: 'smoke', frames: 1, detections: 1, peak_confidence: 0.55, first_seen_seconds: 3.6 },
      { class_name: 'fire', frames: 2, detections: 3, peak_confidence: 0.91, first_seen_seconds: 1.2 },
    ],
    timeline: [
      {
        frame_index: 12,
        timestamp_seconds: 1.2,
        detections: [{ class_id: 1, class_name: 'fire', confidence: 0.91, xyxy: [0, 0, 10, 10] }],
      },
      {
        frame_index: 36,
        timestamp_seconds: 3.6,
        detections: [
          { class_id: 1, class_name: 'fire', confidence: 0.7, xyxy: [0, 0, 10, 10] },
          { class_id: 1, class_name: 'fire', confidence: 0.4, xyxy: [5, 5, 20, 20] },
          { class_id: 0, class_name: 'smoke', confidence: 0.55, xyxy: [0, 0, 30, 30] },
        ],
      },
    ],
    timeline_truncated: false,
    codec: 'h264',
    ...overrides,
  };
}

export function makeJob(overrides: Partial<JobStatus> = {}): JobStatus {
  return {
    job_id: 'abc123def456',
    kind: 'video',
    state: 'done',
    progress: 100,
    message: 'complete',
    error: null,
    filename: 'forest.mp4',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:10Z',
    expires_at: null,
    download_url: '/api/jobs/abc123def456/download',
    summary: makeSummary(),
    ...overrides,
  };
}

export function makeHealth(overrides: Partial<HealthResponse> = {}): HealthResponse {
  return {
    status: 'ok',
    version: '0.1.0',
    model_loaded: true,
    model_path: 'runs/detect/fire_smoke/weights/best.pt',
    model_source: 'trained',
    device: 'auto',
    class_names: ['smoke', 'fire'],
    detail: null,
    ...overrides,
  };
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
