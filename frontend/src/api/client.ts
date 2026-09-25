/**
 * Typed wrapper around the detection API.
 *
 * Every call returns parsed JSON or throws an {@link ApiRequestError} carrying
 * the backend's stable error code, so components can branch on
 * `error.code === 'model_unavailable'` instead of matching on message text.
 */

import type {
  ErrorResponse,
  HealthResponse,
  ImageDetectionResponse,
  JobCreated,
  JobDeleted,
  JobStatus,
} from '../types';

/** Base URL of the API; empty string means "same origin" (dev proxy / nginx). */
export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

/** Error thrown for any non-2xx response or transport failure. */
export class ApiRequestError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: Record<string, unknown>;

  constructor(message: string, status: number, code: string, detail: Record<string, unknown> = {}) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
    this.code = code;
    this.detail = detail;
  }

  /** True when the backend has no usable weights loaded. */
  get isModelUnavailable(): boolean {
    return this.code === 'model_unavailable';
  }
}

function url(path: string): string {
  return `${API_BASE_URL}${path}`;
}

function isErrorResponse(value: unknown): value is ErrorResponse {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as ErrorResponse).error === 'string' &&
    typeof (value as ErrorResponse).message === 'string'
  );
}

async function toApiError(response: Response): Promise<ApiRequestError> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    // Non-JSON error bodies (proxy errors, HTML pages) fall through.
  }
  if (isErrorResponse(payload)) {
    return new ApiRequestError(payload.message, response.status, payload.error, payload.detail ?? {});
  }
  return new ApiRequestError(
    `request failed with status ${response.status}`,
    response.status,
    'http_error',
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url(path), init);
  } catch (cause) {
    throw new ApiRequestError(
      'could not reach the detection API; is the backend running?',
      0,
      'network_error',
      { cause: String(cause) },
    );
  }
  if (!response.ok) {
    throw await toApiError(response);
  }
  try {
    return (await response.json()) as T;
  } catch (cause) {
    throw new ApiRequestError('the API returned a malformed response', response.status, 'bad_json', {
      cause: String(cause),
    });
  }
}

/** Fetch service and model status. */
export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health', signal ? { signal } : undefined);
}

/** Options accepted by {@link detectImage}. */
export interface DetectImageOptions {
  conf?: number;
  iou?: number;
  signal?: AbortSignal;
}

/** Run synchronous detection on a still image. */
export function detectImage(file: File, options: DetectImageOptions = {}): Promise<ImageDetectionResponse> {
  const body = new FormData();
  body.append('file', file, file.name);
  if (options.conf !== undefined) {
    body.append('conf', String(options.conf));
  }
  if (options.iou !== undefined) {
    body.append('iou', String(options.iou));
  }
  const init: RequestInit = { method: 'POST', body };
  if (options.signal) {
    init.signal = options.signal;
  }
  return request<ImageDetectionResponse>('/api/detect/image', init);
}

/** Upload a video and queue an annotation job. */
export function uploadVideo(file: File, signal?: AbortSignal): Promise<JobCreated> {
  const body = new FormData();
  body.append('file', file, file.name);
  const init: RequestInit = { method: 'POST', body };
  if (signal) {
    init.signal = signal;
  }
  return request<JobCreated>('/api/detect/video', init);
}

/** Read the current state of a job. */
export function fetchJob(jobId: string, signal?: AbortSignal): Promise<JobStatus> {
  return request<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`, signal ? { signal } : undefined);
}

/** Delete a job together with its annotated output. */
export function deleteJob(jobId: string): Promise<JobDeleted> {
  return request<JobDeleted>(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
}

/** Absolute URL of a finished job's annotated MP4. */
export function jobDownloadUrl(jobId: string): string {
  return url(`/api/jobs/${encodeURIComponent(jobId)}/download`);
}

/** Callbacks accepted by {@link subscribeToJob}. */
export interface JobStreamHandlers {
  onProgress: (status: JobStatus) => void;
  onFinished: (status: JobStatus) => void;
  /** Called when the stream itself fails; the caller should fall back to polling. */
  onStreamError: () => void;
}

/**
 * Subscribe to a job's Server-Sent Events stream.
 *
 * @returns An unsubscribe function that closes the underlying `EventSource`.
 */
export function subscribeToJob(jobId: string, handlers: JobStreamHandlers): () => void {
  const source = new EventSource(url(`/api/jobs/${encodeURIComponent(jobId)}/stream`));
  let closed = false;

  const close = (): void => {
    if (!closed) {
      closed = true;
      source.close();
    }
  };

  const parse = (event: MessageEvent<string>): JobStatus | null => {
    try {
      return JSON.parse(event.data) as JobStatus;
    } catch {
      return null;
    }
  };

  source.addEventListener('progress', (event) => {
    const status = parse(event as MessageEvent<string>);
    if (status) {
      handlers.onProgress(status);
    }
  });

  const finish = (event: Event): void => {
    const status = parse(event as MessageEvent<string>);
    if (status) {
      handlers.onFinished(status);
    }
    close();
  };

  source.addEventListener('done', finish);
  source.addEventListener('failed', finish);

  source.onerror = () => {
    // EventSource retries on its own; we close and let the caller poll instead.
    if (!closed) {
      close();
      handlers.onStreamError();
    }
  };

  return close;
}
