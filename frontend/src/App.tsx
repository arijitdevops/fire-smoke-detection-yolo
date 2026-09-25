import { type JSX, useCallback, useEffect, useState } from 'react';

import { ApiRequestError, deleteJob, fetchHealth, jobDownloadUrl, uploadVideo } from './api/client';
import { DetectionSummary } from './components/DetectionSummary';
import { DetectionTimeline } from './components/DetectionTimeline';
import { ErrorBanner } from './components/ErrorBanner';
import { Header } from './components/Header';
import { ImageDetect } from './components/ImageDetect';
import { JobStatusCard } from './components/JobStatusCard';
import { UploadDropzone } from './components/UploadDropzone';
import { type SeekRequest, VideoComparison } from './components/VideoComparison';
import { useJobProgress } from './hooks/useJobProgress';
import type { HealthResponse } from './types';

/** Mirrors ALLOWED_IMAGE_EXTENSIONS in backend/.env.example. */
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];

/** Mirrors ALLOWED_VIDEO_EXTENSIONS in backend/.env.example. */
const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];

/** Mirrors MAX_UPLOAD_MB in backend/.env.example; the server enforces the real limit. */
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

type Tab = 'video' | 'image';

export default function App(): JSX.Element {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<ApiRequestError | null>(null);
  const [tab, setTab] = useState<Tab>('video');
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<ApiRequestError | null>(null);
  const [isUploading, setUploading] = useState(false);
  /** Object URL of the uploaded file, shown next to the annotated result. */
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [seek, setSeek] = useState<SeekRequest | null>(null);
  const [currentTime, setCurrentTime] = useState(0);

  const { status, error: progressError, transport } = useJobProgress(jobId);

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then((next) => {
        setHealth(next);
        setHealthError(null);
      })
      .catch((cause: unknown) => {
        // An aborted request is a normal unmount, not a failure worth showing.
        if (!controller.signal.aborted && cause instanceof ApiRequestError) {
          setHealthError(cause);
        }
      });
    return () => controller.abort();
  }, []);

  // Release the previous object URL whenever it is replaced or the app unmounts.
  useEffect(() => {
    return () => {
      if (originalUrl) {
        URL.revokeObjectURL(originalUrl);
      }
    };
  }, [originalUrl]);

  const resetJob = useCallback((): void => {
    setJobId(null);
    setOriginalUrl(null);
    setSeek(null);
    setCurrentTime(0);
  }, []);

  const seekTo = useCallback((time: number): void => {
    setSeek({ time, nonce: Date.now() });
  }, []);

  const onVideoSelected = useCallback(async (file: File): Promise<void> => {
    setUploading(true);
    setUploadError(null);
    try {
      const created = await uploadVideo(file);
      setOriginalUrl(URL.createObjectURL(file));
      setCurrentTime(0);
      setJobId(created.job_id);
    } catch (cause) {
      setJobId(null);
      setUploadError(
        cause instanceof ApiRequestError
          ? cause
          : new ApiRequestError('the upload failed unexpectedly', 0, 'unknown'),
      );
    } finally {
      setUploading(false);
    }
  }, []);

  const onDeleteJob = useCallback(async (): Promise<void> => {
    if (!jobId) {
      return;
    }
    try {
      await deleteJob(jobId);
    } catch (cause) {
      if (cause instanceof ApiRequestError) {
        setUploadError(cause);
      }
    } finally {
      resetJob();
    }
  }, [jobId, resetJob]);

  const modelUnavailable = health !== null && !health.model_loaded;
  const usingFallback = health?.model_source === 'coco-fallback';

  return (
    <div className="app">
      <Header health={health} />

      <main className="main">
        {healthError ? (
          <ErrorBanner
            title="Cannot reach the API"
            message={healthError.message}
            hint="Start the backend with: uvicorn app.main:app --port 8000"
            onDismiss={() => setHealthError(null)}
          />
        ) : null}

        {modelUnavailable ? (
          <ErrorBanner
            tone="warning"
            title="No detector weights loaded"
            message={health?.detail ?? 'The API has no weights to run inference with.'}
            hint={`Train a model (python training/train.py) and point MODEL_PATH at the result, currently ${health?.model_path ?? 'unset'}.`}
          />
        ) : null}

        {usingFallback ? (
          <ErrorBanner
            tone="warning"
            title="Pretrained COCO weights in use"
            message="The API fell back to a generic COCO checkpoint so the pipeline can be exercised end to end."
            hint="It will not detect fire or smoke until you train on the dataset and set MODEL_PATH."
          />
        ) : null}

        <nav className="tabs" aria-label="Detection mode">
          <button
            type="button"
            className={`tab${tab === 'video' ? ' tab--active' : ''}`}
            onClick={() => setTab('video')}
            aria-pressed={tab === 'video'}
          >
            Video
          </button>
          <button
            type="button"
            className={`tab${tab === 'image' ? ' tab--active' : ''}`}
            onClick={() => setTab('image')}
            aria-pressed={tab === 'image'}
          >
            Image
          </button>
        </nav>

        {tab === 'image' ? (
          <ImageDetect
            accept={IMAGE_EXTENSIONS}
            maxBytes={MAX_UPLOAD_BYTES}
            modelUnavailable={modelUnavailable}
          />
        ) : (
          <div className="stack">
            {jobId === null ? (
              <UploadDropzone
                accept={VIDEO_EXTENSIONS}
                maxBytes={MAX_UPLOAD_BYTES}
                disabled={isUploading || modelUnavailable}
                label={isUploading ? 'Uploading...' : 'Drop a video here'}
                hint="or click to choose a file. Processing runs in the background and streams progress back to this page."
                onFileSelected={(file) => {
                  void onVideoSelected(file);
                }}
              />
            ) : null}

            {uploadError ? (
              <ErrorBanner
                title={uploadError.isModelUnavailable ? 'Model not loaded' : 'Upload failed'}
                message={uploadError.message}
                hint={
                  typeof uploadError.detail.hint === 'string' ? uploadError.detail.hint : undefined
                }
                tone={uploadError.isModelUnavailable ? 'warning' : 'error'}
                onDismiss={() => setUploadError(null)}
              />
            ) : null}

            {progressError ? (
              <ErrorBanner
                title="Lost track of the job"
                message={progressError.message}
                onDismiss={resetJob}
              />
            ) : null}

            {status ? (
              <JobStatusCard
                status={status}
                transport={transport}
                onReset={resetJob}
                onDelete={() => {
                  void onDeleteJob();
                }}
              />
            ) : null}

            {status?.state === 'done' && status.download_url ? (
              <VideoComparison
                originalSrc={originalUrl}
                annotatedSrc={jobDownloadUrl(status.job_id)}
                downloadName={`annotated_${status.filename.replace(/\.[^.]+$/, '')}.mp4`}
                codec={status.summary?.codec}
                seek={seek}
                onTimeUpdate={setCurrentTime}
              />
            ) : null}

            {status?.state === 'done' && status.summary ? (
              <DetectionTimeline
                summary={status.summary}
                currentTime={currentTime}
                onSeek={seekTo}
              />
            ) : null}

            {status?.summary ? <DetectionSummary summary={status.summary} onSeek={seekTo} /> : null}
          </div>
        )}
      </main>

      <footer className="footer">
        <span>
          Model: {health?.model_source ?? 'none'} &middot; device {health?.device ?? 'unknown'}
        </span>
        <a href="/docs" target="_blank" rel="noreferrer">
          API documentation
        </a>
      </footer>
    </div>
  );
}
