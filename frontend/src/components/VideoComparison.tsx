import { type JSX, useCallback, useEffect, useRef, useState } from 'react';

/** A request to move both players to `time` seconds; `nonce` makes repeats distinct. */
export interface SeekRequest {
  time: number;
  nonce: number;
}

interface VideoComparisonProps {
  /** Object URL of the file the user uploaded, or null when unavailable. */
  originalSrc: string | null;
  /** URL of the annotated MP4 served by the API. */
  annotatedSrc: string;
  /** Name suggested to the browser when downloading. */
  downloadName: string;
  /** Codec reported by the backend; anything other than h264 may not play inline. */
  codec?: string | undefined;
  /** External seek requests, e.g. from the detection timeline. */
  seek?: SeekRequest | null | undefined;
  /** Reports the annotated player's playhead in seconds. */
  onTimeUpdate?: ((seconds: number) => void) | undefined;
}

/** Tolerated drift between the two players before the original is re-synced. */
const MAX_DRIFT_SECONDS = 0.25;

/**
 * Plays the original upload and the annotated result side by side.
 *
 * The annotated player is the leader: play, pause, seek and rate changes are
 * mirrored onto the original so both clips stay in step.
 */
export function VideoComparison({
  originalSrc,
  annotatedSrc,
  downloadName,
  codec,
  seek,
  onTimeUpdate,
}: VideoComparisonProps): JSX.Element {
  const annotatedRef = useRef<HTMLVideoElement>(null);
  const originalRef = useRef<HTMLVideoElement>(null);
  const [originalFailed, setOriginalFailed] = useState(false);
  const [annotatedFailed, setAnnotatedFailed] = useState(false);

  useEffect(() => {
    setOriginalFailed(false);
  }, [originalSrc]);

  useEffect(() => {
    setAnnotatedFailed(false);
  }, [annotatedSrc]);

  const syncOriginal = useCallback((): void => {
    const leader = annotatedRef.current;
    const follower = originalRef.current;
    if (!leader || !follower || originalFailed) {
      return;
    }
    if (Math.abs(follower.currentTime - leader.currentTime) > MAX_DRIFT_SECONDS) {
      follower.currentTime = leader.currentTime;
    }
    follower.playbackRate = leader.playbackRate;
    if (leader.paused && !follower.paused) {
      follower.pause();
    } else if (!leader.paused && follower.paused) {
      void follower.play().catch(() => undefined);
    }
  }, [originalFailed]);

  useEffect(() => {
    if (!seek) {
      return;
    }
    const leader = annotatedRef.current;
    if (leader) {
      leader.currentTime = seek.time;
    }
    const follower = originalRef.current;
    if (follower && !originalFailed) {
      follower.currentTime = seek.time;
    }
  }, [seek, originalFailed]);

  const showOriginal = originalSrc !== null && !originalFailed;

  return (
    <section className="card">
      <div className="card__header">
        <h2 className="card__title">Original vs annotated</h2>
        <a className="button button--primary" href={annotatedSrc} download={downloadName}>
          Download MP4
        </a>
      </div>

      <div className="compare">
        <figure className="compare__pane">
          <figcaption className="compare__label">Original</figcaption>
          {showOriginal ? (
            <video
              ref={originalRef}
              className="player"
              src={originalSrc}
              muted
              playsInline
              preload="metadata"
              aria-label="Original upload"
              onError={() => setOriginalFailed(true)}
            />
          ) : (
            <div className="compare__placeholder">
              {originalSrc === null
                ? 'The original is only available in the tab it was uploaded from.'
                : 'Your browser cannot preview this container format (e.g. AVI/MKV). The annotated MP4 plays normally.'}
            </div>
          )}
        </figure>

        <figure className="compare__pane">
          <figcaption className="compare__label">Annotated</figcaption>
          <video
            ref={annotatedRef}
            className="player"
            src={annotatedSrc}
            controls
            playsInline
            preload="metadata"
            aria-label="Annotated video"
            onPlay={syncOriginal}
            onPause={syncOriginal}
            onSeeked={syncOriginal}
            onRateChange={syncOriginal}
            onTimeUpdate={(event) => {
              syncOriginal();
              onTimeUpdate?.(event.currentTarget.currentTime);
            }}
            onError={() => setAnnotatedFailed(true)}
          >
            Your browser cannot play this video. Use the download button instead.
          </video>
          {annotatedFailed || (codec !== undefined && codec !== 'h264') ? (
            <p className="card__error" role="alert">
              The annotated file was encoded as {codec ?? 'an unknown codec'}, which browsers may
              not play. Install <code>imageio-ffmpeg</code> on the backend for H.264 output, or
              download the file and open it in a desktop player.
            </p>
          ) : null}
        </figure>
      </div>

      <p className="card__note">
        Use the controls on the annotated video; the original follows it. Boxes are drawn on every
        frame, while detection runs on a sampled subset, so overlays persist between sampled
        frames.
      </p>
    </section>
  );
}
