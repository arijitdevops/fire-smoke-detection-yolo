import type { JSX, MouseEvent } from 'react';

import type { DetectionSummary } from '../types';

interface DetectionTimelineProps {
  summary: DetectionSummary;
  /** Current playhead of the annotated video, in seconds. */
  currentTime?: number | undefined;
  /** Called with a timestamp when the user clicks a track or a marker. */
  onSeek?: ((seconds: number) => void) | undefined;
}

interface Marker {
  start: number;
  confidence: number;
}

/** Preferred row order; any other class the model reports is appended after these. */
const PREFERRED_ORDER = ['fire', 'smoke'];

function formatTime(value: number): string {
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`;
}

/** Group timeline entries by class, keeping the best confidence per sampled frame. */
export function buildTracks(summary: DetectionSummary): Map<string, Marker[]> {
  const tracks = new Map<string, Marker[]>();
  const names = [
    ...PREFERRED_ORDER.filter((name) => summary.per_class.some((item) => item.class_name === name)),
    ...summary.per_class
      .map((item) => item.class_name)
      .filter((name) => !PREFERRED_ORDER.includes(name)),
  ];
  for (const name of names) {
    tracks.set(name, []);
  }
  for (const frame of summary.timeline) {
    const best = new Map<string, number>();
    for (const detection of frame.detections) {
      best.set(detection.class_name, Math.max(best.get(detection.class_name) ?? 0, detection.confidence));
    }
    for (const [name, confidence] of best) {
      const track = tracks.get(name) ?? [];
      track.push({ start: frame.timestamp_seconds, confidence });
      tracks.set(name, track);
    }
  }
  return tracks;
}

/**
 * One horizontal track per class showing when it was detected.
 *
 * Each marker covers one sampled interval (`frame_stride / fps` seconds) and its
 * opacity follows the detection confidence. Clicking seeks the players.
 */
export function DetectionTimeline({ summary, currentTime, onSeek }: DetectionTimelineProps): JSX.Element {
  const duration = summary.duration_seconds > 0 ? summary.duration_seconds : 1;
  const step = summary.fps > 0 ? summary.frame_stride / summary.fps : 0;
  const tracks = buildTracks(summary);

  const seekFromClick = (event: MouseEvent<HTMLDivElement>): void => {
    if (!onSeek) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0;
    onSeek(Math.min(duration, Math.max(0, ratio * duration)));
  };

  return (
    <section className="card">
      <div className="card__header">
        <h2 className="card__title">Detection timeline</h2>
        <span className="card__subtitle">
          {formatTime(0)} to {formatTime(summary.duration_seconds)}
          {summary.timeline_truncated ? ' (first detections only, list truncated)' : ''}
        </span>
      </div>

      {tracks.size === 0 ? (
        <p className="card__note">Nothing was detected, so there is nothing to plot.</p>
      ) : (
        <div className="tracks">
          {[...tracks.entries()].map(([name, markers]) => (
            <div className="track" key={name}>
              <span className="track__label">
                <span className={`swatch swatch--${name.toLowerCase()}`} aria-hidden="true" />
                {name}
              </span>
              <div
                className="track__bar"
                role="group"
                aria-label={`${name} detections`}
                onClick={seekFromClick}
              >
                {markers.map((marker) => (
                  <button
                    key={marker.start}
                    type="button"
                    className={`track__marker track__marker--${name.toLowerCase()}`}
                    style={{
                      left: `${(marker.start / duration) * 100}%`,
                      width: `max(2px, ${(step / duration) * 100}%)`,
                      opacity: 0.35 + 0.65 * marker.confidence,
                    }}
                    title={`${name} ${(marker.confidence * 100).toFixed(0)}% at ${formatTime(marker.start)}`}
                    aria-label={`${name} at ${formatTime(marker.start)}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSeek?.(marker.start);
                    }}
                  />
                ))}
                {currentTime !== undefined ? (
                  <span
                    className="track__playhead"
                    style={{ left: `${Math.min(100, (currentTime / duration) * 100)}%` }}
                    aria-hidden="true"
                  />
                ) : null}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
