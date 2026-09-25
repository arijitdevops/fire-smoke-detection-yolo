import type { JSX } from 'react';
import type { ClassSummary, DetectionSummary as Summary } from '../types';

interface DetectionSummaryProps {
  summary: Summary;
  /** When provided, timeline timestamps become buttons that seek the players. */
  onSeek?: ((seconds: number) => void) | undefined;
}

function formatSeconds(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return '--';
  }
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`;
}

function firstSeen(summary: Summary, className: string): number | null {
  return summary.per_class.find((item) => item.class_name === className)?.first_seen_seconds ?? null;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }): JSX.Element {
  return (
    <div className="tile">
      <span className="tile__label">{label}</span>
      <span className="tile__value">{value}</span>
      {note ? <span className="tile__note">{note}</span> : null}
    </div>
  );
}

function ClassRow({ item }: { item: ClassSummary }): JSX.Element {
  return (
    <tr>
      <th scope="row">
        <span className={`swatch swatch--${item.class_name.toLowerCase()}`} aria-hidden="true" />
        {item.class_name}
      </th>
      <td>{item.frames.toLocaleString()}</td>
      <td>{item.detections.toLocaleString()}</td>
      <td>{formatPercent(item.peak_confidence)}</td>
      <td>{formatSeconds(item.first_seen_seconds)}</td>
    </tr>
  );
}

/** Stat tiles plus a per-class breakdown of an annotated clip. */
export function DetectionSummary({ summary, onSeek }: DetectionSummaryProps): JSX.Element {
  const detectionRate =
    summary.processed_frames > 0
      ? summary.frames_with_detections / summary.processed_frames
      : 0;

  return (
    <section className="card">
      <h2 className="card__title">Detection summary</h2>
      <div className="tiles">
        <Tile
          label="Frames analysed"
          value={summary.processed_frames.toLocaleString()}
          note={`every ${summary.frame_stride} of ${summary.total_frames.toLocaleString()} frames`}
        />
        <Tile
          label="Frames with detections"
          value={summary.frames_with_detections.toLocaleString()}
          note={`${formatPercent(detectionRate)} of analysed frames`}
        />
        <Tile
          label="Peak confidence"
          value={formatPercent(summary.peak_confidence)}
          note={summary.peak_confidence_class ?? 'no detections'}
        />
        <Tile
          label="Frames with fire"
          value={summary.frames_with_fire.toLocaleString()}
          note={`first at ${formatSeconds(firstSeen(summary, 'fire'))}`}
        />
        <Tile
          label="Frames with smoke"
          value={summary.frames_with_smoke.toLocaleString()}
          note={`first at ${formatSeconds(firstSeen(summary, 'smoke'))}`}
        />
        <Tile
          label="First detection"
          value={formatSeconds(summary.first_detection_seconds)}
          note={`clip length ${formatSeconds(summary.duration_seconds)}`}
        />
      </div>

      <table className="table">
        <caption className="table__caption">
          Per-class totals over the analysed frames ({summary.width}&times;{summary.height} at{' '}
          {summary.fps.toFixed(1)} fps)
        </caption>
        <thead>
          <tr>
            <th scope="col">Class</th>
            <th scope="col">Frames</th>
            <th scope="col">Boxes</th>
            <th scope="col">Peak conf.</th>
            <th scope="col">First seen</th>
          </tr>
        </thead>
        <tbody>
          {summary.per_class.length > 0 ? (
            summary.per_class.map((item) => <ClassRow key={item.class_name} item={item} />)
          ) : (
            <tr>
              <td colSpan={5} className="table__empty">
                No fire or smoke was detected above the configured confidence threshold.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {summary.timeline.length > 0 ? (
        <details className="timeline">
          <summary>
            Detection timeline ({summary.timeline.length.toLocaleString()} frames
            {summary.timeline_truncated ? ', truncated' : ''})
          </summary>
          <ul className="timeline__list">
            {summary.timeline.slice(0, 40).map((frame) => (
              <li key={frame.frame_index}>
                <span className="timeline__time">
                  {onSeek ? (
                    <button
                      type="button"
                      className="timeline__seek"
                      onClick={() => onSeek(frame.timestamp_seconds)}
                    >
                      {formatSeconds(frame.timestamp_seconds)}
                    </button>
                  ) : (
                    formatSeconds(frame.timestamp_seconds)
                  )}
                </span>
                <span className="timeline__classes">
                  {frame.detections
                    .map((item) => `${item.class_name} ${formatPercent(item.confidence)}`)
                    .join(', ')}
                </span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
