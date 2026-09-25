import type { JSX } from 'react';
interface ProgressBarProps {
  /** Completion in percent, 0-100. */
  value: number;
  label?: string | undefined;
  /** Render as an indeterminate bar when the total is unknown. */
  indeterminate?: boolean;
}

/** Accessible progress bar used while a video job runs. */
export function ProgressBar({ value, label, indeterminate = false }: ProgressBarProps): JSX.Element {
  const clamped = Math.min(100, Math.max(0, Number.isFinite(value) ? value : 0));

  return (
    <div className="progress">
      <div className="progress__meta">
        <span>{label ?? 'Processing'}</span>
        <span className="progress__value">{indeterminate ? '' : `${clamped.toFixed(0)}%`}</span>
      </div>
      <div
        className={`progress__track${indeterminate ? ' progress__track--indeterminate' : ''}`}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(clamped)}
        aria-label={label ?? 'Processing'}
      >
        <div className="progress__fill" style={{ width: indeterminate ? '35%' : `${clamped}%` }} />
      </div>
    </div>
  );
}
