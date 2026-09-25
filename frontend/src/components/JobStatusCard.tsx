import type { JSX } from 'react';
import { ProgressBar } from './ProgressBar';
import type { JobStatus } from '../types';
import type { JobTransport } from '../hooks/useJobProgress';

interface JobStatusCardProps {
  status: JobStatus;
  transport: JobTransport;
  onDelete?: (() => void) | undefined;
  onReset?: (() => void) | undefined;
}

const STATE_LABELS: Record<JobStatus['state'], string> = {
  queued: 'Queued',
  running: 'Processing',
  done: 'Complete',
  failed: 'Failed',
};

/** Shows a job's state, progress and controls. */
export function JobStatusCard({
  status,
  transport,
  onDelete,
  onReset,
}: JobStatusCardProps): JSX.Element {
  const isActive = status.state === 'queued' || status.state === 'running';

  return (
    <section className="card">
      <div className="card__header">
        <div>
          <h2 className="card__title">{status.filename}</h2>
          <p className="card__subtitle">
            Job {status.job_id.slice(0, 8)} &middot;{' '}
            {transport === 'polling' ? 'polling for updates' : 'live updates'}
          </p>
        </div>
        <span className={`badge badge--${status.state}`}>{STATE_LABELS[status.state]}</span>
      </div>

      {isActive ? (
        <ProgressBar
          value={status.progress}
          label={status.message ?? 'Processing'}
          indeterminate={status.state === 'queued'}
        />
      ) : null}

      {status.state === 'failed' && status.error ? (
        <p className="card__error" role="alert">
          {status.error}
        </p>
      ) : null}

      <div className="card__actions">
        {onReset ? (
          <button type="button" className="button" onClick={onReset}>
            Upload another
          </button>
        ) : null}
        {onDelete && !isActive ? (
          <button type="button" className="button button--ghost" onClick={onDelete}>
            Delete job
          </button>
        ) : null}
      </div>
    </section>
  );
}
