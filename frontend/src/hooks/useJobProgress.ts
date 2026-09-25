/**
 * Subscribe to a video job's progress.
 *
 * The hook prefers the Server-Sent Events stream and automatically falls back to
 * polling `GET /api/jobs/{id}` when the stream cannot be opened (an old proxy, a
 * corporate filter, or a backend restart). Every listener and timer is cleaned
 * up when the job id changes or the component unmounts.
 */

import { useEffect, useRef, useState } from 'react';

import { ApiRequestError, fetchJob, subscribeToJob } from '../api/client';
import type { JobStatus } from '../types';

/** Transport currently delivering updates. */
export type JobTransport = 'idle' | 'stream' | 'polling';

/** Value returned by {@link useJobProgress}. */
export interface JobProgressState {
  status: JobStatus | null;
  error: ApiRequestError | null;
  transport: JobTransport;
}

const DEFAULT_POLL_INTERVAL_MS = 1500;

function isTerminal(status: JobStatus | null): boolean {
  return status?.state === 'done' || status?.state === 'failed';
}

export function useJobProgress(
  jobId: string | null,
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
): JobProgressState {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<ApiRequestError | null>(null);
  const [transport, setTransport] = useState<JobTransport>('idle');
  const finishedRef = useRef(false);

  useEffect(() => {
    if (!jobId) {
      setStatus(null);
      setError(null);
      setTransport('idle');
      return;
    }

    let cancelled = false;
    let pollTimer: number | undefined;
    let unsubscribe: (() => void) | null = null;
    const controller = new AbortController();
    finishedRef.current = false;

    const stop = (): void => {
      cancelled = true;
      if (pollTimer !== undefined) {
        window.clearTimeout(pollTimer);
        pollTimer = undefined;
      }
      if (unsubscribe) {
        unsubscribe();
        unsubscribe = null;
      }
      controller.abort();
    };

    const poll = async (): Promise<void> => {
      if (cancelled) {
        return;
      }
      try {
        const next = await fetchJob(jobId, controller.signal);
        if (cancelled) {
          return;
        }
        setStatus(next);
        setError(null);
        if (isTerminal(next)) {
          finishedRef.current = true;
          return;
        }
      } catch (cause) {
        if (cancelled) {
          return;
        }
        if (cause instanceof ApiRequestError) {
          setError(cause);
          // A 404 will never recover: stop hammering the API.
          if (cause.status === 404) {
            return;
          }
        }
      }
      pollTimer = window.setTimeout(() => {
        void poll();
      }, pollIntervalMs);
    };

    const startPolling = (): void => {
      if (cancelled || finishedRef.current) {
        return;
      }
      setTransport('polling');
      void poll();
    };

    // Fetch once immediately so the card is populated before the first event.
    void fetchJob(jobId, controller.signal)
      .then((initial) => {
        if (!cancelled) {
          setStatus(initial);
          if (isTerminal(initial)) {
            finishedRef.current = true;
          }
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled && cause instanceof ApiRequestError && cause.status !== 0) {
          setError(cause);
        }
      });

    if (typeof window.EventSource === 'undefined') {
      startPolling();
    } else {
      setTransport('stream');
      unsubscribe = subscribeToJob(jobId, {
        onProgress: (next) => {
          if (!cancelled) {
            setStatus(next);
            setError(null);
          }
        },
        onFinished: (next) => {
          if (!cancelled) {
            finishedRef.current = true;
            setStatus(next);
            setTransport('idle');
          }
        },
        onStreamError: () => {
          if (!cancelled && !finishedRef.current) {
            startPolling();
          }
        },
      });
    }

    return stop;
  }, [jobId, pollIntervalMs]);

  return { status, error, transport };
}
