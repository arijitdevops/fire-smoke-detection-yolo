import type { JSX } from 'react';
import type { HealthResponse } from '../types';

interface HeaderProps {
  health: HealthResponse | null;
}

/** Application header with a compact model-status pill. */
export function Header({ health }: HeaderProps): JSX.Element {
  const state = health === null ? 'unknown' : health.model_loaded ? 'ready' : 'offline';
  const labels: Record<typeof state, string> = {
    unknown: 'checking model',
    ready: health?.model_source === 'coco-fallback' ? 'COCO fallback' : 'model ready',
    offline: 'no model',
  };

  return (
    <header className="header">
      <div className="header__brand">
        <span className="header__mark" aria-hidden="true" />
        <div>
          <h1 className="header__title">Fire &amp; Smoke Detection</h1>
          <p className="header__subtitle">YOLO inference for still images and video</p>
        </div>
      </div>
      <div className={`pill pill--${state}`} title={health?.model_path ?? ''}>
        <span className="pill__dot" aria-hidden="true" />
        {labels[state]}
        {health ? <span className="pill__meta">v{health.version}</span> : null}
      </div>
    </header>
  );
}
