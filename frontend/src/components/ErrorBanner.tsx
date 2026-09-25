import type { JSX } from 'react';
interface ErrorBannerProps {
  title: string;
  message: string;
  hint?: string | undefined;
  tone?: 'error' | 'warning';
  onDismiss?: (() => void) | undefined;
}

/** Inline banner used for API failures and for the "no weights" state. */
export function ErrorBanner({
  title,
  message,
  hint,
  tone = 'error',
  onDismiss,
}: ErrorBannerProps): JSX.Element {
  return (
    <div className={`banner banner--${tone}`} role="alert">
      <div className="banner__body">
        <p className="banner__title">{title}</p>
        <p className="banner__message">{message}</p>
        {hint ? <p className="banner__hint">{hint}</p> : null}
      </div>
      {onDismiss ? (
        <button type="button" className="banner__close" onClick={onDismiss} aria-label="Dismiss">
          &times;
        </button>
      ) : null}
    </div>
  );
}
