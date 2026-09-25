import { type JSX, useCallback, useState } from 'react';

import { ApiRequestError, detectImage } from '../api/client';
import type { ImageDetectionResponse } from '../types';
import { ErrorBanner } from './ErrorBanner';
import { UploadDropzone } from './UploadDropzone';

interface ImageDetectProps {
  accept: string[];
  maxBytes: number;
  /** True when the backend reports that no weights are loaded. */
  modelUnavailable: boolean;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Still-image tab: upload a picture and inspect the annotated result. */
export function ImageDetect({ accept, maxBytes, modelUnavailable }: ImageDetectProps): JSX.Element {
  const [result, setResult] = useState<ImageDetectionResponse | null>(null);
  const [error, setError] = useState<ApiRequestError | null>(null);
  const [isBusy, setBusy] = useState(false);

  const onFileSelected = useCallback(async (file: File): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      setResult(await detectImage(file));
    } catch (cause) {
      setResult(null);
      setError(
        cause instanceof ApiRequestError
          ? cause
          : new ApiRequestError('detection failed unexpectedly', 0, 'unknown'),
      );
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <div className="stack">
      <UploadDropzone
        accept={accept}
        maxBytes={maxBytes}
        disabled={isBusy || modelUnavailable}
        label={isBusy ? 'Running detection...' : 'Drop an image here'}
        hint="or click to choose a file. Detection runs immediately and returns boxes plus an annotated preview."
        onFileSelected={(file) => {
          void onFileSelected(file);
        }}
      />

      {error ? (
        <ErrorBanner
          title={error.isModelUnavailable ? 'Model not loaded' : 'Detection failed'}
          message={error.message}
          hint={typeof error.detail.hint === 'string' ? error.detail.hint : undefined}
          tone={error.isModelUnavailable ? 'warning' : 'error'}
          onDismiss={() => setError(null)}
        />
      ) : null}

      {result ? (
        <section className="card">
          <div className="card__header">
            <div>
              <h2 className="card__title">{result.filename}</h2>
              <p className="card__subtitle">
                {result.width}&times;{result.height} &middot; {result.inference_ms.toFixed(0)} ms
                &middot; conf &ge; {formatPercent(result.conf_threshold)}
              </p>
            </div>
            <a className="button" href={result.annotated_image} download={`annotated_${result.filename}`}>
              Download
            </a>
          </div>

          <img className="preview" src={result.annotated_image} alt={`Detections for ${result.filename}`} />

          {result.detections.length > 0 ? (
            <table className="table">
              <caption className="table__caption">
                {result.detections.length} detection(s):{' '}
                {Object.entries(result.class_counts)
                  .map(([name, count]) => `${count} ${name}`)
                  .join(', ')}
              </caption>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Class</th>
                  <th scope="col">Confidence</th>
                  <th scope="col">Box (x1, y1, x2, y2)</th>
                </tr>
              </thead>
              <tbody>
                {result.detections.map((detection, index) => (
                  <tr key={`${detection.class_name}-${index}`}>
                    <td>{index + 1}</td>
                    <th scope="row">
                      <span
                        className={`swatch swatch--${detection.class_name.toLowerCase()}`}
                        aria-hidden="true"
                      />
                      {detection.class_name}
                    </th>
                    <td>{formatPercent(detection.confidence)}</td>
                    <td className="mono">
                      {detection.xyxy.map((value) => Math.round(value)).join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="card__note">
              Nothing was detected above the confidence threshold. Lower CONF_THRESHOLD on the
              backend to see weaker candidates.
            </p>
          )}
        </section>
      ) : null}
    </div>
  );
}
