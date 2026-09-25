import { type JSX, useCallback, useId, useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react';

interface UploadDropzoneProps {
  /** Allowed extensions including the dot, e.g. ['.mp4', '.mov']. */
  accept: string[];
  /** Client-side size cap in bytes; the server enforces its own limit too. */
  maxBytes: number;
  label: string;
  hint: string;
  disabled?: boolean;
  onFileSelected: (file: File) => void;
}

function extensionOf(name: string): string {
  const index = name.lastIndexOf('.');
  return index >= 0 ? name.slice(index).toLowerCase() : '';
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  }
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** Drag-and-drop area with a file picker and client-side validation. */
export function UploadDropzone({
  accept,
  maxBytes,
  label,
  hint,
  disabled = false,
  onFileSelected,
}: UploadDropzoneProps): JSX.Element {
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();
  const [isDragging, setDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const validate = useCallback(
    (file: File): string | null => {
      const extension = extensionOf(file.name);
      if (!accept.includes(extension)) {
        return `"${extension || file.name}" is not supported. Allowed: ${accept.join(', ')}.`;
      }
      if (file.size === 0) {
        return 'That file is empty.';
      }
      if (file.size > maxBytes) {
        return `That file is ${formatBytes(file.size)}; the limit is ${formatBytes(maxBytes)}.`;
      }
      return null;
    },
    [accept, maxBytes],
  );

  const handleFile = useCallback(
    (file: File | undefined): void => {
      if (!file) {
        return;
      }
      const problem = validate(file);
      setValidationError(problem);
      if (!problem) {
        onFileSelected(file);
      }
    },
    [onFileSelected, validate],
  );

  const onDrop = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    setDragging(false);
    if (disabled) {
      return;
    }
    handleFile(event.dataTransfer.files?.[0]);
  };

  const onDragOver = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    if (!disabled) {
      setDragging(true);
    }
  };

  const onChange = (event: ChangeEvent<HTMLInputElement>): void => {
    handleFile(event.target.files?.[0]);
    // Allow re-selecting the same file after a failed attempt.
    event.target.value = '';
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      inputRef.current?.click();
    }
  };

  return (
    <div className="dropzone-wrapper">
      <div
        className={[
          'dropzone',
          isDragging ? 'dropzone--active' : '',
          disabled ? 'dropzone--disabled' : '',
        ]
          .filter(Boolean)
          .join(' ')}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={() => setDragging(false)}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={onKeyDown}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        aria-describedby={inputId}
      >
        <p className="dropzone__label">{label}</p>
        <p className="dropzone__hint" id={inputId}>
          {hint}
        </p>
        <span className="dropzone__accepts">
          {accept.join(' / ')} &middot; up to {formatBytes(maxBytes)}
        </span>
        <input
          ref={inputRef}
          id={`${inputId}-input`}
          className="dropzone__input"
          type="file"
          accept={accept.join(',')}
          disabled={disabled}
          onChange={onChange}
        />
      </div>
      {validationError ? (
        <p className="dropzone__error" role="alert">
          {validationError}
        </p>
      ) : null}
    </div>
  );
}
