import { afterEach, describe, expect, it, vi } from 'vitest';

import { jsonResponse } from '../test/fixtures';
import { ApiRequestError, jobDownloadUrl, uploadVideo } from './client';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('api client', () => {
  it('maps the error envelope onto ApiRequestError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            error: 'model_unavailable',
            message: 'no weights found - train first or set MODEL_PATH',
            detail: { hint: 'Run training' },
          },
          503,
        ),
      ),
    );

    const error = await uploadVideo(new File(['x'], 'clip.mp4')).catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(ApiRequestError);
    const apiError = error as ApiRequestError;
    expect(apiError.status).toBe(503);
    expect(apiError.isModelUnavailable).toBe(true);
    expect(apiError.detail.hint).toBe('Run training');
  });

  it('reports network failures with a stable code', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    await expect(uploadVideo(new File(['x'], 'clip.mp4'))).rejects.toMatchObject({
      code: 'network_error',
      status: 0,
    });
  });

  it('builds download URLs from the job id', () => {
    expect(jobDownloadUrl('abc')).toBe('/api/jobs/abc/download');
  });
});
