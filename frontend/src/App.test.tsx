import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import { jsonResponse, makeHealth } from './test/fixtures';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('App', () => {
  it('warns and disables uploads when the API has no weights', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse(
          makeHealth({
            status: 'degraded',
            model_loaded: false,
            model_source: null,
            detail: 'no weights found - train first or set MODEL_PATH',
          }),
        ),
      ),
    );

    render(<App />);

    expect(await screen.findByText('No detector weights loaded')).toBeInTheDocument();
    expect(screen.getByText(/no weights found/)).toBeInTheDocument();
  });

  it('shows the video upload area when a model is loaded', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(makeHealth())));

    render(<App />);

    expect(await screen.findByText('Drop a video here')).toBeInTheDocument();
    expect(screen.queryByText('No detector weights loaded')).not.toBeInTheDocument();
  });
});
