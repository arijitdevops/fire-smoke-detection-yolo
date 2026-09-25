import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { VideoComparison } from './VideoComparison';

describe('VideoComparison', () => {
  it('shows the original and the annotated video side by side', () => {
    render(
      <VideoComparison
        originalSrc="blob:original"
        annotatedSrc="/api/jobs/1/download"
        downloadName="annotated_clip.mp4"
        codec="h264"
      />,
    );

    expect(screen.getByLabelText('Original upload')).toHaveAttribute('src', 'blob:original');
    expect(screen.getByLabelText('Annotated video')).toHaveAttribute('src', '/api/jobs/1/download');
    expect(screen.getByRole('link', { name: 'Download MP4' })).toHaveAttribute(
      'download',
      'annotated_clip.mp4',
    );
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('falls back to a placeholder without the original and warns about non-H.264 output', () => {
    render(
      <VideoComparison
        originalSrc={null}
        annotatedSrc="/api/jobs/1/download"
        downloadName="annotated_clip.mp4"
        codec="mp4v"
      />,
    );

    expect(screen.queryByLabelText('Original upload')).not.toBeInTheDocument();
    expect(screen.getByText(/only available in the tab/i)).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('mp4v');
  });
});
