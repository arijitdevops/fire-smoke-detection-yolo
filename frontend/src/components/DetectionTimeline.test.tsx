import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { makeSummary } from '../test/fixtures';
import { DetectionTimeline, buildTracks } from './DetectionTimeline';

describe('buildTracks', () => {
  it('groups sampled frames per class, fire first, keeping the best confidence', () => {
    const tracks = buildTracks(makeSummary());

    expect([...tracks.keys()]).toEqual(['fire', 'smoke']);
    expect(tracks.get('fire')).toEqual([
      { start: 1.2, confidence: 0.91 },
      { start: 3.6, confidence: 0.7 },
    ]);
    expect(tracks.get('smoke')).toEqual([{ start: 3.6, confidence: 0.55 }]);
  });
});

describe('DetectionTimeline', () => {
  it('renders one track per class and seeks when a marker is clicked', () => {
    const onSeek = vi.fn();
    render(<DetectionTimeline summary={makeSummary()} currentTime={0} onSeek={onSeek} />);

    expect(screen.getByRole('group', { name: 'fire detections' })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'smoke detections' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'smoke at 0:03.6' }));
    expect(onSeek).toHaveBeenCalledWith(3.6);
  });

  it('explains when nothing was detected', () => {
    render(
      <DetectionTimeline summary={makeSummary({ per_class: [], timeline: [], frames_with_detections: 0 })} />,
    );

    expect(screen.getByText(/nothing was detected/i)).toBeInTheDocument();
  });
});
