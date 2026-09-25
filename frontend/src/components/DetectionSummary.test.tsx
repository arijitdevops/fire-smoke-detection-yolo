import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { makeSummary } from '../test/fixtures';
import { DetectionSummary } from './DetectionSummary';

describe('DetectionSummary', () => {
  it('shows per-class frame counts and first-seen timestamps', () => {
    render(<DetectionSummary summary={makeSummary()} />);

    const fireTile = screen.getByText('Frames with fire').closest('.tile') as HTMLElement;
    expect(within(fireTile).getByText('2')).toBeInTheDocument();
    expect(within(fireTile).getByText('first at 0:01.2')).toBeInTheDocument();

    const smokeRow = screen.getByRole('row', { name: /smoke/ });
    expect(within(smokeRow).getByText('0:03.6')).toBeInTheDocument();
  });

  it('renders the empty state when nothing was detected', () => {
    render(<DetectionSummary summary={makeSummary({ per_class: [], timeline: [] })} />);

    expect(screen.getByText(/no fire or smoke was detected/i)).toBeInTheDocument();
  });

  it('turns timeline timestamps into seek buttons', () => {
    const onSeek = vi.fn();
    render(<DetectionSummary summary={makeSummary()} onSeek={onSeek} />);

    fireEvent.click(screen.getAllByRole('button', { name: '0:01.2' })[0] as HTMLElement);
    expect(onSeek).toHaveBeenCalledWith(1.2);
  });
});
