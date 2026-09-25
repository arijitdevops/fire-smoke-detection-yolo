"""Browser-playable MP4 writer.

OpenCV's ``cv2.VideoWriter`` can only produce MPEG-4 Part 2 (``mp4v``) with the
pip wheels, and no mainstream browser can play that inside a ``<video>`` tag.
This module pipes raw BGR frames into ``ffmpeg`` and encodes H.264
(``libx264``, ``yuv420p``, ``+faststart``), which plays everywhere.

The ``ffmpeg`` executable comes from the ``imageio-ffmpeg`` wheel, which ships a
static build for Windows, macOS and Linux, so no system install is required.  Set
``IMAGEIO_FFMPEG_EXE`` to use a different binary.  If ``imageio-ffmpeg`` is not
importable the writer falls back to ``mp4v`` and logs a warning: the file can
still be downloaded, but it will not play in the browser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from ..errors import MediaProcessingError

LOGGER: Final = logging.getLogger(__name__)

#: Codec label reported in the job summary.
CODEC_H264: Final[str] = "h264"
CODEC_MP4V: Final[str] = "mp4v"

#: x264 settings: visually lossless-ish quality at a sensible encode speed.
X264_CRF: Final[str] = "23"
X264_PRESET: Final[str] = "veryfast"


def _ffmpeg_writer(path: Path, width: int, height: int, fps: float) -> Any | None:
    """Start an ``imageio-ffmpeg`` H.264 writer generator, or return ``None``."""
    try:
        import imageio_ffmpeg
    except ImportError:
        LOGGER.warning(
            "imageio-ffmpeg is not installed; falling back to mp4v, which browsers "
            "cannot play. Run 'pip install imageio-ffmpeg'."
        )
        return None

    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=2,  # yuv420p needs even dimensions; pads odd ones by 1px
        ffmpeg_log_level="error",
        output_params=["-crf", X264_CRF, "-preset", X264_PRESET, "-movflags", "+faststart"],
    )
    try:
        writer.send(None)  # start the ffmpeg subprocess
    except Exception as exc:  # binary missing, not executable, codec unavailable
        LOGGER.warning("could not start ffmpeg (%s); falling back to mp4v", exc)
        return None
    return writer


class AnnotatedVideoWriter:
    """Write BGR frames to an MP4 file, preferring browser-playable H.264.

    Use it as a context manager::

        with AnnotatedVideoWriter(path, width, height, fps) as writer:
            writer.write(frame)

    Args:
        path: Destination ``.mp4`` file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Output frame rate.
        prefer_h264: Set ``False`` to force the OpenCV ``mp4v`` path (tests).

    """

    def __init__(
        self, path: Path, width: int, height: int, fps: float, prefer_h264: bool = True
    ) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self._ffmpeg: Any | None = None
        self._cv_writer: cv2.VideoWriter | None = None
        self.codec: str = CODEC_MP4V

        path.parent.mkdir(parents=True, exist_ok=True)
        if prefer_h264:
            self._ffmpeg = _ffmpeg_writer(path, width, height, fps)
        if self._ffmpeg is not None:
            self.codec = CODEC_H264
            return

        self._cv_writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
        )
        if not self._cv_writer.isOpened():
            raise MediaProcessingError(
                "could not open a video writer: install imageio-ffmpeg for H.264 output",
                {"output": path.name},
            )

    @property
    def browser_playable(self) -> bool:
        """``True`` when the output is H.264 and will play in a ``<video>`` tag."""
        return self.codec == CODEC_H264

    def write(self, frame: np.ndarray) -> None:
        """Append one HxWx3 BGR ``uint8`` frame."""
        if self._ffmpeg is not None:
            self._ffmpeg.send(np.ascontiguousarray(frame, dtype=np.uint8))
        elif self._cv_writer is not None:
            self._cv_writer.write(frame)

    def close(self) -> None:
        """Flush and finalise the file. Safe to call more than once."""
        if self._ffmpeg is not None:
            try:
                self._ffmpeg.close()
            finally:
                self._ffmpeg = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None

    def __enter__(self) -> AnnotatedVideoWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
