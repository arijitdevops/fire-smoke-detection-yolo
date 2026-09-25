"""Safe handling of multipart uploads.

The client filename is never trusted: it is reduced to its basename, stripped of
anything outside a conservative character set, and only its suffix is reused.
The stored name is always a fresh UUID, so a malicious ``../../etc/passwd`` or a
Windows device name cannot escape the upload directory.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from fastapi import UploadFile

from .errors import PayloadTooLargeError, UploadValidationError

LOGGER: Final = logging.getLogger(__name__)

#: Characters kept when echoing a client filename back to the user.
_SAFE_CHARS: Final = re.compile(r"[^A-Za-z0-9._-]+")

#: Read size for streaming an upload to disk (1 MiB).
CHUNK_SIZE: Final[int] = 1024 * 1024

#: Content-type prefixes accepted per upload kind. Browsers and curl disagree
#: often enough that ``application/octet-stream`` is tolerated as well.
_CONTENT_TYPE_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "image": ("image/", "application/octet-stream"),
    "video": ("video/", "application/octet-stream"),
}


@dataclass(slots=True, frozen=True)
class SavedUpload:
    """A successfully stored upload."""

    path: Path
    original_filename: str
    size_bytes: int
    suffix: str


def sanitise_filename(raw: str | None, fallback: str = "upload") -> str:
    """Return a display-safe filename derived from an untrusted client value.

    Args:
        raw: Filename as supplied in the multipart part.
        fallback: Value used when nothing usable remains.

    Returns:
        A basename containing only ``[A-Za-z0-9._-]`` and at most 120 characters.

    """
    if not raw:
        return fallback
    # ``Path(...).name`` drops POSIX directories; the split handles Windows paths
    # that arrive verbatim from some browsers (e.g. ``C:\\fakepath\\clip.mp4``).
    candidate = Path(raw.replace("\\", "/")).name
    cleaned = _SAFE_CHARS.sub("_", candidate).strip("._-")
    if not cleaned:
        return fallback
    return cleaned[:120]


def _check_extension(suffix: str, allowed: Iterable[str], kind: str) -> None:
    """Raise :class:`UploadValidationError` when ``suffix`` is not allowed."""
    allowed_set = {item.lower() for item in allowed}
    if suffix.lower() not in allowed_set:
        raise UploadValidationError(
            f"unsupported {kind} type '{suffix or '(none)'}'",
            {"allowed_extensions": sorted(allowed_set)},
        )


def _check_content_type(content_type: str | None, kind: str) -> None:
    """Raise :class:`UploadValidationError` when the declared MIME type is wrong."""
    prefixes = _CONTENT_TYPE_PREFIXES[kind]
    value = (content_type or "").split(";", 1)[0].strip().lower()
    if not value or not value.startswith(prefixes):
        raise UploadValidationError(
            f"unexpected content type '{content_type or '(missing)'}' for {kind} upload",
            {"expected_prefixes": list(prefixes)},
        )


async def save_upload(
    upload: UploadFile,
    destination_dir: Path,
    allowed_extensions: Iterable[str],
    max_bytes: int,
    kind: str,
) -> SavedUpload:
    """Validate and stream an upload to disk under a UUID filename.

    Args:
        upload: The incoming multipart file.
        destination_dir: Directory to write into; created if missing.
        allowed_extensions: Suffix allowlist, e.g. ``{'.mp4', '.mov'}``.
        max_bytes: Hard size cap; the partial file is removed when exceeded.
        kind: ``'image'`` or ``'video'``; selects the content-type allowlist.

    Returns:
        The :class:`SavedUpload` describing the stored file.

    Raises:
        UploadValidationError: Bad extension, content type, or empty body.
        PayloadTooLargeError: The body exceeded ``max_bytes``.

    """
    if kind not in _CONTENT_TYPE_PREFIXES:
        raise ValueError(f"unknown upload kind: {kind!r}")

    original = sanitise_filename(upload.filename, fallback=f"{kind}_upload")
    suffix = Path(original).suffix
    _check_extension(suffix, allowed_extensions, kind)
    _check_content_type(upload.content_type, kind)

    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UploadValidationError(f"upload directory is not writable: {exc}") from exc

    target = destination_dir / f"{uuid.uuid4().hex}{suffix.lower()}"
    written = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    handle.close()
                    _discard(target)
                    raise PayloadTooLargeError(
                        f"file exceeds the {max_bytes // (1024 * 1024)} MB limit",
                        {"max_bytes": max_bytes},
                    )
                handle.write(chunk)
    except OSError as exc:
        _discard(target)
        raise UploadValidationError(f"could not store upload: {exc}") from exc
    finally:
        await upload.close()

    if written == 0:
        _discard(target)
        raise UploadValidationError("uploaded file is empty")

    LOGGER.info("stored upload %s (%s, %d bytes)", target.name, original, written)
    return SavedUpload(
        path=target, original_filename=original, size_bytes=written, suffix=suffix.lower()
    )


def _discard(path: Path) -> None:
    """Delete a partial upload, logging but never raising on failure."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - platform-specific lock failures
        LOGGER.warning("could not remove partial upload %s: %s", path, exc)
