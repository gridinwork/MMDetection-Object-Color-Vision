"""Checkpoint downloader with a temporary file and progress callback."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from utils.logger import get_logger

log = get_logger("download")


class DownloadError(RuntimeError):
    pass


def download_file(url: str, dest: Path, progress=None, timeout: int = 60) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".part")
    log.info("Downloading %s", url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MMDetectionVisionStudio/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        log.exception("Download failed")
        raise DownloadError(str(exc)) from exc
    if total and done < total:
        temporary.unlink(missing_ok=True)
        raise DownloadError("Download ended before the full file was received.")
    temporary.replace(dest)
    log.info("Saved %s (%s bytes)", dest, dest.stat().st_size)
    return dest
