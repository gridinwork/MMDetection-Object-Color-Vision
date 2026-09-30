"""MMDetection Object & Color Vision Studio."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from utils.logger import get_logger, log_environment, setup_logging
    from utils.paths import ensure_dirs

    ensure_dirs()
    setup_logging()
    log = get_logger("app")
    log_environment()
    log.info("Starting MMDetection Object & Color Vision Studio")

    def _hook(exc_type, exc, tb):
        log.exception("Unhandled exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook
    from app.main_window import run_app

    return run_app()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
