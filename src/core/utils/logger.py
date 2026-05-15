import atexit
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger as _logger

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / "artifacts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d-%H%M%S")
PID = os.getpid()
RUN_BASENAME = f"eval-{RUN_TS}-{PID}"
_LOG_FILE = LOG_DIR / f"{RUN_BASENAME}.log"

_logger.remove()
_logger.add(
    sys.stderr,
    enqueue=True,
    backtrace=True,
    diagnose=True,
    level="INFO",
)
_logger.add(
    _LOG_FILE,
    enqueue=True,
    backtrace=True,
    diagnose=True,
    rotation="100 MB",
    level="TRACE",
)

_logger.info(f"=== process start === pid={PID} log_file={_LOG_FILE}")


def _on_exit() -> None:
    _logger.info("=== process exit ===")
    _logger.complete()


atexit.register(_on_exit)


class ArgLogger:
    def __getattr__(self, level):
        def log(msg, *args, **kwargs):
            placeholders = len(re.findall(r"(?<!\{)\{[^{]*?\}(?!\})", msg))
            used = args[:placeholders]
            unused = args[placeholders:]

            formatted = msg.format(*used) if used else msg
            if unused:
                formatted += " | " + " | ".join(str(a) for a in unused)

            getattr(_logger.opt(depth=1), level)(formatted, **kwargs)

        return log


logger = ArgLogger()
