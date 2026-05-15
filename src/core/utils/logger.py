import atexit
import re
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger as _logger

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR = _REPO_ROOT / "artifacts" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_RUN_TS = datetime.now().strftime("%Y%m%d-%H%M%S")
_LOG_FILE = _LOG_DIR / f"eval-{_RUN_TS}-{__import__('os').getpid()}.log"

_logger.remove()
_logger.add(
    sys.stderr,
    enqueue=True,
    backtrace=True,
    diagnose=True,
    level="DEBUG",
)
_logger.add(
    _LOG_FILE,
    enqueue=True,
    backtrace=True,
    diagnose=True,
    rotation="100 MB",
    level="DEBUG",
)

_logger.info(f"=== process start === pid={__import__('os').getpid()} log_file={_LOG_FILE}")


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
