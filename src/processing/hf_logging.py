"""Narrow logging filters for noisy third-party ML dependencies."""

from __future__ import annotations

import logging

_TRANSFORMERS_LOGGER_NAMES = ("transformers", "transformers.__init__")
_TRANSFORMERS_ALIAS_WARNING_SUFFIX = (
    "Behavior may be different and this alias will be removed in future versions."
)


class _TransformersAliasWarningFilter(logging.Filter):
    """Drop a known noisy Transformers alias warning without muting other warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.levelno >= logging.WARNING
            and "Accessing `__path__` from `.models." in message
            and _TRANSFORMERS_ALIAS_WARNING_SUFFIX in message
        )


_ALIAS_WARNING_FILTER = _TransformersAliasWarningFilter()


def configure_third_party_logging() -> None:
    """Install targeted filters for known dependency startup noise."""
    for logger_name in _TRANSFORMERS_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        if _ALIAS_WARNING_FILTER not in logger.filters:
            logger.addFilter(_ALIAS_WARNING_FILTER)
