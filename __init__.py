"""GoneBy Hermes plugin — adapter between Hermes' plugin API and the goneby library.

Registration only. All behavior lives in :mod:`goneby_plugin.adapter`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Wire GoneBy tools and hooks into Hermes.

    Raises at import/registration time only for a missing goneby package or a
    malformed config — a missing encryption key is reported by the tools
    themselves so the plugin still loads and ``goneby_status`` stays usable.
    """
    from .adapter import GoneByPlugin

    plugin = GoneByPlugin(ctx)
    plugin.register()
    logger.info("goneby plugin registered (project=%s)", plugin.project)
