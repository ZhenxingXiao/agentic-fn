"""Third-party log sink adapters used by the CLI demos."""

from cli.sinks.picologging_sink import PicologgingLogSink, configure_picologging

__all__ = [
    "PicologgingLogSink",
    "configure_picologging",
]
