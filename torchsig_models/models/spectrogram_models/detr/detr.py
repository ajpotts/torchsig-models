"""Ultralytics RT-DETR model factories for spectrogram detection."""

from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

__all__ = ["rtdetr_l"]


def rtdetr_l(
    *,
    pretrained: bool = True,
    path: str | Path | None = None,
) -> RTDETR:
    """Construct an Ultralytics RT-DETR-L detector.

    Args:
        pretrained: Load Ultralytics' pretrained RT-DETR-L weights when
            ``path`` is not supplied. If false, construct the architecture
            from its YAML definition.
        path: Optional checkpoint or model-YAML path. This takes precedence
            over ``pretrained``.

    Returns:
        An Ultralytics RT-DETR model ready for training or inference.
    """
    source = Path(path) if path is not None else None
    if source is None:
        source = Path("rtdetr-l.pt" if pretrained else "rtdetr-l.yaml")
    return RTDETR(str(source))
