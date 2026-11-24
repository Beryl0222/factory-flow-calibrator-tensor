"""分组单调有理二次样条的可逆校准层与梯度回传。"""
from __future__ import annotations

from .splines import (
    KNOT_TOLERANCE,
    ROUNDTRIP_TOLERANCE,
    LOGJAC_TOLERANCE,
    GRADIENT_TOLERANCE,
    TAIL_LEFT,
    TAIL_RIGHT,
    RqsGroup,
    GroupValidation,
    build_group,
    forward,
    inverse,
    forward_vjp,
    inverse_vjp,
    ParamGradient,
    PointResult,
)
from .registry import GroupRegistry, InvalidRevision, UnknownRevision
from .chains import (
    CalibrationService,
    ChainSpec,
    StageSpec,
    BatchResult,
    RowResult,
    UnavailableReason,
    BackpropResult,
    CommitmentMismatch,
)
from .receipts import ReceiptStore, StageTrace

__all__ = [
    "KNOT_TOLERANCE",
    "ROUNDTRIP_TOLERANCE",
    "LOGJAC_TOLERANCE",
    "GRADIENT_TOLERANCE",
    "TAIL_LEFT",
    "TAIL_RIGHT",
    "RqsGroup",
    "GroupValidation",
    "build_group",
    "forward",
    "inverse",
    "forward_vjp",
    "inverse_vjp",
    "ParamGradient",
    "PointResult",
    "GroupRegistry",
    "InvalidRevision",
    "UnknownRevision",
    "CalibrationService",
    "ChainSpec",
    "StageSpec",
    "BatchResult",
    "RowResult",
    "UnavailableReason",
    "BackpropResult",
    "CommitmentMismatch",
    "ReceiptStore",
    "StageTrace",
]
