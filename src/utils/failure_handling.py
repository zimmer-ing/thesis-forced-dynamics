from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


class NumericalFailureError(RuntimeError):
    """Structured error for solver failures and numerically invalid predictions."""

    def __init__(
        self,
        failure_type: str,
        stage: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.failure_type = failure_type
        self.stage = stage
        self.details = details or {}


def _short_message(message: str, max_len: int = 1000) -> str:
    message = str(message)
    if len(message) <= max_len:
        return message
    return message[: max_len - 3] + "..."


def classify_exception(exc: BaseException, stage: str) -> Dict[str, Any]:
    """Map exceptions to a stable result schema for HPO/evaluation bookkeeping."""

    if isinstance(exc, NumericalFailureError):
        return {
            "success": False,
            "stage": exc.stage or stage,
            "failure_type": exc.failure_type,
            "exception_type": type(exc).__name__,
            "message": _short_message(str(exc)),
            "details": exc.details,
        }

    msg = str(exc)
    msg_lower = msg.lower()
    if "underflow in dt" in msg_lower:
        failure_type = "solver_failure"
    elif "out of memory" in msg_lower or "outofmemory" in msg_lower:
        failure_type = "resource_failure"
    elif any(token in msg_lower for token in ["non-finite", "nonfinite", "nan", "inf", "infinity"]):
        failure_type = "divergence_failure"
    else:
        failure_type = "runtime_failure"

    return {
        "success": False,
        "stage": stage,
        "failure_type": failure_type,
        "exception_type": type(exc).__name__,
        "message": _short_message(msg),
        "details": {},
    }


def success_record(stage: str, **details: Any) -> Dict[str, Any]:
    return {
        "success": True,
        "stage": stage,
        "failure_type": None,
        "exception_type": None,
        "message": None,
        "details": details,
    }


def get_failure_policy(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    experiment_cfg = (config or {}).get("experiment", {})
    policy = dict(experiment_cfg.get("failure_policy", {}) or {})
    return {
        "non_finite_is_failure": bool(policy.get("non_finite_is_failure", True)),
        "divergent_prediction_is_failure": bool(policy.get("divergent_prediction_is_failure", True)),
        "max_abs_prediction": policy.get("max_abs_prediction", 1e6),
        "max_abs_loss": policy.get("max_abs_loss", 1e12),
    }


def _to_float(value: Any) -> Optional[float]:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_finite_number(
    value: Any,
    name: str,
    stage: str,
    *,
    max_abs_threshold: Optional[float] = None,
) -> None:
    scalar = _to_float(value)
    if scalar is None:
        return
    if not math.isfinite(scalar):
        raise NumericalFailureError(
            "divergence_failure",
            stage,
            f"{name} is non-finite: {scalar}",
            {"name": name, "value": scalar},
        )
    if max_abs_threshold is not None and abs(scalar) > float(max_abs_threshold):
        raise NumericalFailureError(
            "divergence_failure",
            stage,
            f"{name} exceeded threshold {max_abs_threshold}: {scalar}",
            {"name": name, "value": scalar, "max_abs_threshold": float(max_abs_threshold)},
        )


def ensure_finite_tensor(
    tensor: Any,
    name: str,
    stage: str,
    *,
    max_abs_threshold: Optional[float] = None,
) -> None:
    if not torch.is_tensor(tensor):
        ensure_finite_number(tensor, name, stage, max_abs_threshold=max_abs_threshold)
        return

    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    if not bool(finite_mask.all().item()):
        total = int(detached.numel())
        n_nan = int(torch.isnan(detached).sum().item())
        n_posinf = int(torch.isposinf(detached).sum().item())
        n_neginf = int(torch.isneginf(detached).sum().item())
        raise NumericalFailureError(
            "divergence_failure",
            stage,
            f"{name} contains non-finite values.",
            {
                "name": name,
                "numel": total,
                "n_nan": n_nan,
                "n_posinf": n_posinf,
                "n_neginf": n_neginf,
                "n_nonfinite": total - int(finite_mask.sum().item()),
            },
        )

    if max_abs_threshold is not None and detached.numel() > 0:
        max_abs = float(detached.abs().max().detach().cpu().item())
        if max_abs > float(max_abs_threshold):
            raise NumericalFailureError(
                "divergence_failure",
                stage,
                f"{name} exceeded max_abs_prediction threshold {max_abs_threshold}: {max_abs}",
                {"name": name, "max_abs": max_abs, "max_abs_threshold": float(max_abs_threshold)},
            )


def finite_tree_summary(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Inspect nested metric dictionaries for non-finite numeric values."""

    nonfinite_paths = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                child_path = f"{path}[{idx}]"
                visit(child, child_path)
        elif torch.is_tensor(value):
            try:
                ensure_finite_tensor(value, path or "tensor", stage="metric_check")
            except NumericalFailureError:
                nonfinite_paths.append(path or "tensor")
        elif isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                nonfinite_paths.append(path or "value")

    visit(obj, prefix)
    return {
        "all_finite": len(nonfinite_paths) == 0,
        "nonfinite_paths": nonfinite_paths,
    }
