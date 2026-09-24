"""Safe structured logging and user-facing error classification."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import httpx

from config import setting


_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_configured = False


def configure_logging() -> None:
    """Configure one rotating JSON-lines application log."""
    global _configured
    if _configured:
        return

    log_dir = Path(setting.DATA_DIR).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("shop_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    _configured = True


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str) -> Token[str]:
    return _request_id.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    _request_id.reset(token)


def current_request_id() -> str:
    return _request_id.get()


def log_event(
    *,
    component: str,
    operation: str,
    success: bool,
    started_at: float,
    tool_name: str | None = None,
    error_type: str | None = None,
    exception_class: str | None = None,
    outcome: str | None = None,
) -> None:
    """Write metadata only; never include prompts, keys, or user content."""
    configure_logging()
    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request_id": current_request_id(),
        "component": component,
        "operation": operation,
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "success": success,
    }
    if tool_name:
        payload["tool_name"] = tool_name
    if error_type:
        payload["error_type"] = error_type
    if exception_class:
        payload["exception_class"] = exception_class
    if outcome:
        payload["outcome"] = outcome

    logging.getLogger("shop_agent").info(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def classify_error(exc: Exception) -> str:
    """Map provider/library exceptions to stable internal categories."""
    message = str(exc).lower()
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "model_timeout"
    if isinstance(exc, sqlite3.Error):
        return "database_error"
    if isinstance(exc, FileNotFoundError):
        return "resource_not_found"
    if isinstance(exc, TypeError):
        return "invalid_response_format"
    if any(
        word in message
        for word in (
            "api key",
            "api_key",
            "invalidapikey",
            "unauthorized",
            "authentication",
            "401",
            "403",
        )
    ):
        return "model_authentication_error"
    if any(word in message for word in ("timeout", "timed out")):
        return "model_timeout"
    if any(word in message for word in ("429", "throttling", "quota")):
        return "model_rate_limit"
    if isinstance(exc, ValueError):
        return "invalid_parameter"
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code in (401, 403):
            return "model_authentication_error"
        if status_code == 429:
            return "model_rate_limit"
        if status_code >= 500:
            return "model_service_error"
    return "unexpected_error"


def agent_error_message(error_type: str, request_id: str) -> str:
    messages = {
        "model_timeout": "模型响应超时，请稍后重试。",
        "model_authentication_error": "模型服务认证失败，请联系管理员检查配置。",
        "model_rate_limit": "模型服务当前繁忙或额度受限，请稍后重试。",
        "model_service_error": "模型服务暂时不可用，请稍后重试。",
    }
    message = messages.get(error_type, "系统暂时无法处理该请求，请稍后重试。")
    return f"{message}请求编号：{request_id}"
