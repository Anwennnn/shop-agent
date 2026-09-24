"""Order query, return, and complaint tools with safe error handling."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from langchain.tools import tool

from config import setting
from utils.observability import classify_error, log_event


DATABASE_PATH = Path(setting.DATABASE_PATH)
GENERIC_ACCESS_MESSAGE = "未找到该用户的订单，或当前用户无权操作此订单"
DATABASE_ERROR_MESSAGE = "订单服务暂时不可用，请稍后重试"
TOOL_ERROR_MESSAGE = "订单操作暂时无法完成，请稍后重试"


@contextmanager
def _connect(database_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the configured database; the path is never exposed to the LLM."""
    resolved_path = Path(database_path) if database_path is not None else DATABASE_PATH
    connection = sqlite3.connect(resolved_path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _normalize_order_id(order_id: str) -> str:
    if isinstance(order_id, dict):
        order_id = order_id.get("order_id")
    elif isinstance(order_id, str):
        order_id = order_id.strip()
        if order_id.startswith("{") and order_id.endswith("}"):
            try:
                order_id = json.loads(order_id).get("order_id")
            except (json.JSONDecodeError, AttributeError):
                pass

    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("订单编号不能为空")
    return order_id.strip().upper()


def _normalize_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("用户编号不能为空")
    return user_id.strip().upper()


def _create_service_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS return_requests (
            order_id TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            requested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );

        CREATE TABLE IF NOT EXISTS complaints (
            complaint_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            complaint_content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
        """
    )


def _get_order_status(
    order_id: str,
    user_id: str,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    order_id = _normalize_order_id(order_id)
    user_id = _normalize_user_id(user_id)

    connection_manager = _connect() if database_path is None else _connect(database_path)
    with connection_manager as connection:
        order = connection.execute(
            """
            SELECT order_id, product_name, order_status,
                   tracking_number, estimated_delivery
            FROM orders
            WHERE order_id = ? AND user_id = ?
            """,
            (order_id, user_id),
        ).fetchone()

    if order is None:
        return {"success": False, "message": GENERIC_ACCESS_MESSAGE}

    return {
        "success": True,
        "order_id": order["order_id"],
        "product_name": order["product_name"],
        "order_status": order["order_status"],
        "tracking_number": order["tracking_number"],
        "estimated_delivery": order["estimated_delivery"],
    }


def _return_order(
    order_id: str,
    user_id: str,
    reason: str,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    order_id = _normalize_order_id(order_id)
    user_id = _normalize_user_id(user_id)
    reason = reason.strip() if isinstance(reason, str) else ""
    if not reason:
        raise ValueError("退货原因不能为空")

    connection_manager = _connect() if database_path is None else _connect(database_path)
    with connection_manager as connection:
        _create_service_tables(connection)
        order = connection.execute(
            "SELECT order_status FROM orders WHERE order_id = ? AND user_id = ?",
            (order_id, user_id),
        ).fetchone()

        if order is None:
            return {"success": False, "message": GENERIC_ACCESS_MESSAGE}

        current_status = order["order_status"]
        if current_status == "退货中":
            return {
                "success": True,
                "order_id": order_id,
                "order_status": current_status,
                "message": "该订单已在退货处理中，请勿重复申请",
            }
        if current_status == "待付款":
            return {
                "success": False,
                "order_id": order_id,
                "order_status": current_status,
                "message": "待付款订单不能申请退货，可直接取消订单",
            }

        connection.execute(
            """
            INSERT INTO return_requests (order_id, reason)
            VALUES (?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                reason = excluded.reason,
                requested_at = CURRENT_TIMESTAMP
            """,
            (order_id, reason),
        )
        connection.execute(
            "UPDATE orders SET order_status = '退货中' "
            "WHERE order_id = ? AND user_id = ?",
            (order_id, user_id),
        )

    return {
        "success": True,
        "order_id": order_id,
        "order_status": "退货中",
        "message": "退货申请已提交",
    }


def _complain_order(
    order_id: str,
    user_id: str,
    complaint_content: str,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    order_id = _normalize_order_id(order_id)
    user_id = _normalize_user_id(user_id)
    complaint_content = (
        complaint_content.strip() if isinstance(complaint_content, str) else ""
    )
    if not complaint_content:
        raise ValueError("投诉内容不能为空")

    connection_manager = _connect() if database_path is None else _connect(database_path)
    with connection_manager as connection:
        _create_service_tables(connection)
        order_exists = connection.execute(
            "SELECT 1 FROM orders WHERE order_id = ? AND user_id = ?",
            (order_id, user_id),
        ).fetchone()
        if order_exists is None:
            return {"success": False, "message": GENERIC_ACCESS_MESSAGE}

        cursor = connection.execute(
            "INSERT INTO complaints (order_id, complaint_content) VALUES (?, ?)",
            (order_id, complaint_content),
        )

    return {
        "success": True,
        "order_id": order_id,
        "complaint_id": cursor.lastrowid,
        "message": "投诉已提交",
    }


def _safe_order_operation(
    tool_name: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run one order operation and always return the documented dict shape."""
    started_at = time.perf_counter()
    try:
        result = operation()
        if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
            raise TypeError("工具返回格式异常")
        log_event(
            component="order_tool",
            operation="execute",
            tool_name=tool_name,
            success=result["success"],
            started_at=started_at,
            outcome="business_success" if result["success"] else "business_rejected",
        )
        return result
    except ValueError as exc:
        log_event(
            component="order_tool",
            operation="execute",
            tool_name=tool_name,
            success=False,
            started_at=started_at,
            error_type="invalid_parameter",
        )
        return {"success": False, "message": str(exc)}
    except sqlite3.Error as exc:
        log_event(
            component="order_tool",
            operation="execute",
            tool_name=tool_name,
            success=False,
            started_at=started_at,
            error_type=classify_error(exc),
        )
        return {"success": False, "message": DATABASE_ERROR_MESSAGE}
    except Exception as exc:
        log_event(
            component="order_tool",
            operation="execute",
            tool_name=tool_name,
            success=False,
            started_at=started_at,
            error_type=classify_error(exc),
        )
        return {"success": False, "message": TOOL_ERROR_MESSAGE}


def create_order_tools(
    current_user_id: str,
    pending_action: dict[str, Any],
    database_path: str | Path | None = None,
):
    """Create tools bound to one trusted authenticated user."""
    trusted_user_id = _normalize_user_id(current_user_id)
    trusted_database_path = Path(database_path) if database_path is not None else None

    @tool("get_order_status")
    def get_order_status(order_id: str) -> dict[str, Any]:
        """查询当前登录用户的订单状态和物流信息。"""
        return _safe_order_operation(
            "get_order_status",
            lambda: _get_order_status(
                order_id,
                trusted_user_id,
                trusted_database_path,
            ),
        )

    @tool("prepare_return_order")
    def prepare_return_order(order_id: str, reason: str) -> dict[str, Any]:
        """准备退货申请；只生成待确认操作，不修改订单。"""

        def prepare() -> dict[str, Any]:
            order = _get_order_status(
                order_id,
                trusted_user_id,
                trusted_database_path,
            )
            if not order["success"]:
                return order
            if order["order_status"] == "退货中":
                return {"success": False, "message": "该订单已在退货处理中"}
            if order["order_status"] == "待付款":
                return {
                    "success": False,
                    "message": "待付款订单不能申请退货，可直接取消订单",
                }

            normalized_reason = reason.strip() if isinstance(reason, str) else ""
            if not normalized_reason:
                raise ValueError("退货原因不能为空")

            pending_action.clear()
            pending_action.update(
                {
                    "type": "return_order",
                    "order_id": order["order_id"],
                    "reason": normalized_reason,
                }
            )
            return {
                "success": True,
                "requires_confirmation": True,
                "message": (
                    f"即将为订单 {order['order_id']} 提交退货申请，"
                    f"原因为“{normalized_reason}”。是否确认？"
                ),
            }

        return _safe_order_operation("prepare_return_order", prepare)

    @tool("prepare_complaint")
    def prepare_complaint(
        order_id: str,
        complaint_content: str,
    ) -> dict[str, Any]:
        """准备订单投诉；只生成待确认操作，不写入投诉记录。"""

        def prepare() -> dict[str, Any]:
            order = _get_order_status(
                order_id,
                trusted_user_id,
                trusted_database_path,
            )
            if not order["success"]:
                return order

            normalized_content = (
                complaint_content.strip()
                if isinstance(complaint_content, str)
                else ""
            )
            if not normalized_content:
                raise ValueError("投诉内容不能为空")

            pending_action.clear()
            pending_action.update(
                {
                    "type": "complain_order",
                    "order_id": order["order_id"],
                    "complaint_content": normalized_content,
                }
            )
            return {
                "success": True,
                "requires_confirmation": True,
                "message": (
                    f"即将针对订单 {order['order_id']} 提交投诉，"
                    f"投诉内容为“{normalized_content}”。是否确认？"
                ),
            }

        return _safe_order_operation("prepare_complaint", prepare)

    return [get_order_status, prepare_return_order, prepare_complaint]


def execute_pending_action(
    current_user_id: str,
    action: dict[str, Any],
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a confirmed action. This function is intentionally not an LLM tool."""
    trusted_user_id = _normalize_user_id(current_user_id)
    action_type = action.get("type")

    if action_type == "return_order":
        return _safe_order_operation(
            "return_order",
            lambda: _return_order(
                action.get("order_id", ""),
                trusted_user_id,
                action.get("reason", ""),
                database_path,
            ),
        )
    if action_type == "complain_order":
        return _safe_order_operation(
            "complain_order",
            lambda: _complain_order(
                action.get("order_id", ""),
                trusted_user_id,
                action.get("complaint_content", ""),
                database_path,
            ),
        )
    return {"success": False, "message": "待确认操作无效"}


__all__ = ["create_order_tools", "execute_pending_action"]


if __name__ == "__main__":
    pending: dict[str, Any] = {}
    tools = create_order_tools("USR001", pending)
    tool_map = {current_tool.name: current_tool for current_tool in tools}
    status_tool = tool_map["get_order_status"]

    print("工具参数：", status_tool.args)
    print(status_tool.invoke({"order_id": "OD1001"}))
    print(status_tool.invoke({"order_id": "OD1002"}))

    assert "user_id" not in status_tool.args
    assert "database_path" not in status_tool.args
    print("测试通过：敏感参数未暴露给大模型")
