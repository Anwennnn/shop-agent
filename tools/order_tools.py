"""订单查询、退单和投诉相关的数据库工具。"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from langchain.tools import tool
import json


# 无论程序从哪个目录启动，都使用项目 data 目录下的 orders.db。
DATABASE_PATH = Path(__file__).resolve().parent.parent / "data" / "orders.db"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """创建数据库连接，完成事务后自动关闭连接。"""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        # 内层上下文负责自动提交事务，发生异常时自动回滚。
        with connection:
            yield connection
    finally:
        connection.close()


def _normalize_order_id(order_id: str) -> str:
    # 如果传入的是 JSON 对象(dict)
    if isinstance(order_id, dict):
        order_id = order_id.get("order_id")

    # 如果传入的是 JSON 字符串
    elif isinstance(order_id, str):
        # 判断是否是 JSON 格式字符串
        order_id = order_id.strip()

        if order_id.startswith("{") and order_id.endswith("}"):
            import json
            try:
                order_obj = json.loads(order_id)
                order_id = order_obj.get("order_id")
            except json.JSONDecodeError:
                pass
    """清理并检查订单编号。"""
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("订单编号不能为空")
    return order_id.strip().upper()

def _normalize_user_id(user_id:str) -> str:
    """清理并检查用户编号"""
    if not isinstance(user_id,str) or not user_id.strip():
        raise ValueError("用户编号不能为空")
    return user_id.strip().upper()


def _create_service_tables(connection: sqlite3.Connection) -> None:
    """创建退单申请表和投诉表（如果尚不存在）。"""
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
) -> dict[str, Any]:
    """
    根据订单编号查询订单状态和物流信息

    Args:
      order_id(str):订单ID，订单的唯一标识
    
    Returns:
      dict[str, Any]：包含订单状态信息，查询状态，订单号，商品名称，订单状态，物流单号，预计到达的时间

    """
    order_id = _normalize_order_id(order_id)
    user_id = _normalize_user_id(user_id)

    with _connect() as connection:
        order = connection.execute(
            """
            SELECT
                order_id,
                product_name,
                order_status,
                tracking_number,
                estimated_delivery
            FROM orders
            WHERE order_id = ? AND user_id = ?
            """,
            (order_id,user_id),
        ).fetchone()

    if order is None:
        return {
            "success": False,
            "message": f"未找到订单 {order_id}，或当前用户{user_id}无权操作此订单",
        }

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
    reason: str = "七天无理由退货",
) -> dict[str, Any]:
    """提交退单申请，并将符合条件的订单状态更新为“退货中”。"""
    order_id = _normalize_order_id(order_id)
    user_id = _normalize_user_id(user_id)
    reason = reason.strip() if isinstance(reason, str) else ""
    if not reason:
        raise ValueError("退单原因不能为空")

    with _connect() as connection:
        _create_service_tables(connection)
        order = connection.execute(
            "SELECT order_status FROM orders WHERE order_id = ? AND user_id = ?",
            (order_id,user_id),
        ).fetchone()

        if order is None:
            return {
                "success": False,
                "message": f"未找到订单 {order_id}",
            }

        current_status = order["order_status"]
        if current_status == "退货中":
            return {
                "success": True,
                "order_id": order_id,
                "order_status": current_status,
                "message": "该订单已在退货处理中，请勿重复申请",
            }

        # 未付款订单尚未完成交易，不能走退货流程。
        if current_status == "待付款":
            return {
                "success": False,
                "order_id": order_id,
                "order_status": current_status,
                "message": "待付款订单不能申请退货，可直接取消订单",
            }

        # 一个订单只保留一条退单申请；再次申请时更新原因和申请时间。
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
            "UPDATE orders SET order_status = '退货中' WHERE order_id = ? AND user_id = ?",
            (order_id,user_id),
        )

    return {
        "success": True,
        "order_id": order_id,
        "order_status": "退货中",
        "message": "退单申请已提交",
    }


def _complain_order(
    order_id: str,
    user_id: str,
    complaint_content: str,
) -> dict[str, Any]:
    """针对指定订单提交投诉，并返回新生成的投诉编号。"""
    order_id = _normalize_order_id(order_id)
    user_id = _normalize_user_id(user_id)
    complaint_content = (
        complaint_content.strip() if isinstance(complaint_content, str) else ""
    )
    if not complaint_content:
        raise ValueError("投诉内容不能为空")

    with _connect() as connection:
        _create_service_tables(connection)
        order_exists = connection.execute(
            "SELECT 1 FROM orders WHERE order_id = ? AND user_id = ?",
            (order_id,user_id),
        ).fetchone()

        if order_exists is None:
            return {
                "success": False,
                "message": f"未找到订单 {order_id}，或当前用户{user_id}无权操作此订单",
            }

        cursor = connection.execute(
            """
            INSERT INTO complaints (order_id, complaint_content)
            VALUES (?, ?)
            """,
            (order_id, complaint_content),
        )
        complaint_id = cursor.lastrowid

    return {
        "success": True,
        "order_id": order_id,
        "complaint_id": complaint_id,
        "message": "投诉已提交",
    }

def create_order_tools(current_user_id: str,pending_action: dict[str,Any],):
    """为当前登录用户创建订单工具。"""

    # 用户身份在创建工具时固定，不能被大模型修改。
    trusted_user_id = _normalize_user_id(current_user_id)

    @tool("get_order_status")
    def get_order_status(order_id: str) -> dict[str, Any]:
        """根据订单编号查询当前登录用户的订单状态和物流信息。"""
        return _get_order_status(
            order_id=order_id,
            user_id=trusted_user_id,
        )

    @tool("prepare_return_order")
    def prepare_return_order(
        order_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """准备退货申请。此工具只生成待确认操作，不会真正修改订单。"""

        # 先检查订单是否属于当前用户。
        order = _get_order_status(
            order_id=order_id,
            user_id=trusted_user_id,
        )

        if not order.get("success"):
            return order

        current_status = order.get("order_status")

        if current_status == "退货中":
            return {
                "success": False,
                "message": "该订单已经在退货处理中",
            }

        if current_status == "待付款":
            return {
                "success": False,
                "message": "待付款订单不能申请退货，可直接取消订单",
            }

        reason = reason.strip() if isinstance(reason, str) else ""
        if not reason:
            return {
                "success": False,
                "message": "请提供退货原因",
            }

        # 必须修改原字典，不能重新给 pending_action 赋值。
        pending_action.clear()
        pending_action.update(
            {
                "type": "return_order",
                "order_id": order["order_id"],
                "reason": reason,
            }
        )

        return {
            "success": True,
            "requires_confirmation": True,
            "message": (
                f"即将为订单 {order['order_id']} 提交退货申请，"
                f"原因为“{reason}”。是否确认？"
            ),
        }

    @tool("prepare_complaint")
    def prepare_complaint(
        order_id: str,
        complaint_content: str,
    ) -> dict[str, Any]:
        """准备订单投诉。此工具只生成待确认操作，不会写入投诉记录。"""

        order = _get_order_status(
            order_id=order_id,
            user_id=trusted_user_id,
        )

        if not order.get("success"):
            return order

        complaint_content = (
            complaint_content.strip()
            if isinstance(complaint_content, str)
            else ""
        )

        if not complaint_content:
            return {
                "success": False,
                "message": "请提供投诉内容",
            }

        pending_action.clear()
        pending_action.update(
            {
                "type": "complain_order",
                "order_id": order["order_id"],
                "complaint_content": complaint_content,
            }
        )

        return {
            "success": True,
            "requires_confirmation": True,
            "message": (
                f"即将针对订单 {order['order_id']} 提交投诉，"
                f"投诉内容为“{complaint_content}”。是否确认？"
            ),
    }

    return [
        get_order_status,
        prepare_return_order,
        prepare_complaint,
    ]

def execute_pending_action(
    current_user_id: str,
    action: dict[str, Any],
) -> dict[str, Any]:
    """执行已经由用户明确确认的操作。此函数不注册为 Agent 工具。"""

    trusted_user_id = _normalize_user_id(current_user_id)
    action_type = action.get("type")

    if action_type == "return_order":
        return _return_order(
            order_id=action["order_id"],
            user_id=trusted_user_id,
            reason=action["reason"],
        )

    if action_type == "complain_order":
        return _complain_order(
            order_id=action["order_id"],
            user_id=trusted_user_id,
            complaint_content=action["complaint_content"],
        )

    return {
        "success": False,
        "message": "待确认操作无效",
    }


__all__ = ["create_order_tools","execute_pending_action"]

if __name__ == "__main__":
    # 模拟当前已经登录的用户
    current_user_id = "USR001"

    # 为当前用户创建工具
    tools = create_order_tools(current_user_id)

    # 根据工具名称查找工具
    tool_map = {
        current_tool.name: current_tool
        for current_tool in tools
    }

    status_tool = tool_map["get_order_status"]

    print("工具参数：", status_tool.args)

    print("\n1. 查询当前用户自己的订单 OD1001")
    result = status_tool.invoke({
        "order_id": "OD1001",
    })
    print(result)

    print("\n2. 尝试查询其他用户的订单 OD1002")
    result = status_tool.invoke({
        "order_id": "OD1002",
    })
    print(result)

    # 验证敏感参数没有暴露给大模型
    assert "user_id" not in status_tool.args
    assert "database_path" not in status_tool.args

    print("\n测试通过：user_id 和 database_path 未暴露给大模型")