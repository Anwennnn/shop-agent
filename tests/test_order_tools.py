"""Tests for order tools that never touch the production database."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

import pytest

from tools import order_tools


@pytest.fixture()
def temporary_database(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[sqlite3.Connection]:
    """Create an isolated in-memory database and bind the tools to it."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        connection.execute(
            """
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                total_price REAL NOT NULL,
                order_status TEXT NOT NULL,
                tracking_number TEXT,
                estimated_delivery TEXT,
                order_date TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO orders (
                order_id, user_id, product_name, quantity, total_price,
                order_status, tracking_number, estimated_delivery, order_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "OD1001",
                    "USR001",
                    "无线耳机",
                    1,
                    299.0,
                    "已完成",
                    "SF1001",
                    "2026-09-21",
                    "2026-09-10",
                ),
                (
                    "OD1002",
                    "USR002",
                    "机械键盘",
                    1,
                    399.0,
                    "已完成",
                    "SF1002",
                    "2026-09-22",
                    "2026-09-11",
                ),
                (
                    "OD1003",
                    "USR001",
                    "显示器",
                    1,
                    1299.0,
                    "待付款",
                    None,
                    None,
                    "2026-09-12",
                ),
            ],
        )
        connection.commit()

        @contextmanager
        def in_memory_connection() -> Iterator[sqlite3.Connection]:
            with connection:
                yield connection

        monkeypatch.setattr(order_tools, "_connect", in_memory_connection)
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def user_tools(temporary_database: sqlite3.Connection):
    pending_action: dict = {}
    tools = order_tools.create_order_tools("USR001", pending_action)
    return {current_tool.name: current_tool for current_tool in tools}


def _fetch_one(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple = (),
):
    row = connection.execute(sql, parameters).fetchone()
    return tuple(row) if row is not None else None


def test_get_existing_order(user_tools):
    result = user_tools["get_order_status"].invoke({"order_id": "OD1001"})

    assert result["success"] is True
    assert result["order_id"] == "OD1001"
    assert result["product_name"] == "无线耳机"
    assert result["order_status"] == "已完成"


def test_get_nonexistent_order(user_tools):
    result = user_tools["get_order_status"].invoke({"order_id": "OD9999"})

    assert result == {
        "success": False,
        "message": order_tools.GENERIC_ACCESS_MESSAGE,
    }


def test_get_another_users_order_is_rejected(user_tools):
    result = user_tools["get_order_status"].invoke({"order_id": "OD1002"})

    assert result == {
        "success": False,
        "message": order_tools.GENERIC_ACCESS_MESSAGE,
    }


def test_submit_return_successfully(temporary_database: sqlite3.Connection):
    result = order_tools.execute_pending_action(
        "USR001",
        {"type": "return_order", "order_id": "OD1001", "reason": "商品损坏"},
    )

    assert result["success"] is True
    assert result["order_status"] == "退货中"
    assert _fetch_one(
        temporary_database,
        "SELECT order_status FROM orders WHERE order_id = ?",
        ("OD1001",),
    ) == ("退货中",)
    assert _fetch_one(
        temporary_database,
        "SELECT reason FROM return_requests WHERE order_id = ?",
        ("OD1001",),
    ) == ("商品损坏",)


def test_empty_return_reason_is_rejected(temporary_database: sqlite3.Connection):
    result = order_tools.execute_pending_action(
        "USR001",
        {"type": "return_order", "order_id": "OD1001", "reason": "   "},
    )

    assert result == {"success": False, "message": "退货原因不能为空"}
    assert _fetch_one(
        temporary_database,
        "SELECT order_status FROM orders WHERE order_id = ?",
        ("OD1001",),
    ) == ("已完成",)


def test_unpaid_order_cannot_be_returned(temporary_database: sqlite3.Connection):
    result = order_tools.execute_pending_action(
        "USR001",
        {"type": "return_order", "order_id": "OD1003", "reason": "不需要了"},
    )

    assert result["success"] is False
    assert "待付款订单不能申请退货" in result["message"]
    assert _fetch_one(
        temporary_database,
        "SELECT COUNT(*) FROM return_requests",
    ) == (0,)


def test_duplicate_return_does_not_create_another_record(
    temporary_database: sqlite3.Connection,
):
    first_result = order_tools.execute_pending_action(
        "USR001",
        {"type": "return_order", "order_id": "OD1001", "reason": "商品损坏"},
    )
    second_result = order_tools.execute_pending_action(
        "USR001",
        {"type": "return_order", "order_id": "OD1001", "reason": "重复申请"},
    )

    assert first_result["success"] is True
    assert second_result["success"] is True
    assert "请勿重复申请" in second_result["message"]
    assert _fetch_one(
        temporary_database,
        "SELECT COUNT(*) FROM return_requests WHERE order_id = ?",
        ("OD1001",),
    ) == (1,)
    assert _fetch_one(
        temporary_database,
        "SELECT reason FROM return_requests WHERE order_id = ?",
        ("OD1001",),
    ) == ("商品损坏",)


def test_submit_complaint_successfully(temporary_database: sqlite3.Connection):
    result = order_tools.execute_pending_action(
        "USR001",
        {
            "type": "complain_order",
            "order_id": "OD1001",
            "complaint_content": "商品存在明显划痕",
        },
    )

    assert result["success"] is True
    assert isinstance(result["complaint_id"], int)
    assert _fetch_one(
        temporary_database,
        "SELECT order_id, complaint_content FROM complaints WHERE complaint_id = ?",
        (result["complaint_id"],),
    ) == ("OD1001", "商品存在明显划痕")


def test_empty_complaint_content_is_rejected(
    temporary_database: sqlite3.Connection,
):
    result = order_tools.execute_pending_action(
        "USR001",
        {
            "type": "complain_order",
            "order_id": "OD1001",
            "complaint_content": "\t  ",
        },
    )

    assert result == {"success": False, "message": "投诉内容不能为空"}


def test_sql_special_characters_do_not_change_query_semantics(user_tools):
    malicious_order_id = "OD1001' OR '1'='1"

    result = user_tools["get_order_status"].invoke(
        {"order_id": malicious_order_id}
    )

    assert result == {
        "success": False,
        "message": order_tools.GENERIC_ACCESS_MESSAGE,
    }
