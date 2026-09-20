from __future__ import annotations

import sqlite3
import tempfile
import unittest
import json
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import httpx

from agent.service_agent import ServiceAgent
from tools import knowledge_tools, order_tools
from utils import observability


class _FakeExecutor:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error

    def invoke(self, _inputs):
        if self.error is not None:
            raise self.error
        return self.result


class _ToolCallingExecutor:
    def __init__(self, tool):
        self.tool = tool

    def invoke(self, _inputs):
        self.tool.invoke({"order_id": "OD1001"})
        return {"output": "查询完成"}


def _agent_with_executor(executor: _FakeExecutor) -> ServiceAgent:
    agent = ServiceAgent.__new__(ServiceAgent)
    agent.current_user_id = "USR001"
    agent.pending_action = {}
    agent.agent_executor = executor
    return agent


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "orders.db"
        with closing(sqlite3.connect(self.database_path)) as connection:
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
            connection.execute(
                """
                INSERT INTO orders VALUES (
                    'OD1001', 'USR001', '测试商品', 1, 99.0,
                    '已完成', NULL, NULL, '2026-09-20'
                )
                """
            )
            connection.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _tool_map(self):
        pending = {}
        tools = order_tools.create_order_tools("USR001", pending)
        return {current_tool.name: current_tool for current_tool in tools}

    def test_empty_order_id_returns_message_instead_of_raising(self):
        with patch.object(order_tools, "DATABASE_PATH", self.database_path):
            result = self._tool_map()["get_order_status"].invoke({"order_id": ""})
        self.assertFalse(result["success"])
        self.assertIn("订单编号", result["message"])

    def test_database_failure_returns_safe_message(self):
        missing_database = Path(self.temp_dir.name) / "missing" / "orders.db"
        with patch.object(order_tools, "DATABASE_PATH", missing_database):
            result = self._tool_map()["get_order_status"].invoke(
                {"order_id": "OD1001"}
            )
        self.assertEqual(
            result,
            {"success": False, "message": order_tools.DATABASE_ERROR_MESSAGE},
        )

    def test_prepare_return_does_not_write_database(self):
        pending = {}
        with patch.object(order_tools, "DATABASE_PATH", self.database_path):
            tools = order_tools.create_order_tools("USR001", pending)
            tool_map = {current_tool.name: current_tool for current_tool in tools}
            result = tool_map["prepare_return_order"].invoke(
                {"order_id": "OD1001", "reason": "商品质量问题"}
            )
            with closing(sqlite3.connect(self.database_path)) as connection:
                status = connection.execute(
                    "SELECT order_status FROM orders WHERE order_id = 'OD1001'"
                ).fetchone()[0]

        self.assertTrue(result["success"])
        self.assertEqual(status, "已完成")
        self.assertEqual(pending["type"], "return_order")

    def test_agent_and_tool_share_request_id(self):
        tool = self._tool_map()["get_order_status"]
        agent = _agent_with_executor(_ToolCallingExecutor(tool))

        observability.configure_logging()
        log_path = Path(observability.setting.DATA_DIR).parent / "logs" / "app.log"
        before_count = len(log_path.read_text(encoding="utf-8").splitlines())

        with patch.object(order_tools, "DATABASE_PATH", self.database_path):
            self.assertEqual(agent.chat("查询订单"), "查询完成")

        new_records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()[before_count:]
        ]
        request_ids = {record["request_id"] for record in new_records}
        self.assertEqual(len(new_records), 2)
        self.assertEqual(len(request_ids), 1)
        self.assertNotIn("-", request_ids)

    def test_model_timeout_returns_friendly_message(self):
        request = httpx.Request("POST", "https://example.invalid")
        agent = _agent_with_executor(
            _FakeExecutor(error=httpx.ReadTimeout("timeout", request=request))
        )
        result = agent.chat("查询订单")
        self.assertIn("模型响应超时", result)
        self.assertIn("请求编号", result)

    def test_model_authentication_failure_returns_friendly_message(self):
        agent = _agent_with_executor(
            _FakeExecutor(error=ValueError("InvalidApiKey"))
        )
        result = agent.chat("查询订单")
        self.assertIn("模型服务认证失败", result)

    def test_invalid_agent_result_does_not_escape(self):
        agent = _agent_with_executor(_FakeExecutor(result={"output": 123}))
        result = agent.chat("查询订单")
        self.assertIn("系统暂时无法处理", result)

    def test_knowledge_no_match_has_stable_fallback(self):
        class NoMatchRag:
            @staticmethod
            def query(_question):
                return None

        with patch.object(knowledge_tools, "get_rag_system", return_value=NoMatchRag()):
            result = knowledge_tools.query_knowledge.invoke({"question": "无关问题"})
        self.assertEqual(result, knowledge_tools.NO_ANSWER_MESSAGE)

    def test_missing_embedding_has_safe_message(self):
        class MissingEmbeddingRag:
            @staticmethod
            def query(_question):
                raise FileNotFoundError("模型不存在")

        with patch.object(
            knowledge_tools,
            "get_rag_system",
            return_value=MissingEmbeddingRag(),
        ):
            result = knowledge_tools.query_knowledge.invoke({"question": "退货政策"})
        self.assertIn("知识库资源不完整", result)


if __name__ == "__main__":
    unittest.main()
