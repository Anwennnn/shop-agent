"""Lazy-loaded knowledge-base tool with safe error handling."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from langchain.tools import tool

from utils.observability import classify_error, log_event


NO_ANSWER_MESSAGE = "这个问题暂时我还不会，您可以联系人工客服"
KNOWLEDGE_ERROR_MESSAGE = "知识库暂时无法访问，请稍后重试或联系人工客服"

_rag_system: Any | None = None
_rag_lock = Lock()


def get_rag_system():
    """Load the RAG system only when the first knowledge query arrives."""
    global _rag_system
    if _rag_system is None:
        with _rag_lock:
            if _rag_system is None:
                from models.rag import RAGSystem

                _rag_system = RAGSystem()
    return _rag_system


@tool(description="查询商城知识库，适用于退货政策、优惠规则等问题")
def query_knowledge(question: str) -> str:
    """Answer one store-policy question without exposing internal failures."""
    started_at = time.perf_counter()
    question = question.strip() if isinstance(question, str) else ""

    if not question:
        log_event(
            component="rag",
            operation="query",
            tool_name="query_knowledge",
            success=False,
            started_at=started_at,
            error_type="invalid_parameter",
        )
        return "请提供一个具体的知识库问题"

    try:
        result = get_rag_system().query(question)
        if result is None:
            log_event(
                component="rag",
                operation="query",
                tool_name="query_knowledge",
                success=True,
                started_at=started_at,
                outcome="no_relevant_context",
            )
            return NO_ANSWER_MESSAGE

        log_event(
            component="rag",
            operation="query",
            tool_name="query_knowledge",
            success=True,
            started_at=started_at,
            outcome="answered",
        )
        return result
    except FileNotFoundError as exc:
        log_event(
            component="rag",
            operation="query",
            tool_name="query_knowledge",
            success=False,
            started_at=started_at,
            error_type="embedding_or_document_not_found",
        )
        return "知识库资源不完整，请联系管理员处理"
    except Exception as exc:
        log_event(
            component="rag",
            operation="query",
            tool_name="query_knowledge",
            success=False,
            started_at=started_at,
            error_type=classify_error(exc),
        )
        return KNOWLEDGE_ERROR_MESSAGE
