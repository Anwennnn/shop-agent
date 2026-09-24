"""Retrieval-augmented generation for store policy questions."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import setting
from models.llm import LLMInitializer


class RAGSystem:
    """Lazy singleton for the policy knowledge base."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.qa_chain = None
                    instance.vector_store = None
                    cls._instance = instance
                    try:
                        instance.initialize()
                    except Exception:
                        # A failed initialization must not poison later retries.
                        cls._instance = None
                        raise
        return cls._instance

    def initialize(self):
        if self.qa_chain is not None:
            return self.qa_chain

        embedding_path = Path(setting.EMBEDDING_MODEL)
        if not embedding_path.exists():
            raise FileNotFoundError("本地向量模型不存在")

        embeddings = HuggingFaceEmbeddings(
            model_name=str(embedding_path),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

        db_dir = Path(setting.CHROMA_DB_PATH)
        db_file = db_dir / "chroma.sqlite3"

        if db_file.exists():
            vector_store = Chroma(
                persist_directory=str(db_dir),
                embedding_function=embeddings,
            )
        else:
            document_path = Path(setting.DOC_PATH)
            if not document_path.exists():
                raise FileNotFoundError("知识库文档不存在")

            documents = TextLoader(
                str(document_path),
                encoding="utf-8",
            ).load()
            splits = RecursiveCharacterTextSplitter(
                chunk_size=setting.CHUNK_SIZE,
                chunk_overlap=setting.CHUNK_OVERLAP,
                separators=["\n\n", "\n", "。", "！", "？", "!", "?"],
            ).split_documents(documents)

            if not splits:
                raise ValueError("知识库文档没有可索引内容")

            db_dir.mkdir(parents=True, exist_ok=True)
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=str(db_dir),
            )

        prompt = PromptTemplate(
            template="""
你是一名专业的电商客服助手。请只根据下面的商城政策回答问题。

已知信息：
{context}

用户问题：
{question}

如果已知信息不足以回答，请回复：
“这个问题暂时我还不会，您可以联系人工客服”。
不得编造已知信息中不存在的内容，请使用简洁、自然的中文回答。
""".strip(),
            input_variables=["context", "question"],
        )

        self.vector_store = vector_store
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=LLMInitializer().get_llm(),
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
        )
        return self.qa_chain

    def has_relevant_context(self, question: str) -> bool:
        """Reject clearly unrelated questions before calling the LLM."""
        if self.vector_store is None:
            raise RuntimeError("知识库尚未初始化")

        results = self.vector_store.similarity_search_with_relevance_scores(
            question,
            k=1,
        )
        if not results:
            return False
        return results[0][1] >= setting.RAG_SCORE_THRESHOLD

    def query(self, question: str) -> str | None:
        if not self.has_relevant_context(question):
            return None
        if self.qa_chain is None:
            raise RuntimeError("知识库问答链尚未初始化")

        answer: Any = self.qa_chain.invoke({"query": question})
        if not isinstance(answer, dict):
            raise TypeError("知识库返回格式异常")
        result = answer.get("result")
        if not isinstance(result, str):
            raise TypeError("知识库返回格式异常")
        return result.strip() or None

    def get_chain(self):
        return self.qa_chain


if __name__ == "__main__":
    RAGSystem()
