from threading import Lock
from typing import Any

from langchain.tools import tool

_rag_system:Any | None = None
_rag_lock = Lock()

def get_rag_system():
  """首次查询知识库时才加载RAG，并保证只初始化一次。"""
  global _rag_system

  if _rag_system is None:
    with _rag_lock:
      if _rag_system is None:
        # 放在函数内部，避免导入工具时立即加载向量模型和PyTorch
        from models.rag import RAGSystem
        _rag_system = RAGSystem()
  return _rag_system

@tool(description='查询知识库，当用户问退货政策与优惠规则时，可以调用此工具')
def query_knowledge(question:str) -> str:
  '''
  查询知识库，回答用户问题
  当用户问退货政策与优惠规则时，可以调用此工具
  :param question:用户问题
  :return:回答
  '''
  if not question:
    return "请提供一个问题"

  # 使用RAG系统回答问题
  rag = get_rag_system()
  answer = rag.get_chain().invoke({"query":question})
  rs = answer.get("result","")
  return rs if rs else "这个问题暂时我还不会，您可以联系人工客服"
