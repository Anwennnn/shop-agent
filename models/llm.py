from threading import Lock

from langchain_community.chat_models import ChatTongyi
from config import setting

class LLMInitializer:
    """通义千问 LLM 单例初始化器。"""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.llm = None
        return cls._instance

    def get_llm(self):
        """获取全局唯一的通义千问 LLM 对象。"""
        if self.llm is None:
            with self._lock:
                if self.llm is None:
                    api_key = setting.TONGYI_KEY
                    if not api_key:
                        raise RuntimeError(
                            "请先配置 DASHSCOPE_API_KEY 环境变量"
                        )
                    self.llm = ChatTongyi(
                        model=setting.MODEL_NAME,
                        api_key=api_key,
                        max_retries=2,
                        model_kwargs={"temperature": setting.TEMPERATURE},
                    )
        return self.llm

if __name__ == '__main__':
    llm = LLMInitializer().get_llm()
    llm2 = LLMInitializer().get_llm()
    print(llm)
