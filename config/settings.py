"""项目配置：从环境变量读取密钥，并统一管理文件路径。"""

import os
from pathlib import Path

from dotenv import load_dotenv


# 无论从哪个目录启动，都从项目根目录加载 .env。
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


class Settings:
    """电商客服 Agent 的运行配置。"""

    # 真实密钥只应保存在 .env 或部署平台的环境变量中。
    TONGYI_KEY: str | None =  os.getenv("TONGYI_KEY")
    ZHIPU_KEY: str | None = os.getenv("ZHIPU_KEY")

    # 模型配置
    MODEL_NAME: str = os.getenv("MODEL_NAME", "deepseek-v4-flash-0731")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

    # 订单数据路径
    DATA_DIR: Path = BASE_DIR / "data"
    JSON_PATH: Path = DATA_DIR / "orders.json"
    DATABASE_PATH: Path = DATA_DIR / "orders.db"

    # RAG 配置
    DOC_PATH: Path = BASE_DIR / "docs" / "policy_docs.md"
    EMBEDDING_MODEL: str = str(BASE_DIR / "bge-small-zh-v1.5")
    CHROMA_DB_PATH: str = str(BASE_DIR / ".cache" / "chroma")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "30"))
    RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.2"))


setting = Settings()
