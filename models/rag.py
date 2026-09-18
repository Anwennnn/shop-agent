import os
from pathlib import Path
from threading import Lock

from models.llm import LLMInitializer
from config import setting

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

class RAGSystem:
    """通义千问 LLM 单例初始化器。"""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.qa_chain = None
                    cls._instance.initialize()
        return cls._instance

    def initialize(self):
        if self.qa_chain is not None:
            return self.qa_chain
        # 获取LLM
        llm = LLMInitializer().get_llm()

        # 加载本地嵌入模型。
        embeddings = HuggingFaceEmbeddings(
            model_name=setting.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

        db_dir = Path(setting.CHROMA_DB_PATH)
        db_file = db_dir / "chroma.sqlite3"

        if db_file.exists():
            # 向量数据库已经存在，直接加载，不再重复切分和向量化文档。
            print("正在加载已有知识库……")

            db = Chroma(
                persist_directory=str(db_dir),
                embedding_function=embeddings,
            )
        else:
            # 只有第一次运行时才读取、切分并向量化政策文档。
            print("首次运行，正在创建知识库……")

            # 加载数据
            loader = TextLoader(
                str(setting.DOC_PATH),
                encoding="utf-8",
            )
            documents = loader.load()

            # 文档切割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=setting.CHUNK_SIZE,
                chunk_overlap=setting.CHUNK_OVERLAP,
                separators=["\n\n", "\n", "。", "！", "？", "!", "?"],
            )
            splits = text_splitter.split_documents(documents)

            db_dir.mkdir(parents=True, exist_ok=True)

            # 创建数据库
            db = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=str(db_dir),
            )

        # 创建提示词模板
        rag_prompt_template = '''
        你是一个专业的电商客服助手，根据以下商品政策信息，用自然友好的对话风格回答用户问题，就像在聊天一样。
    ​
        已知信息:
        {context} # 检索出来的原始文档
    ​
        用户问题:
        {question} # 用户的问题
    ​
        如果已知信息中不包含用户问题的答案，或者已知信息无法回答用户问题，请直接返回"这个问题暂时我还不会。您可以联系人工客服"。
        请不要输出已知信息中不包含的信息或者答案。
        请用中文回答用户问题。
        '''
        rag_prompt = PromptTemplate(
          template=rag_prompt_template,
          input_variables=["context", "question"]
        )

        # 创建qa链
        self.qa_chain = RetrievalQA.from_chain_type(
          llm = llm,
          retriever = db.as_retriever(search_kwargs={"k":1}),
          return_source_documents = False, # 返回源文档
          chain_type_kwargs = {"prompt":rag_prompt} # 自定义提示词
        )
        return self.qa_chain
    
    def get_chain(self):
        return self.qa_chain

if __name__ == '__main__':
    RAGSystem()