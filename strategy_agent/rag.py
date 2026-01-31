import os
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils import serialize


class Rag:
    """
    Lightweight RAG wrapper that builds/loads a Chroma vector store from CSV knowledge.

    CSV required columns: title, content (source optional)
    """

    def __init__(self, embedding_model_path: str, device: str = "cpu"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
        )
        self.title_vector_store = None
        self.content_vector_store = None
        self.title_docs: List[Document] = []
        self.content_docs: List[Document] = []

    def _load_csv(self, data_path: str) -> pd.DataFrame:
        df = serialize.dataframe_read(data_path, file_type="csv")
        if df is None:
            return pd.DataFrame()
        return df.reset_index(drop=True)

    def data_process_4_build_vector_store(self, data_path: str):
        data = self._load_csv(data_path)
        if data.empty:
            return [], []

        title_docs = []
        content_docs = []

        for _, row in data.iterrows():
            source = row.get("source", "")
            title = row.get("title", "")
            content = row.get("content", "")

            metadata = {"source": source, "title": title, "content": content} if source else {"title": title, "content": content}

            title_doc = Document(page_content=title, metadata=metadata)
            content_doc = Document(page_content=content, metadata=metadata)

            title_docs.append(title_doc)
            content_docs.append(content_doc)

        self.title_docs = title_docs
        self.content_docs = content_docs
        return title_docs, content_docs

    def build_title_vector_store(self, db_path: str):
        if not self.title_docs:
            raise ValueError("No title_docs to build. Call data_process_4_build_vector_store first.")
        os.makedirs(db_path, exist_ok=True)
        batch_size = 64
        if len(self.title_docs) <= batch_size:
            vectorstore = Chroma.from_documents(self.title_docs, self.embedding_model, persist_directory=db_path)
        else:
            first_batch = self.title_docs[:batch_size]
            vectorstore = Chroma.from_documents(first_batch, self.embedding_model, persist_directory=db_path)
            for i in range(batch_size, len(self.title_docs), batch_size):
                batch = self.title_docs[i : i + batch_size]
                vectorstore.add_documents(batch)
        self.title_vector_store = vectorstore

    def build_content_vector_store(self, db_path: str):
        if not self.content_docs:
            raise ValueError("No content_docs to build. Call data_process_4_build_vector_store first.")
        os.makedirs(db_path, exist_ok=True)
        batch_size = 64
        if len(self.content_docs) <= batch_size:
            vectorstore = Chroma.from_documents(self.content_docs, self.embedding_model, persist_directory=db_path)
        else:
            first_batch = self.content_docs[:batch_size]
            vectorstore = Chroma.from_documents(first_batch, self.embedding_model, persist_directory=db_path)
            for i in range(batch_size, len(self.content_docs), batch_size):
                batch = self.content_docs[i : i + batch_size]
                vectorstore.add_documents(batch)
        self.content_vector_store = vectorstore

    def load_title_vector_store(self, db_path: str):
        self.title_vector_store = Chroma(persist_directory=db_path, embedding_function=self.embedding_model)

    def load_content_vector_store(self, db_path: str):
        self.content_vector_store = Chroma(persist_directory=db_path, embedding_function=self.embedding_model)

    def similarity_title_search(self, query: str, k: int = 3) -> List[Document]:
        if self.title_vector_store is None:
            raise ValueError("title_vector_store not loaded/built")
        return self.title_vector_store.similarity_search(query, k=k)

    def similarity_content_search(self, query: str, k: int = 3) -> List[Document]:
        if self.content_vector_store is None:
            raise ValueError("content_vector_store not loaded/built")
        return self.content_vector_store.similarity_search(query, k=k)

    def exact_title_search(self, keyword: str) -> List[Document]:
        """
        精确title匹配：返回title中包含关键词的所有文档
        用于确保关键事件类型（引种、入群等）的规则稳定检索
        """
        if not self.title_docs:
            return []
        results = []
        for doc in self.title_docs:
            title = doc.metadata.get("title", "")
            if keyword in title:
                results.append(doc)
        return results
