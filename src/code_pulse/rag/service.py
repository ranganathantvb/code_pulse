import io
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from code_pulse.config import get_settings
from code_pulse.logger import setup_logging

logger = setup_logging(__name__)


class RAGService:
    def __init__(self, data_dir: Path | None = None, model_name: Optional[str] = None):
        settings = get_settings()
        self.data_dir = data_dir or settings.data_dir
        self.model_name = model_name or settings.rag_embeddings_model
        self.ingest_dir = self.data_dir.joinpath("ingest")
        self.vector_dir = self.data_dir.joinpath("vectorstores")
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    def _store_path(self, namespace: str) -> Path:
        safe = namespace.replace("/", "_")
        return self.vector_dir.joinpath(safe)

    def _load_vectorstore(self, namespace: str) -> Optional[FAISS]:
        path = self._store_path(namespace)
        if path.exists():
            return FAISS.load_local(path, self._embeddings, allow_dangerous_deserialization=True)
        return None

    def _loader_for(self, file_path: Path):
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        if suffix == ".csv":
            return CSVLoader(str(file_path))
        return TextLoader(str(file_path), autodetect_encoding=True)

    def ingest_files(self, files: Iterable[tuple[str, io.BytesIO]], namespace: str = "default") -> int:
        docs: List[Document] = []
        for name, stream in files:
            destination = self.ingest_dir.joinpath(namespace)
            destination.mkdir(parents=True, exist_ok=True)
            file_path = destination.joinpath(name)
            with file_path.open("wb") as f:
                shutil.copyfileobj(stream, f)
            loader = self._loader_for(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = str(file_path)
            docs.extend(loaded_docs)

        if not docs:
            return 0
        chunks = self.splitter.split_documents(docs)
        store = self._load_vectorstore(namespace) or FAISS.from_documents(chunks, self._embeddings)
        if store:
            store.add_documents(chunks)
        store.save_local(self._store_path(namespace))
        return len(chunks)

    def query(self, question: str, namespace: str = "default", k: int = 4) -> List[Document]:
        store = self._load_vectorstore(namespace)
        if not store:
            return []
        return store.similarity_search(question, k=k)
