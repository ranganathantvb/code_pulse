import html
import io
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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

    def _persist_docs(self, docs: List[Document], namespace: str) -> int:
        if not docs:
            return 0
        chunks = self.splitter.split_documents(docs)
        store = self._load_vectorstore(namespace) or FAISS.from_documents(chunks, self._embeddings)
        if store:
            store.add_documents(chunks)
        store.save_local(self._store_path(namespace))
        return len(chunks)

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

        return self._persist_docs(docs, namespace)

    @staticmethod
    def _html_to_text(raw_html: str) -> str:
        # Lightweight HTML -> plaintext conversion without extra deps.
        text = re.sub(r"<[^>]+>", " ", raw_html or "")
        text = html.unescape(text)
        return " ".join(text.split())

    def ingest_sonar_rules(self, json_path: str | Path, namespace: str = "sonar") -> int:
        """Ingest Sonar rule JSON (rules/search output) into a vector store."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Sonar rules file not found: {path}")
        payload = json.loads(path.read_text())
        rules = payload.get("rules", [])
        docs: List[Document] = []

        for rule in rules:
            desc_html = rule.get("mdDesc") or rule.get("htmlDesc") or ""
            desc_text = self._html_to_text(desc_html)
            sections = rule.get("descriptionSections") or []
            section_bits = []
            for section in sections:
                key = section.get("key")
                content = self._html_to_text(section.get("content", ""))
                if content:
                    label = f"{key}:" if key else ""
                    section_bits.append(f"{label} {content}".strip())

            content_parts = [
                f"Rule {rule.get('key')}: {rule.get('name')}",
                f"Type: {rule.get('type')} Severity: {rule.get('severity')}",
                f"Language: {rule.get('langName') or rule.get('lang')}",
            ]
            if rule.get("tags"):
                content_parts.append(f"Tags: {', '.join(rule.get('tags'))}")
            if desc_text:
                content_parts.append(f"Summary: {desc_text}")
            if section_bits:
                content_parts.append("Details: " + " ".join(section_bits))

            doc = Document(
                page_content="\n".join(content_parts),
                metadata={
                    "source": str(path),
                    "rule_key": rule.get("key"),
                    "type": rule.get("type"),
                    "severity": rule.get("severity"),
                },
            )
            docs.append(doc)

        return self._persist_docs(docs, namespace)

    def query(self, question: str, namespace: str = "default", k: int = 4) -> List[Document]:
        store = self._load_vectorstore(namespace)
        if not store:
            return []
        return store.similarity_search(question, k=k)
