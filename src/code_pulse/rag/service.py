import datetime
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

    def _persist_docs(self, docs: List[Document], namespace: str, split: bool = True) -> int:
        if not docs:
            return 0
        chunks = self.splitter.split_documents(docs) if split else docs
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
            logger.info("RAG ingest: saving file %s to %s", name, file_path)
            with file_path.open("wb") as f:
                shutil.copyfileobj(stream, f)
            loader = self._loader_for(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = str(file_path)
            docs.extend(loaded_docs)

        logger.info("RAG ingest: loaded %d document(s) for namespace '%s'", len(docs), namespace)
        return self._persist_docs(docs, namespace)

    @staticmethod
    def _html_to_text(raw_html: str) -> str:
        # Lightweight HTML -> plaintext conversion without extra deps.
        text = re.sub(r"<[^>]+>", " ", raw_html or "")
        text = html.unescape(text)
        return " ".join(text.split())

    @staticmethod
    def _first_sentence(text: str, limit: int = 240) -> str:
        if not text:
            return ""
        match = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
        sentence = match[0] if match else text.strip()
        return sentence[:limit].strip()

    def _extract_section_texts(self, rule: dict) -> dict[str, str]:
        sections = rule.get("descriptionSections") or []
        extracted: dict[str, str] = {}
        for section in sections:
            key = (section.get("key") or "").strip().lower()
            content = self._html_to_text(section.get("content", ""))
            if not content:
                continue
            if "noncompliant" in key:
                extracted["noncompliant"] = content
            elif "compliant" in key or "solution" in key or "fix" in key:
                extracted["compliant"] = content
            elif "remediation" in key or "recommend" in key:
                extracted["remediation"] = content
            elif "why" in key or "rationale" in key or "root_cause" in key:
                extracted["rationale"] = content
        return extracted

    def ingest_sonar_rules(self, json_path: str | Path, namespace: str = "sonar") -> int:
        """Ingest Sonar rule JSON into a vector store using section-based chunks."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Sonar rules file not found: {path}")
        payload = json.loads(path.read_text())
        rules = payload.get("rules", [])
        docs: List[Document] = []
        fetched_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        logger.info("Sonar ingest: loaded %d rule(s) from %s", len(rules), path)

        for rule in rules:
            rule_key = rule.get("key") or rule.get("rule_key")
            if not rule_key:
                logger.warning("Sonar ingest: skipping rule without key")
                continue
            lang = rule.get("lang")
            name = rule.get("name") or rule.get("title") or rule_key
            desc_html = rule.get("mdDesc") or rule.get("htmlDesc") or rule.get("description") or ""
            desc_text = self._html_to_text(desc_html)
            section_texts = self._extract_section_texts(rule)

            base_meta = {
                "doc_type": "sonar_rule",
                "rule_key": rule_key,
                "language": lang,
                "section": None,
                "source": "sonar_rule_data.json",
                "fetched_at": fetched_at,
            }

            summary_text = self._first_sentence(desc_text) if desc_text else ""
            summary_chunk = name if not summary_text else f"{name}\n{summary_text}"
            if summary_chunk.strip():
                logger.info("Sonar ingest: rule=%s section=summary chars=%d", rule_key, len(summary_chunk))
                docs.append(
                    Document(
                        page_content=summary_chunk.strip(),
                        metadata={**base_meta, "section": "summary"},
                    )
                )

            rationale_text = section_texts.get("rationale") or desc_text
            if rationale_text:
                logger.info("Sonar ingest: rule=%s section=rationale chars=%d", rule_key, len(rationale_text))
                docs.append(
                    Document(
                        page_content=rationale_text,
                        metadata={**base_meta, "section": "rationale"},
                    )
                )

            for section in ("noncompliant", "compliant", "remediation"):
                content = section_texts.get(section, "")
                if content:
                    logger.info("Sonar ingest: rule=%s section=%s chars=%d", rule_key, section, len(content))
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={**base_meta, "section": section},
                        )
                    )

        logger.info("Sonar ingest: total chunks=%d namespace='%s'", len(docs), namespace)
        return self._persist_docs(docs, namespace, split=False)

    def query(
        self,
        question: str,
        namespace: str = "default",
        k: int = 4,
        metadata_filter: Optional[dict[str, str]] = None,
    ) -> List[Document]:
        store = self._load_vectorstore(namespace)
        if not store:
            return []
        if not metadata_filter:
            return store.similarity_search(question, k=k)
        try:
            return store.similarity_search(question, k=k, filter=metadata_filter)
        except TypeError:
            docs = store.similarity_search(question, k=max(k * 5, 20))
            filtered: List[Document] = []
            for doc in docs:
                if all(doc.metadata.get(key) == value for key, value in metadata_filter.items()):
                    filtered.append(doc)
                    if len(filtered) >= k:
                        break
            return filtered

    def lookup_by_metadata(
        self,
        namespace: str,
        metadata_filter: dict[str, str],
    ) -> List[Document]:
        store = self._load_vectorstore(namespace)
        if not store:
            return []
        docstore = getattr(store, "docstore", None)
        raw_dict = getattr(docstore, "_dict", None)
        if isinstance(raw_dict, dict):
            matches: List[Document] = []
            for doc in raw_dict.values():
                if all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    matches.append(doc)
            return matches
        try:
            return store.similarity_search("sonar rule", k=50, filter=metadata_filter)
        except Exception:
            return []
