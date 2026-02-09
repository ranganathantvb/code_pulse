import datetime
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from code_pulse.config import get_settings
from code_pulse.logger import setup_logging
from code_pulse import telemetry

logger = setup_logging(__name__)


@dataclass(frozen=True)
class RagLogEvent:
    event: str
    service: str
    env: str
    namespace: str
    rule_key: Optional[str]
    request_id: Optional[str]
    retrieval_id: str
    timestamp_ms: int
    start_time_ms: int
    end_time_ms: int
    duration_ms: int
    vector_store: Optional[str]
    index_name: str
    embedding_model: str
    reranker_model: Optional[str]
    top_k: int
    candidate_count: Optional[int]
    returned_count: int
    scores: Optional[List[float]]
    score_min: Optional[float]
    score_max: Optional[float]
    score_avg: Optional[float]
    cache_hit: bool
    cache_key_hash: Optional[str]
    cache_ttl_seconds: Optional[int]
    status: str
    error_class: Optional[str]
    error_message: Optional[str]
    query_hash: Optional[str]
    query_len: Optional[int]

    def as_dict(self) -> dict:
        return {
            "event": self.event,
            "service": self.service,
            "env": self.env,
            "namespace": self.namespace,
            "rule_key": self.rule_key,
            "request_id": self.request_id,
            "retrieval_id": self.retrieval_id,
            "timestamp_ms": self.timestamp_ms,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "duration_ms": self.duration_ms,
            "vector_store": self.vector_store,
            "index_name": self.index_name,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "top_k": self.top_k,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "scores": self.scores,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_avg": self.score_avg,
            "cache_hit": self.cache_hit,
            "cache_key_hash": self.cache_key_hash,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "status": self.status,
            "error_class": self.error_class,
            "error_message": self.error_message,
            "query_hash": self.query_hash,
            "query_len": self.query_len,
        }


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _short_error_message(message: str, limit: int = 200) -> str:
    text = (message or "").strip().replace("\n", " ")
    return text[:limit]


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _vector_store_name(store: Optional[FAISS]) -> Optional[str]:
    if store is None:
        return None
    return store.__class__.__name__.lower()


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _score_summary(scores: Sequence[float], returned_count: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if returned_count <= 0 or not scores:
        return None, None, None
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    return float(min_score), float(max_score), float(avg_score)


def _current_request_id() -> Optional[str]:
    try:
        import newrelic.agent  # type: ignore

        trace_id = getattr(newrelic.agent, "current_trace_id", None)
        if callable(trace_id):
            value = trace_id()
            return str(value) if value else None
        txn = newrelic.agent.current_transaction()
        if txn:
            trace_value = getattr(txn, "trace_id", None) or getattr(txn, "guid", None)
            return str(trace_value) if trace_value else None
    except Exception:
        return None
    return None


def _emit_rag_log(event: RagLogEvent) -> None:
    logger.info("%s", json.dumps(event.as_dict(), separators=(",", ":")))


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
        cache_hit = path.exists()
        try:
            telemetry.record_rag_cache_hit(cache_hit)
        except Exception:
            pass
        if telemetry.cache_metrics_enabled():
            try:
                telemetry.log_cache_access(
                    cache_name="rag_vectorstore",
                    cache_hit=cache_hit,
                    key_hash=telemetry.hash_key(namespace),
                    request_id=telemetry.get_request_id(),
                )
            except Exception:
                pass
        if cache_hit:
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
        start_time_ms = _now_ms()
        retrieval_id = str(uuid.uuid4())
        request_id = _current_request_id()
        service_name = os.getenv("CODEPULSE_SERVICE_NAME", "codepulse-agent-fastapi")
        env_name = os.getenv("CODEPULSE_ENV", "dev")
        query_hash = _hash_query(question)
        query_len = len(question or "")
        store = self._load_vectorstore(namespace)
        vector_store = _vector_store_name(store)
        candidates: List[Tuple[Document, float]] = []
        docs: List[Document] = []
        scores: List[float] = []
        status = "ok"
        error_class = None
        error_message = None
        try:
            if not store:
                docs = []
            elif not metadata_filter:
                candidates = store.similarity_search_with_score(question, k=k)
            else:
                try:
                    candidates = store.similarity_search_with_score(question, k=k, filter=metadata_filter)
                except TypeError:
                    candidates = store.similarity_search_with_score(question, k=max(k * 5, 20))
                    filtered: List[Tuple[Document, float]] = []
                    for doc, score in candidates:
                        if all(doc.metadata.get(key) == value for key, value in metadata_filter.items()):
                            filtered.append((doc, score))
                            if len(filtered) >= k:
                                break
                    candidates = filtered
            docs = [doc for doc, _ in candidates] if candidates else docs
            scores = [float(score) for _, score in candidates] if candidates else []
            return docs
        except Exception as exc:
            status = "error"
            error_class = exc.__class__.__name__
            error_message = _short_error_message(str(exc))
            raise
        finally:
            end_time_ms = _now_ms()
            duration_ms = max(0, end_time_ms - start_time_ms)
            returned_count = len(docs)
            score_min, score_max, score_avg = _score_summary(scores, returned_count)
            event = RagLogEvent(
                event="rag_retrieval",
                service=service_name,
                env=env_name,
                namespace=namespace,
                rule_key=None,
                request_id=request_id,
                retrieval_id=retrieval_id,
                timestamp_ms=end_time_ms,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                duration_ms=duration_ms,
                vector_store=vector_store,
                index_name=namespace,
                embedding_model=self.model_name,
                reranker_model=None,
                top_k=k,
                candidate_count=len(candidates) if candidates else 0,
                returned_count=returned_count,
                scores=scores[:10] if scores else [],
                score_min=score_min,
                score_max=score_max,
                score_avg=score_avg,
                cache_hit=False,
                cache_key_hash=None,
                cache_ttl_seconds=None,
                status=status,
                error_class=error_class,
                error_message=error_message,
                query_hash=query_hash,
                query_len=query_len,
            )
            _emit_rag_log(event)
            try:
                telemetry.record_rag_metrics(
                    duration_ms=duration_ms,
                    top_k=k,
                    chunks_used=returned_count,
                    cache_hit=None,
                    query_len=query_len,
                )
            except Exception:
                pass
            if telemetry.perf_metrics_enabled():
                try:
                    telemetry.log_latency("rag_retrieval", duration_ms, request_id)
                except Exception:
                    pass

    def lookup_by_metadata(
        self,
        namespace: str,
        metadata_filter: dict[str, str],
        query_text: Optional[str] = None,
    ) -> List[Document]:
        start_time_ms = _now_ms()
        retrieval_id = str(uuid.uuid4())
        request_id = _current_request_id()
        service_name = os.getenv("CODEPULSE_SERVICE_NAME", "codepulse-agent-fastapi")
        env_name = os.getenv("CODEPULSE_ENV", "dev")
        query_hash = _hash_query(query_text) if query_text else None
        query_len = len(query_text or "") if query_text else None
        store = self._load_vectorstore(namespace)
        vector_store = _vector_store_name(store)
        docs: List[Document] = []
        scores: List[float] = []
        status = "ok"
        error_class = None
        error_message = None
        try:
            if not store:
                docs = []
            else:
                docstore = getattr(store, "docstore", None)
                raw_dict = getattr(docstore, "_dict", None)
                if isinstance(raw_dict, dict):
                    matches: List[Document] = []
                    for doc in raw_dict.values():
                        if all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                            matches.append(doc)
                    docs = matches
                else:
                    try:
                        docs = store.similarity_search("sonar rule", k=50, filter=metadata_filter)
                    except TypeError:
                        candidates = store.similarity_search("sonar rule", k=50)
                        filtered: List[Document] = []
                        for doc in candidates:
                            if all(doc.metadata.get(key) == value for key, value in metadata_filter.items()):
                                filtered.append(doc)
                        docs = filtered
            if docs and query_text:
                query_vector = self._embeddings.embed_query(query_text)
                doc_vectors = self._embeddings.embed_documents([doc.page_content for doc in docs])
                scores = [float(_cosine_similarity(query_vector, vec)) for vec in doc_vectors]
            return docs
        except Exception as exc:
            status = "error"
            error_class = exc.__class__.__name__
            error_message = _short_error_message(str(exc))
            raise
        finally:
            end_time_ms = _now_ms()
            duration_ms = max(0, end_time_ms - start_time_ms)
            returned_count = len(docs)
            score_min, score_max, score_avg = _score_summary(scores, returned_count)
            event = RagLogEvent(
                event="rag_retrieval",
                service=service_name,
                env=env_name,
                namespace=namespace,
                rule_key=metadata_filter.get("rule_key"),
                request_id=request_id,
                retrieval_id=retrieval_id,
                timestamp_ms=end_time_ms,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                duration_ms=duration_ms,
                vector_store=vector_store,
                index_name=namespace,
                embedding_model=self.model_name,
                reranker_model=None,
                top_k=len(docs),
                candidate_count=returned_count,
                returned_count=returned_count,
                scores=scores[:10] if scores else [],
                score_min=score_min,
                score_max=score_max,
                score_avg=score_avg,
                cache_hit=False,
                cache_key_hash=None,
                cache_ttl_seconds=None,
                status=status,
                error_class=error_class,
                error_message=error_message,
                query_hash=query_hash,
                query_len=query_len,
            )
            _emit_rag_log(event)
            try:
                telemetry.record_rag_metrics(
                    duration_ms=duration_ms,
                    top_k=len(docs),
                    chunks_used=returned_count,
                    cache_hit=None,
                    query_len=query_len,
                )
            except Exception:
                pass
            if telemetry.perf_metrics_enabled():
                try:
                    telemetry.log_latency("rag_retrieval", duration_ms, request_id)
                except Exception:
                    pass
