import asyncio
import io
import os
import random
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import newrelic.agent
from pydantic import BaseModel, Field
from starlette.requests import Request

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

from code_pulse.agents import AgentRunner
from code_pulse.logger import setup_logging
from code_pulse.memory import MemoryStore
from code_pulse.rag.service import RAGService
from code_pulse.mcp.sonar_mcp_server import router as sonar_router
from code_pulse import telemetry

logger = setup_logging(__name__)

if load_dotenv:
    load_dotenv()

try:
    newrelic.agent.initialize(os.getenv("NEW_RELIC_CONFIG_FILE"))
except Exception:
    pass

app = FastAPI(title="Code Pulse", version="0.1.0")
# Wrap the ASGI app so New Relic captures FastAPI requests as transactions
app = newrelic.agent.ASGIApplicationWrapper(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(sonar_router)

rag_service = RAGService()
agent_runner = AgentRunner()
memory_store = MemoryStore()

PULSE_URL = os.getenv("CODEPULSE_PULSE_URL", "http://localhost:8081/pulses")
_pulse_task: Optional[asyncio.Task] = None
_pulse_stop_event: Optional[asyncio.Event] = None
_pulse_queue: Optional[asyncio.Queue] = None


def _metric_payload() -> Dict[str, Any]:
    snapshot = telemetry.capture_process_resources() or {}
    cpu_pct = snapshot.get("cpu_percent") or 0.0
    rss_mb = snapshot.get("rss_mb") or 0.0
    req_stats = telemetry.get_latency_stats("request")
    rag_stats = telemetry.get_latency_stats("rag_retrieval")
    llm_stats = telemetry.get_latency_stats("rag_llm")
    latency_ms = round(req_stats.get("latest", 0.0), 2)
    p95_latency_ms = round(req_stats.get("p95", 0.0), 2)
    throughput_rps = 0.0
    rag_snapshot = telemetry.get_rag_metrics_snapshot()
    rag_retrieval_ms = round(rag_stats.get("latest", 0.0), 2)
    rag_rerank_ms = 0.0
    rag_llm_ms = round(llm_stats.get("latest", 0.0), 2)
    rag_total_ms = round(rag_retrieval_ms + rag_rerank_ms + rag_llm_ms, 2)
    rag_top_k = int(rag_snapshot.get("rag_top_k") or 0)
    rag_chunks_used = int(rag_snapshot.get("rag_chunks_used") or 0)
    token_snapshot = telemetry.get_token_usage_snapshot()
    rag_ctx_tokens_in = int(token_snapshot.get("prompt_tokens") or 0)
    rag_ctx_tokens_out = int(token_snapshot.get("completion_tokens") or 0)
    rag_cache_hit = bool(rag_snapshot.get("rag_cache_hit"))
    return {
        "metrics": {
            "cpu_pct": float(cpu_pct),
            "rss_mb": float(rss_mb),
            "latency_ms": latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "throughput_rps": throughput_rps,
            "rag_retrieval_ms": rag_retrieval_ms,
            "rag_rerank_ms": rag_rerank_ms,
            "rag_llm_ms": rag_llm_ms,
            "rag_total_ms": rag_total_ms,
            "rag_top_k": rag_top_k,
            "rag_chunks_used": rag_chunks_used,
            "rag_ctx_tokens_in": rag_ctx_tokens_in,
            "rag_ctx_tokens_out": rag_ctx_tokens_out,
            "rag_cache_hit": rag_cache_hit,
        },
        "tags": {
            "model": os.getenv("CODEPULSE_MODEL", "local"),
            "vector_store": os.getenv("CODEPULSE_VECTOR_STORE", "unknown"),
            "pipeline": os.getenv("CODEPULSE_PIPELINE", "default"),
        },
    }


def _error_payload(
    exc: Optional[Exception] = None,
    stage: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    stages = ["retrieval", "rerank", "llm", "postprocess", "unknown"]
    error_type = type(exc).__name__ if exc else "RuntimeError"
    message = str(exc) if exc else "unknown error"
    return {
        "error": {
            "type": error_type,
            "message": message,
            "stage": stage or random.choice(stages),
            "retryable": random.random() < 0.6,
        },
        "request_id": request_id,
    }


def _task_payload(task_name: str, stage: str, request_id: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"task": task_name, "stage": stage}
    if request_id:
        payload["request_id"] = request_id
    return payload


def _enqueue_pulse(payload: Dict[str, Any]) -> None:
    if _pulse_queue is None:
        return
    try:
        _pulse_queue.put_nowait(payload)
    except Exception:
        return


def _enqueue_task_pulse(task_name: str, stage: str, request_id: Optional[str], status: str) -> None:
    _enqueue_pulse(telemetry.build_pulse("TASK", status, _task_payload(task_name, stage, request_id)))


def _enqueue_error_pulse(exc: Exception, request_id: Optional[str], stage: Optional[str] = None) -> None:
    _enqueue_pulse(telemetry.build_pulse("ERROR", "ERROR", _error_payload(exc, stage=stage, request_id=request_id)))


async def _post_pulse(client: httpx.AsyncClient, payload: Dict[str, Any]) -> None:
    try:
        response = await client.post(PULSE_URL, json=payload, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Pulse post failed: %s", exc)


async def _pulse_sender(
    client: httpx.AsyncClient,
    stop_event: asyncio.Event,
    queue: asyncio.Queue,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            break
        await _post_pulse(client, item)
        if stop_event.is_set() and queue.empty():
            break


async def _heartbeat_loop(stop_event: asyncio.Event, queue: asyncio.Queue) -> None:
    while not stop_event.is_set():
        queue.put_nowait(telemetry.build_pulse("HEARTBEAT", "OK", {"status": "alive"}))
        await asyncio.sleep(5)


async def _metric_loop(stop_event: asyncio.Event, queue: asyncio.Queue) -> None:
    while not stop_event.is_set():
        queue.put_nowait(telemetry.build_pulse("METRIC", "OK", _metric_payload()))
        await asyncio.sleep(random.uniform(2, 5))


async def _pulse_background(stop_event: asyncio.Event, queue: asyncio.Queue) -> None:
    async with httpx.AsyncClient() as client:
        sender = asyncio.create_task(_pulse_sender(client, stop_event, queue))
        heartbeat = asyncio.create_task(_heartbeat_loop(stop_event, queue))
        metrics = asyncio.create_task(_metric_loop(stop_event, queue))
        tasks = [sender, heartbeat, metrics]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        if done:
            for finished in done:
                if finished.exception():
                    logger.warning("Pulse background task failed: %s", finished.exception())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@app.on_event("startup")
async def start_pulse_emitter() -> None:
    global _pulse_task, _pulse_stop_event, _pulse_queue
    if _pulse_task and not _pulse_task.done():
        return
    _pulse_stop_event = asyncio.Event()
    _pulse_queue = asyncio.Queue()
    telemetry.set_pulse_emitter(_enqueue_pulse)
    _pulse_task = asyncio.create_task(_pulse_background(_pulse_stop_event, _pulse_queue))


@app.on_event("shutdown")
async def stop_pulse_emitter() -> None:
    global _pulse_task, _pulse_stop_event, _pulse_queue
    if _pulse_stop_event:
        _pulse_stop_event.set()
    telemetry.set_pulse_emitter(None)
    if _pulse_queue:
        try:
            _pulse_queue.put_nowait(None)
        except Exception:
            pass
    if _pulse_task:
        _pulse_task.cancel()
        await asyncio.gather(_pulse_task, return_exceptions=True)
    _pulse_task = None
    _pulse_stop_event = None
    _pulse_queue = None


@app.middleware("http")
async def perf_metrics_middleware(request: Request, call_next):
    request_id = None
    token = None
    start_ms = None
    start_snapshot = None
    try:
        request_id = telemetry.ensure_request_id()
        token = telemetry.set_request_id(request_id)
        if telemetry.perf_metrics_enabled():
            start_ms = telemetry.monotonic_ms()
            start_snapshot = telemetry.capture_process_resources()
    except Exception:
        start_ms = None
        start_snapshot = None
    response = await call_next(request)
    try:
        if telemetry.perf_metrics_enabled() and start_ms is not None:
            end_ms = telemetry.monotonic_ms()
            duration_ms = max(0, end_ms - start_ms)
            telemetry.log_latency("request", duration_ms, request_id)
            end_snapshot = telemetry.capture_process_resources()
            telemetry.log_process_resources(end_snapshot or start_snapshot, request_id)
    except Exception:
        pass
    finally:
        telemetry.reset_request_id(token)
    return response


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask the knowledge base")
    namespace: str = Field(default="default")


class AgentRequest(BaseModel):
    task: str
    tools: List[str]
    memory_key: str
    namespace: str = Field(default="default")
    tool_args: Optional[Dict[str, Dict[str, Any]]] = None


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Code Pulse API", "docs": "/docs", "health": "/health"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(namespace: str = "default", files: List[UploadFile] = File(...)):
    request_id = telemetry.ensure_request_id()
    _enqueue_task_pulse("ingest", "start", request_id, "OK")
    status = "OK"
    file_streams = []
    for f in files:
        content = await f.read()
        file_streams.append((f.filename, io.BytesIO(content)))
    try:
        chunks = rag_service.ingest_files(file_streams, namespace=namespace)
    except Exception as exc:  # noqa: BLE001
        status = "ERROR"
        _enqueue_error_pulse(exc, request_id, stage="postprocess")
        logger.exception("Failed to ingest files")
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        _enqueue_task_pulse("ingest", "end", request_id, status)
    return {"namespace": namespace, "chunks": chunks}


@app.post("/rag/query")
async def rag_query(body: QueryRequest):
    request_id = telemetry.ensure_request_id()
    _enqueue_task_pulse("rag_query", "start", request_id, "OK")
    status = "OK"
    try:
        docs = rag_service.query(body.question, namespace=body.namespace)
    except Exception as exc:  # noqa: BLE001
        status = "ERROR"
        _enqueue_error_pulse(exc, request_id, stage="retrieval")
        raise
    finally:
        _enqueue_task_pulse("rag_query", "end", request_id, status)
    return {
        "matches": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
    }


@app.post("/agents/run")
async def run_agent(body: AgentRequest):
    request_id = telemetry.ensure_request_id()
    _enqueue_task_pulse("agent_run", "start", request_id, "OK")
    status = "OK"
    try:
        result = await agent_runner.run(
            body.task,
            body.tools,
            body.memory_key,
            namespace=body.namespace,
            tool_args=body.tool_args,
        )
    except Exception as exc:  # noqa: BLE001
        status = "ERROR"
        _enqueue_error_pulse(exc, request_id, stage="postprocess")
        logger.exception("Agent run failed")
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        _enqueue_task_pulse("agent_run", "end", request_id, status)
    return result


@app.get("/memory/{memory_key}")
async def memory(memory_key: str, limit: int = 50):
    history = memory_store.history(memory_key, limit=limit)
    return {"memory_key": memory_key, "history": [vars(m) for m in history]}
