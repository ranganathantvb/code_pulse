import contextvars
import hashlib
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Deque, Dict, Optional, Tuple
from urllib.parse import urlparse

from code_pulse.logger import setup_logging

logger = setup_logging("code_pulse.telemetry")

_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "codepulse_request_id", default=None
)
_sonar_run_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "codepulse_sonar_run_id", default=None
)
_latency_lock = Lock()
_latency_samples: Dict[str, Deque[int]] = {}
_latest_latency: Dict[str, int] = {}
_rag_lock = Lock()
_rag_metrics: Dict[str, Any] = {
    "rag_retrieval_ms": 0.0,
    "rag_top_k": 0,
    "rag_chunks_used": 0,
    "rag_cache_hit": False,
    "rag_query_len": 0,
}
_token_lock = Lock()
_token_usage: Dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}
_sonar_lock = Lock()
_sonar_counts: Dict[str, Any] = {
    "total": 0,
    "by_type": {},
    "by_rule": {},
}
_pulse_emitter: Optional[Callable[[Dict[str, Any]], None]] = None


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _float_env(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def perf_metrics_enabled() -> bool:
    return _bool_env("CODEPULSE_ENABLE_PERF_METRICS", False)


def token_metrics_enabled() -> bool:
    return _bool_env("CODEPULSE_ENABLE_TOKEN_METRICS", False)


def cache_metrics_enabled() -> bool:
    return _bool_env("CODEPULSE_ENABLE_CACHE_METRICS", False)


def sonar_structured_logs_enabled() -> bool:
    return _bool_env("CODEPULSE_ENABLE_SONAR_STRUCTURED_LOGS", False)


def service_name() -> str:
    return os.getenv("CODEPULSE_SERVICE_NAME", "codepulse-agent-fastapi")


def set_request_id(value: Optional[str]) -> Optional[contextvars.Token]:
    try:
        return _request_id_var.set(value)
    except Exception:
        return None


def reset_request_id(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _request_id_var.reset(token)
    except Exception:
        return


def set_sonar_run_id(value: Optional[str]) -> Optional[contextvars.Token]:
    try:
        return _sonar_run_id_var.set(value)
    except Exception:
        return None


def reset_sonar_run_id(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _sonar_run_id_var.reset(token)
    except Exception:
        return


def get_sonar_run_id() -> Optional[str]:
    try:
        return _sonar_run_id_var.get()
    except Exception:
        return None


def _newrelic_request_id() -> Optional[str]:
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


def get_request_id() -> Optional[str]:
    value = _request_id_var.get()
    if value:
        return value
    return _newrelic_request_id()


def ensure_request_id() -> str:
    value = get_request_id()
    if value:
        return value
    value = str(uuid.uuid4())
    set_request_id(value)
    return value


def monotonic_ms() -> int:
    return int(time.perf_counter() * 1000)


def resolve_agent_id() -> str:
    agent_id = os.getenv("CODEPULSE_AGENT_ID") or os.getenv("AGENT_ID")
    if agent_id:
        return agent_id
    api_url = os.getenv("AGENT_API_URL")
    if api_url:
        try:
            parsed = urlparse(api_url)
            host = parsed.hostname or "local"
            port = parsed.port or 8000
            host_label = "local" if host in {"localhost", "127.0.0.1"} else host.replace(".", "-")
            return f"agent-{host_label}-{port}"
        except Exception:
            pass
    return "agent-local-8000"


def pulse_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pulse(pulse_type: str, status: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": pulse_timestamp(),
        "agent_id": resolve_agent_id(),
        "pulse_type": pulse_type,
        "status": status,
        "payload": payload,
    }


def set_pulse_emitter(emitter: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    global _pulse_emitter
    _pulse_emitter = emitter


def emit_pulse(pulse_type: str, status: str, payload: Dict[str, Any]) -> None:
    if _pulse_emitter is None:
        return
    try:
        _pulse_emitter(build_pulse(pulse_type, status, payload))
    except Exception:
        return


def _record_latency(operation: str, duration_ms: int) -> None:
    if duration_ms < 0:
        return
    with _latency_lock:
        if operation not in _latency_samples:
            _latency_samples[operation] = deque(maxlen=200)
        _latency_samples[operation].append(duration_ms)
        _latest_latency[operation] = duration_ms


def get_latency_stats(operation: str) -> Dict[str, float]:
    with _latency_lock:
        samples = list(_latency_samples.get(operation, []))
        latest = float(_latest_latency.get(operation, 0))
    if not samples:
        return {"latest": latest, "avg": 0.0, "p95": 0.0}
    samples.sort()
    avg = sum(samples) / len(samples)
    idx = int(0.95 * (len(samples) - 1))
    p95 = samples[idx]
    return {"latest": latest, "avg": float(avg), "p95": float(p95)}


def record_rag_metrics(
    duration_ms: int,
    top_k: int,
    chunks_used: int,
    cache_hit: Optional[bool],
    query_len: Optional[int],
) -> None:
    with _rag_lock:
        _rag_metrics["rag_retrieval_ms"] = float(duration_ms)
        _rag_metrics["rag_top_k"] = int(top_k)
        _rag_metrics["rag_chunks_used"] = int(chunks_used)
        if cache_hit is not None:
            _rag_metrics["rag_cache_hit"] = bool(cache_hit)
        if query_len is not None:
            _rag_metrics["rag_query_len"] = int(query_len)


def record_rag_cache_hit(cache_hit: bool) -> None:
    with _rag_lock:
        _rag_metrics["rag_cache_hit"] = bool(cache_hit)


def get_rag_metrics_snapshot() -> Dict[str, Any]:
    with _rag_lock:
        return dict(_rag_metrics)


def record_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    with _token_lock:
        _token_usage["prompt_tokens"] = int(prompt_tokens)
        _token_usage["completion_tokens"] = int(completion_tokens)
        _token_usage["total_tokens"] = int(total_tokens)


def get_token_usage_snapshot() -> Dict[str, int]:
    with _token_lock:
        return dict(_token_usage)


def record_sonar_counts(total: int, by_type: Dict[str, int], by_rule: Dict[str, int]) -> None:
    with _sonar_lock:
        _sonar_counts["total"] = int(total)
        _sonar_counts["by_type"] = dict(by_type)
        _sonar_counts["by_rule"] = dict(by_rule)


def get_sonar_counts_snapshot() -> Dict[str, Any]:
    with _sonar_lock:
        return dict(_sonar_counts)


def emit_event(payload: Dict[str, Any]) -> None:
    try:
        logger.info("%s", json.dumps(payload, separators=(",", ":")))
    except Exception:
        return


def emit_structured_log(payload: Dict[str, Any]) -> None:
    try:
        if not sonar_structured_logs_enabled():
            return
        if "service" not in payload:
            payload["service"] = service_name()
        logger.info("%s", json.dumps(payload, separators=(",", ":")))
    except Exception:
        return


def _process_snapshot() -> Optional[Dict[str, float]]:
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=None)
        mem = process.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        vms_mb = mem.vms / (1024 * 1024)
        return {
            "cpu_percent": round(cpu_percent, 1),
            "rss_mb": round(rss_mb, 2),
            "vms_mb": round(vms_mb, 2),
        }
    except Exception:
        return None


def capture_process_resources() -> Optional[Dict[str, float]]:
    return _process_snapshot()


def log_process_resources(snapshot: Optional[Dict[str, float]], request_id: Optional[str]) -> None:
    if not snapshot:
        return
    payload = {
        "event": "codepulse_perf",
        "metric": "process_resources",
        "cpu_percent": snapshot.get("cpu_percent"),
        "rss_mb": snapshot.get("rss_mb"),
        "vms_mb": snapshot.get("vms_mb"),
        "request_id": request_id,
        "service": service_name(),
    }
    emit_event(payload)


def log_latency(operation: str, duration_ms: int, request_id: Optional[str]) -> None:
    try:
        _record_latency(operation, duration_ms)
    except Exception:
        pass
    payload = {
        "event": "codepulse_latency",
        "operation": operation,
        "duration_ms": duration_ms,
        "request_id": request_id,
        "service": service_name(),
    }
    emit_event(payload)


def log_queue_wait(queue_name: str, queue_wait_ms: int, request_id: Optional[str]) -> None:
    payload = {
        "event": "codepulse_queue",
        "queue_name": queue_name,
        "queue_wait_ms": queue_wait_ms,
        "request_id": request_id,
    }
    emit_event(payload)


def log_cache_access(cache_name: str, cache_hit: bool, key_hash: str, request_id: Optional[str]) -> None:
    payload = {
        "event": "codepulse_cache",
        "cache_name": cache_name,
        "cache_hit": cache_hit,
        "key_hash": key_hash,
        "request_id": request_id,
    }
    emit_event(payload)


def _token_costs() -> Optional[Tuple[float, float]]:
    prompt_cost = _float_env("CODEPULSE_TOKEN_COST_USD_PER_1K_PROMPT")
    completion_cost = _float_env("CODEPULSE_TOKEN_COST_USD_PER_1K_COMPLETION")
    if prompt_cost is None or completion_cost is None:
        return None
    return prompt_cost, completion_cost


def log_token_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    request_id: Optional[str],
) -> None:
    try:
        record_token_usage(prompt_tokens, completion_tokens, total_tokens)
    except Exception:
        pass
    payload: Dict[str, Any] = {
        "event": "codepulse_tokens",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_id": request_id,
    }
    costs = _token_costs()
    if costs:
        prompt_cost, completion_cost = costs
        estimated = (prompt_tokens / 1000.0 * prompt_cost) + (
            completion_tokens / 1000.0 * completion_cost
        )
        payload["estimated_cost_usd"] = round(estimated, 6)
    emit_event(payload)


def hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_token_usage(llm_output: Any) -> Optional[Dict[str, int]]:
    if not isinstance(llm_output, dict):
        return None
    usage = llm_output.get("token_usage") or llm_output.get("usage") or llm_output.get("usage_metadata")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    total_tokens = usage.get("total_tokens") or usage.get("total")
    if prompt_tokens is None or completion_tokens is None:
        return None
    total_tokens = total_tokens or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # noqa: BLE001
    BaseCallbackHandler = object  # type: ignore[assignment]


class TokenUsageCallback(BaseCallbackHandler):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.usage: Optional[Dict[str, int]] = None
        self.model_name: Optional[str] = None

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            self.usage = extract_token_usage(llm_output)
            self.model_name = llm_output.get("model_name") or llm_output.get("model")
        except Exception:
            return
