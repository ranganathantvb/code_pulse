import json

from code_pulse.rag.service import RagLogEvent


def test_rag_log_event_has_required_fields():
    event = RagLogEvent(
        event="rag_retrieval",
        service="codepulse-agent-fastapi",
        env="test",
        namespace="sonar",
        rule_key="java:S108",
        request_id="trace-abc",
        retrieval_id="retrieval-123",
        timestamp_ms=1234567890,
        start_time_ms=1234567000,
        end_time_ms=1234567890,
        duration_ms=890,
        vector_store="faiss",
        index_name="sonar",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model=None,
        top_k=4,
        candidate_count=4,
        returned_count=2,
        scores=[0.12, 0.08],
        score_min=0.08,
        score_max=0.12,
        score_avg=0.10,
        cache_hit=False,
        cache_key_hash=None,
        cache_ttl_seconds=None,
        status="ok",
        error_class=None,
        error_message=None,
        query_hash="abc123",
        query_len=12,
    )

    payload = json.loads(json.dumps(event.as_dict(), separators=(",", ":")))

    assert payload["event"] == "rag_retrieval"
    assert payload["service"] == "codepulse-agent-fastapi"
    assert payload["env"] == "test"
    assert payload["namespace"] == "sonar"
    assert payload["rule_key"] == "java:S108"
    assert payload["request_id"] == "trace-abc"
    assert payload["retrieval_id"] == "retrieval-123"
    assert payload["timestamp_ms"] == 1234567890
    assert payload["duration_ms"] == 890
    assert payload["vector_store"] == "faiss"
    assert payload["index_name"] == "sonar"
    assert payload["embedding_model"]
    assert payload["top_k"] == 4
    assert payload["candidate_count"] == 4
    assert payload["returned_count"] == 2
    assert payload["scores"] == [0.12, 0.08]
    assert payload["score_min"] == 0.08
    assert payload["score_max"] == 0.12
    assert payload["score_avg"] == 0.10
    assert payload["cache_hit"] is False
    assert payload["status"] == "ok"
