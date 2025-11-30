from fastapi.testclient import TestClient

from code_pulse.app import app
from code_pulse.memory import MemoryStore, Message


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_memory_roundtrip(tmp_path):
    store = MemoryStore(base_dir=tmp_path)
    store.append("demo", Message(role="user", content="hello"))
    history = store.history("demo")
    assert history
    assert history[0].content == "hello"
