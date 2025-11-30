import io
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from code_pulse.agents import AgentRunner
from code_pulse.logger import setup_logging
from code_pulse.memory import MemoryStore
from code_pulse.rag.service import RAGService

logger = setup_logging(__name__)

app = FastAPI(title="Code Pulse", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()
agent_runner = AgentRunner()
memory_store = MemoryStore()


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask the knowledge base")
    namespace: str = Field(default="default")


class AgentRequest(BaseModel):
    task: str
    tools: List[str]
    memory_key: str
    namespace: str = Field(default="default")
    tool_args: Optional[Dict[str, Dict[str, str]]] = None


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
    file_streams = []
    for f in files:
        content = await f.read()
        file_streams.append((f.filename, io.BytesIO(content)))
    try:
        chunks = rag_service.ingest_files(file_streams, namespace=namespace)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to ingest files")
        raise HTTPException(status_code=400, detail=str(exc))
    return {"namespace": namespace, "chunks": chunks}


@app.post("/rag/query")
async def rag_query(body: QueryRequest):
    docs = rag_service.query(body.question, namespace=body.namespace)
    return {
        "matches": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
    }


@app.post("/agents/run")
async def run_agent(body: AgentRequest):
    try:
        result = await agent_runner.run(
            body.task,
            body.tools,
            body.memory_key,
            namespace=body.namespace,
            tool_args=body.tool_args,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent run failed")
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/memory/{memory_key}")
async def memory(memory_key: str, limit: int = 50):
    history = memory_store.history(memory_key, limit=limit)
    return {"memory_key": memory_key, "history": [vars(m) for m in history]}
