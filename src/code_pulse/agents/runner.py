import asyncio
import os
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from code_pulse.agents.tools import Tooling, ToolResult
from code_pulse.logger import setup_logging
from code_pulse.memory import MemoryStore, Message

logger = setup_logging(__name__)


class Responder:
    def __init__(self):
        self.llm = None

        # Prefer local Ollama deepseek-coder if available; no OpenAI fallback.
        try:
            from langchain_ollama import ChatOllama  # type: ignore

            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
            self.llm = ChatOllama(
                model="deepseek-coder:6.7b",
                temperature=0.2,
                base_url=base_url,
            )
        except Exception:
            self.llm = None

        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful engineering agent.\n"
            "Task: {task}\n"
            "Context:\n{context}\n"
            "Provide a concise plan or answer."
        )
        self.parser = StrOutputParser()

    async def generate(self, task: str, context: str) -> str:
        if self.llm:
            chain = self.prompt | self.llm | self.parser
            return await chain.ainvoke({"task": task, "context": context})
        return f"Task: {task}\nContext Summary:\n{context}"


class AgentRunner:
    def __init__(self, responder: Optional[Responder] = None):
        self.tooling = Tooling()
        self.memory = MemoryStore()
        self.responder = responder or Responder()

    async def _execute_tools(
        self, tools: List[str], tool_args: Dict[str, Dict[str, str]], namespace: str
    ) -> List[ToolResult]:
        tasks = []
        for name in tools:
            if name == "git" and "git" in tool_args:
                args = tool_args["git"]
                tasks.append(self.tooling.use_git(args["owner"], args["repo"]))
            elif name == "sonar":
                args = tool_args.get("sonar", {})
                project_key = args.get("project_key")
                question = args.get("question") or args.get("task") or ""
                # Use the same namespace for RAG; default to "sonar" if not set.
                sonar_namespace = args.get("namespace") or namespace or "sonar"
                tasks.append(self.tooling.use_sonar(project_key, question=question or "", namespace=sonar_namespace))
            elif name == "jira" and "jira" in tool_args:
                args = tool_args["jira"]
                tasks.append(self.tooling.use_jira(args["jql"], args.get("create")))
            elif name == "rag":
                args = tool_args.get("rag", {})
                question = args.get("question") or args.get("task") or ""
                tasks.append(asyncio.to_thread(self.tooling.use_rag, question, namespace))
        if not tasks:
            return []
        results = await asyncio.gather(*tasks)
        return results

    async def run(
        self,
        task: str,
        tools: List[str],
        memory_key: str,
        namespace: str = "default",
        tool_args: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        tool_args = tool_args or {}
        self.memory.append(memory_key, Message(role="user", content=task))
        results = await self._execute_tools(tools, tool_args, namespace)
        context_parts = []
        for res in results:
            context_parts.append(f"{res.name}: {res.output}")
        context = "\n".join(context_parts) if context_parts else "No tools invoked."
        answer = await self.responder.generate(task, context)
        self.memory.append(memory_key, Message(role="assistant", content=answer))
        return {"answer": answer, "tools": [res.__dict__ for res in results], "memory_key": memory_key}
