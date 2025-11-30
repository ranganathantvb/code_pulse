import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from code_pulse.config import get_settings


@dataclass
class Message:
    role: str
    content: str


class MemoryStore:
    def __init__(self, base_dir: Path | None = None):
        settings = get_settings()
        self.base_dir = (base_dir or settings.data_dir).joinpath("memory")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file(self, key: str) -> Path:
        return self.base_dir.joinpath(f"{key}.jsonl")

    def append(self, key: str, message: Message) -> None:
        file_path = self._file(key)
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(message)) + "\n")

    def history(self, key: str, limit: int = 50) -> List[Message]:
        file_path = self._file(key)
        if not file_path.exists():
            return []
        messages: List[Message] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    messages.append(Message(**data))
                except json.JSONDecodeError:
                    continue
        return messages[-limit:]
