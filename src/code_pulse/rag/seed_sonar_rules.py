"""Helper script to ingest Sonar rule JSON into the RAG store."""
from pathlib import Path

from code_pulse.logger import setup_logging
from code_pulse.rag.service import RAGService

logger = setup_logging(__name__)


def main() -> None:
    rag = RAGService()
    default_path = rag.data_dir.joinpath("ingest", "sonar_rule_data.json")
    logger.info("Default Sonar rules path: %s", default_path)

    rules_path = Path(
        (input(f"Path to Sonar rules JSON [{default_path}]: ").strip()) or default_path
    )
    namespace = input("Namespace to store under [sonar]: ").strip() or "sonar"

    logger.info("Starting Sonar rules ingestion path=%s namespace=%s", rules_path, namespace)
    chunks = rag.ingest_sonar_rules(rules_path, namespace=namespace)
    logger.info("Finished ingestion: %s chunks stored under namespace '%s'", chunks, namespace)
    print(f"Ingested {chunks} chunks into namespace '{namespace}'")


if __name__ == "__main__":
    main()
