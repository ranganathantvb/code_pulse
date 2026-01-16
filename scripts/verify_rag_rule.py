"""Verify RAG rule-key lookup returns chunks with metadata."""
import argparse

from code_pulse.config import get_settings
from code_pulse.rag.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-key", required=True)
    parser.add_argument("--namespace", default="sonar")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    if args.no_embeddings:
        from langchain_community.vectorstores import FAISS
        from langchain_core.embeddings import Embeddings

        class DummyEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.0] for _ in texts]

            def embed_query(self, text):
                return [0.0]

        settings = get_settings()
        vector_dir = settings.data_dir.joinpath("vectorstores")
        safe = args.namespace.replace("/", "_")
        path = vector_dir.joinpath(safe)
        if not path.exists():
            print(f"Vector store not found: {path}")
            return
        store = FAISS.load_local(path, DummyEmbeddings(), allow_dangerous_deserialization=True)
        docstore = getattr(store, "docstore", None)
        raw_dict = getattr(docstore, "_dict", None)
        docs = []
        if isinstance(raw_dict, dict):
            for doc in raw_dict.values():
                if doc.metadata.get("doc_type") == "sonar_rule" and doc.metadata.get("rule_key") == args.rule_key:
                    docs.append(doc)
    else:
        rag = RAGService()
        docs = rag.lookup_by_metadata(
            namespace=args.namespace,
            metadata_filter={"doc_type": "sonar_rule", "rule_key": args.rule_key},
            query_text=args.rule_key,
        )
    print(f"matches={len(docs)} rule_key={args.rule_key} namespace={args.namespace}")
    for doc in docs[: args.limit]:
        meta = {"doc_type": doc.metadata.get("doc_type"), "rule_key": doc.metadata.get("rule_key"), "section": doc.metadata.get("section")}
        print(f"meta={meta}")


if __name__ == "__main__":
    main()
