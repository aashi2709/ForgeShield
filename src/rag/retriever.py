from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VECTOR_DB_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "chroma_db"
)

COLLECTION_NAME = (
    "forgeshield_knowledge"
)

EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)


class ForgeShieldRetriever:
    """
    Evidence retrieval interface for ForgeShield.

    The retriever returns source documents and metadata
    from the persistent ChromaDB knowledge base.

    It does not generate answers. Generation is deliberately
    kept separate so that retrieved evidence can be inspected
    before an LLM produces an explanation.
    """

    def __init__(
        self,
        db_path=None,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    ):

        if db_path is None:

            db_path = VECTOR_DB_DIR

        self.db_path = Path(
            db_path
        )

        self.collection_name = (
            collection_name
        )

        self.embedding_model_name = (
            embedding_model
        )

        print(
            "Loading embedding model..."
        )

        self.embedding_model = (
            SentenceTransformer(
                self.embedding_model_name
            )
        )

        print(
            "Connecting to ChromaDB..."
        )

        self.client = (
            chromadb.PersistentClient(
                path=str(
                    self.db_path
                )
            )
        )

        try:

            self.collection = (
                self.client.get_collection(
                    name=self.collection_name
                )
            )

        except Exception as exc:

            raise RuntimeError(
                f"Could not load ChromaDB "
                f"collection '{self.collection_name}'. "
                f"Run build_knowledge_base.py first."
            ) from exc

        print(
            f"Collection loaded: "
            f"{self.collection_name}"
        )

        print(
            f"Documents/chunks indexed: "
            f"{self.collection.count()}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        event_type: str | None = None,
        severity: str | None = None,
        synthetic_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically relevant evidence.

        Optional metadata filters allow downstream modules
        to restrict retrieval to particular event types
        or severity levels.
        """

        query = query.strip()

        if not query:

            raise ValueError(
                "Query cannot be empty."
            )

        if top_k < 1:

            raise ValueError(
                "top_k must be at least 1."
            )

        # Prevent requesting more records than
        # are actually present.

        top_k = min(
            top_k,
            self.collection.count(),
        )

        query_embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )
        )

        where_conditions = []

        if synthetic_only:

            where_conditions.append(
                {
                    "is_synthetic": True
                }
            )

        if event_type is not None:

            where_conditions.append(
                {
                    "event_type":
                        event_type
                }
            )

        if severity is not None:

            where_conditions.append(
                {
                    "severity":
                        severity
                }
            )

        where = None

        if len(where_conditions) == 1:

            where = where_conditions[0]

        elif len(where_conditions) > 1:

            where = {
                "$and":
                    where_conditions
            }

        query_kwargs = {
            "query_embeddings":
                query_embedding.tolist(),

            "n_results":
                top_k,
        }

        if where is not None:

            query_kwargs[
                "where"
            ] = where

        results = (
            self.collection.query(
                **query_kwargs
            )
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        ids = results.get(
            "ids",
            [[]],
        )[0]

        retrieved = []

        for index, document in enumerate(
            documents
        ):

            metadata = {}

            if index < len(
                metadatas
            ):

                metadata = (
                    metadatas[index]
                    or {}
                )

            distance = None

            if index < len(
                distances
            ):

                distance = float(
                    distances[index]
                )

            chunk_id = None

            if index < len(ids):

                chunk_id = ids[index]

            retrieved.append(
                {
                    "rank":
                        index + 1,

                    "chunk_id":
                        chunk_id,

                    "text":
                        document,

                    "distance":
                        distance,

                    "metadata":
                        metadata,

                    "source":
                        metadata.get(
                            "source"
                        ),

                    "incident_id":
                        metadata.get(
                            "incident_id"
                        ),

                    "machine_id":
                        metadata.get(
                            "machine_id"
                        ),

                    "event_type":
                        metadata.get(
                            "event_type"
                        ),

                    "severity":
                        metadata.get(
                            "severity"
                        ),

                    "is_synthetic":
                        metadata.get(
                            "is_synthetic",
                            True,
                        ),
                }
            )

        return retrieved

    def format_evidence(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        """
        Format retrieved records into an evidence block
        suitable for an LLM prompt.

        Source metadata is explicitly retained so the
        generation layer can cite where the evidence came from.
        """

        if not results:

            return (
                "No relevant evidence was retrieved "
                "from the ForgeShield knowledge base."
            )

        sections = []

        for result in results:

            metadata = result[
                "metadata"
            ]

            source = metadata.get(
                "source",
                "Unknown source",
            )

            incident_id = metadata.get(
                "incident_id",
                "Unknown incident",
            )

            event_type = metadata.get(
                "event_type",
                "Unknown event",
            )

            severity = metadata.get(
                "severity",
                "Unknown severity",
            )

            synthetic = metadata.get(
                "is_synthetic",
                True,
            )

            section = (
                f"[Evidence {result['rank']}]\n"
                f"Source: {source}\n"
                f"Incident ID: {incident_id}\n"
                f"Event Type: {event_type}\n"
                f"Severity: {severity}\n"
                f"Synthetic Record: {synthetic}\n"
                f"Retrieval Distance: "
                f"{result['distance']:.4f}\n"
                f"Content:\n"
                f"{result['text']}"
            )

            sections.append(
                section
            )

        return "\n\n".join(
            sections
        )


def print_results(
    query,
    results,
):

    print(
        "\n" + "-" * 70
    )

    print(
        "RETRIEVAL RESULTS"
    )

    print(
        "-" * 70
    )

    print(
        f"\nQuery:\n{query}"
    )

    if not results:

        print(
            "\nNo evidence found."
        )

        return

    for result in results:

        print(
            f"\n[{result['rank']}] "
            f"{result['source']}"
        )

        print(
            f"    Incident: "
            f"{result['incident_id']}"
        )

        print(
            f"    Event: "
            f"{result['event_type']}"
        )

        print(
            f"    Severity: "
            f"{result['severity']}"
        )

        print(
            f"    Distance: "
            f"{result['distance']:.4f}"
        )

        preview = (
            result["text"]
            .replace("\n", " ")
        )

        if len(preview) > 350:

            preview = (
                preview[:350]
                + "..."
            )

        print(
            f"    Preview: {preview}"
        )


def main():

    print("\n" + "=" * 70)

    print(
        "FORGESHIELD | RETRIEVAL ENGINE"
    )

    print("=" * 70)

    retriever = (
        ForgeShieldRetriever()
    )

    # ---------------------------------------------------------
    # Test 1: semantic retrieval
    # ---------------------------------------------------------

    query_1 = (
        "What should an operator do "
        "when a machine shows overheating?"
    )

    results_1 = retriever.retrieve(
        query=query_1,
        top_k=5,
    )

    print_results(
        query_1,
        results_1,
    )

    # ---------------------------------------------------------
    # Test 2: vibration-related retrieval
    # ---------------------------------------------------------

    query_2 = (
        "How should abnormal machine "
        "vibration be handled?"
    )

    results_2 = retriever.retrieve(
        query=query_2,
        top_k=3,
    )

    print_results(
        query_2,
        results_2,
    )

    # ---------------------------------------------------------
    # Test 3: metadata-filtered retrieval
    # ---------------------------------------------------------

    query_3 = (
        "What corrective actions are "
        "recommended?"
    )

    results_3 = retriever.retrieve(
        query=query_3,
        top_k=3,
        event_type="gas_leakage",
    )

    print_results(
        query_3,
        results_3,
    )

    # ---------------------------------------------------------
    # Evidence formatting
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "FORMATTED EVIDENCE SAMPLE"
    )

    print(
        "-" * 70
    )

    evidence = (
        retriever.format_evidence(
            results_1[:2]
        )
    )

    print(
        evidence
    )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL ENGINE TEST COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
