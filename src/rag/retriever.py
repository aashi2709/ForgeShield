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

COLLECTION_NAME = "forgeshield_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ForgeShieldRetriever:
    """
    Evidence retrieval interface for ForgeShield.

    Retrieves evidence from two main document classes:

    1. Synthetic incident records
    2. Synthetic safety procedures

    Generation is deliberately kept separate from retrieval so
    that retrieved evidence can be inspected before an LLM
    generates an answer.
    """

    def __init__(
        self,
        db_path=None,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    ):

        if db_path is None:
            db_path = VECTOR_DB_DIR

        self.db_path = Path(db_path)

        self.collection_name = collection_name

        self.embedding_model_name = embedding_model

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        print("Connecting to ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
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
        document_type: str | None = None,
        evidence_type: str | None = None,
        synthetic_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve semantically relevant evidence.

        Optional metadata filters can restrict retrieval to:

        - event type
        - severity
        - document type
        - evidence type
        - synthetic/non-synthetic documents
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

        # -----------------------------------------------------
        # Build metadata filters
        # -----------------------------------------------------

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
                    "event_type": event_type
                }
            )

        if severity is not None:

            where_conditions.append(
                {
                    "severity": severity
                }
            )

        if document_type is not None:

            where_conditions.append(
                {
                    "document_type": document_type
                }
            )

        if evidence_type is not None:

            where_conditions.append(
                {
                    "evidence_type": evidence_type
                }
            )

        # -----------------------------------------------------
        # Construct Chroma where clause
        # -----------------------------------------------------

        where = None

        if len(where_conditions) == 1:

            where = where_conditions[0]

        elif len(where_conditions) > 1:

            where = {
                "$and": where_conditions
            }

        query_kwargs = {
            "query_embeddings":
                query_embedding.tolist(),

            "n_results":
                top_k,
        }

        if where is not None:

            query_kwargs["where"] = where

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

            if index < len(metadatas):

                metadata = (
                    metadatas[index]
                    or {}
                )

            distance = None

            if index < len(distances):

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

                    "document_type":
                        metadata.get(
                            "document_type"
                        ),

                    "evidence_type":
                        metadata.get(
                            "evidence_type"
                        ),

                    "incident_id":
                        metadata.get(
                            "incident_id"
                        ),

                    "procedure_id":
                        metadata.get(
                            "procedure_id"
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
        Format retrieved records into a structured evidence
        block suitable for an LLM prompt.

        Both incident and procedure evidence are explicitly
        identified so the generation layer can distinguish
        observed events from procedural guidance.
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

            document_type = metadata.get(
                "document_type",
                "Unknown document type",
            )

            evidence_type = metadata.get(
                "evidence_type",
                "Unknown evidence type",
            )

            incident_id = metadata.get(
                "incident_id",
                "Not applicable",
            )

            procedure_id = metadata.get(
                "procedure_id",
                "Not applicable",
            )

            event_type = metadata.get(
                "event_type",
                "Not specified",
            )

            severity = metadata.get(
                "severity",
                "Not specified",
            )

            synthetic = metadata.get(
                "is_synthetic",
                True,
            )

            distance = result.get(
                "distance"
            )

            distance_text = (
                f"{distance:.4f}"
                if distance is not None
                else "N/A"
            )

            section = (
                f"[Evidence {result['rank']}]\n"
                f"Source: {source}\n"
                f"Document Type: {document_type}\n"
                f"Evidence Type: {evidence_type}\n"
                f"Incident ID: {incident_id}\n"
                f"Procedure ID: {procedure_id}\n"
                f"Event Type: {event_type}\n"
                f"Severity: {severity}\n"
                f"Synthetic Record: {synthetic}\n"
                f"Retrieval Distance: {distance_text}\n"
                f"Content:\n"
                f"{result['text']}"
            )

            sections.append(section)

        return "\n\n".join(sections)


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
            f"    Document Type: "
            f"{result['document_type']}"
        )

        print(
            f"    Evidence Type: "
            f"{result['evidence_type']}"
        )

        print(
            f"    Incident: "
            f"{result['incident_id'] or '-'}"
        )

        print(
            f"    Procedure: "
            f"{result['procedure_id'] or '-'}"
        )

        print(
            f"    Event: "
            f"{result['event_type'] or '-'}"
        )

        print(
            f"    Severity: "
            f"{result['severity'] or '-'}"
        )

        print(
            f"    Synthetic: "
            f"{result['is_synthetic']}"
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

    print(
        "\n" + "=" * 70
    )

    print(
        "FORGESHIELD | RETRIEVAL ENGINE"
    )

    print("=" * 70)

    retriever = ForgeShieldRetriever()

    # ---------------------------------------------------------
    # Test 1: General semantic retrieval
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
    # Test 2: Procedure-only retrieval
    # ---------------------------------------------------------

    query_2 = (
        "What procedure should be followed "
        "when a machine is overheating?"
    )

    results_2 = retriever.retrieve(
        query=query_2,
        top_k=3,
        document_type="safety_procedure",
    )

    print_results(
        query_2,
        results_2,
    )

    # ---------------------------------------------------------
    # Test 3: Incident-only retrieval
    # ---------------------------------------------------------

    query_3 = (
        "What incidents have involved "
        "machine overheating?"
    )

    results_3 = retriever.retrieve(
        query=query_3,
        top_k=3,
        document_type="synthetic_incident",
    )

    print_results(
        query_3,
        results_3,
    )

    # ---------------------------------------------------------
    # Test 4: Event-filtered retrieval
    # ---------------------------------------------------------

    query_4 = (
        "What corrective actions are "
        "recommended?"
    )

    results_4 = retriever.retrieve(
        query=query_4,
        top_k=3,
        event_type="gas_leakage",
    )

    print_results(
        query_4,
        results_4,
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
            results_1[:3]
        )
    )

    print(evidence)

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