from pathlib import Path
import hashlib
import json

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DOCUMENT_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "docs"
)

VECTOR_DB_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "chroma_db"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "knowledge_base_metadata.json"
)

EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)

COLLECTION_NAME = (
    "forgeshield_knowledge"
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def load_documents():

    documents = []

    for path in DOCUMENT_DIR.rglob("*.md"):

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append(
            {
                "path": path,
                "text": text,
            }
        )

    return documents


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Split documents into overlapping character chunks.

    Overlap preserves context when an important statement
    occurs near a chunk boundary.
    """

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        if end >= len(text):
            break

        start = (
            end - overlap
        )

    return chunks


def create_chunk_id(
    source,
    chunk_index,
):

    raw = (
        f"{source}:"
        f"{chunk_index}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:20]


def extract_metadata(
    path,
    text,
):

    metadata = {
        "source": str(
            path.relative_to(
                PROJECT_ROOT
            )
        ),
        "document_type": (
            "synthetic_incident"
        ),
        "is_synthetic": True,
    }

    # Extract useful fields from the
    # generated incident markdown.

    for line in text.splitlines():

        line = line.strip()

        if line.startswith(
            "**Incident ID:**"
        ):

            metadata[
                "incident_id"
            ] = line.split(
                "**Incident ID:**",
                1,
            )[1].strip()

        elif line.startswith(
            "**Machine ID:**"
        ):

            metadata[
                "machine_id"
            ] = line.split(
                "**Machine ID:**",
                1,
            )[1].strip()

        elif line.startswith(
            "**Event Type:**"
        ):

            metadata[
                "event_type"
            ] = line.split(
                "**Event Type:**",
                1,
            )[1].strip()

        elif line.startswith(
            "**Severity:**"
        ):

            metadata[
                "severity"
            ] = line.split(
                "**Severity:**",
                1,
            )[1].strip()

    return metadata


def build_chunks(
    documents,
):

    chunks = []

    for document in documents:

        path = document[
            "path"
        ]

        text = document[
            "text"
        ]

        document_chunks = chunk_text(
            text
        )

        base_metadata = (
            extract_metadata(
                path,
                text,
            )
        )

        for index, chunk in enumerate(
            document_chunks
        ):

            metadata = (
                base_metadata.copy()
            )

            metadata[
                "chunk_index"
            ] = index

            metadata[
                "chunk_count"
            ] = len(
                document_chunks
            )

            chunk_id = (
                create_chunk_id(
                    str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    index,
                )
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": metadata,
                }
            )

    return chunks


def main():

    print("\n" + "=" * 70)

    print(
        "FORGESHIELD | RAG KNOWLEDGE BASE"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # Load documents
    # ---------------------------------------------------------

    print(
        "\nSearching knowledge-base documents..."
    )

    documents = load_documents()

    print(
        f"  Documents found: "
        f"{len(documents)}"
    )

    if not documents:

        raise FileNotFoundError(
            "No Markdown documents found in "
            f"{DOCUMENT_DIR}"
        )

    # ---------------------------------------------------------
    # Chunk documents
    # ---------------------------------------------------------

    print(
        "\nChunking documents..."
    )

    chunks = build_chunks(
        documents
    )

    print(
        f"  Total chunks: "
        f"{len(chunks)}"
    )

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    print(
        f"  Model: "
        f"{EMBEDDING_MODEL}"
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
    )

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------

    print(
        "\nGenerating embeddings..."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = (
        embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    )

    print(
        f"\n  Embedding matrix: "
        f"{embeddings.shape}"
    )

    # ---------------------------------------------------------
    # Create persistent ChromaDB
    # ---------------------------------------------------------

    print(
        "\nInitializing ChromaDB..."
    )

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = (
        chromadb.PersistentClient(
            path=str(
                VECTOR_DB_DIR
            )
        )
    )

    # Delete the existing collection
    # so rebuilding the knowledge base
    # remains deterministic.

    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "  Existing collection removed."
        )

    except Exception:

        print(
            "  No existing collection to remove."
        )

    collection = (
        client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description":
                    (
                        "ForgeShield evidence "
                        "retrieval knowledge base"
                    ),
                "embedding_model":
                    EMBEDDING_MODEL,
            },
        )
    )

    # ---------------------------------------------------------
    # Insert chunks
    # ---------------------------------------------------------

    print(
        "\nIndexing chunks in ChromaDB..."
    )

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(
        f"  Indexed: "
        f"{collection.count()} chunks"
    )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata = {
        "project":
            "ForgeShield",

        "collection":
            COLLECTION_NAME,

        "embedding_model":
            EMBEDDING_MODEL,

        "chunk_size":
            CHUNK_SIZE,

        "chunk_overlap":
            CHUNK_OVERLAP,

        "document_count":
            len(documents),

        "chunk_count":
            len(chunks),

        "document_types": [
            "synthetic_incident"
        ],

        "synthetic_data":
            True,

        "source_directory":
            str(
                DOCUMENT_DIR.relative_to(
                    PROJECT_ROOT
                )
            ),

        "vector_database":
            "ChromaDB",
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Retrieval sanity check
    # ---------------------------------------------------------

    print(
        "\nRunning retrieval sanity check..."
    )

    query = (
        "What should be done when "
        "a machine is overheating?"
    )

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )
    )

    results = collection.query(
        query_embeddings=(
            query_embedding.tolist()
        ),
        n_results=3,
    )

    print(
        f"\n  Query:"
    )

    print(
        f"  {query}"
    )

    print(
        "\n  Retrieved documents:"
    )

    retrieved_documents = (
        results.get(
            "documents",
            [[]],
        )[0]
    )

    retrieved_metadatas = (
        results.get(
            "metadatas",
            [[]],
        )[0]
    )

    retrieved_distances = (
        results.get(
            "distances",
            [[]],
        )[0]
    )

    for index, document in enumerate(
        retrieved_documents
    ):

        metadata = (
            retrieved_metadatas[
                index
            ]
        )

        distance = (
            retrieved_distances[
                index
            ]
        )

        print(
            f"\n  Result {index + 1}"
        )

        print(
            f"    Source: "
            f"{metadata.get('source')}"
        )

        print(
            f"    Event: "
            f"{metadata.get('event_type')}"
        )

        print(
            f"    Severity: "
            f"{metadata.get('severity')}"
        )

        print(
            f"    Distance: "
            f"{distance:.4f}"
        )

        preview = (
            document
            .replace("\n", " ")
            [:250]
        )

        print(
            f"    Preview: "
            f"{preview}..."
        )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG KNOWLEDGE BASE COMPLETE"
    )

    print("=" * 70)

    print(
        "\nPipeline:"
    )

    print(
        "  Documents"
    )

    print(
        "      ↓"
    )

    print(
        "  Chunking"
    )

    print(
        "      ↓"
    )

    print(
        "  Sentence-Transformer Embeddings"
    )

    print(
        "      ↓"
    )

    print(
        "  ChromaDB"
    )

    print(
        "      ↓"
    )

    print(
        "  Semantic Retrieval"
    )

    print(
        "\nKnowledge base:"
    )

    print(
        f"  {VECTOR_DB_DIR}"
    )

    print(
        "\nMetadata:"
    )

    print(
        f"  {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()
