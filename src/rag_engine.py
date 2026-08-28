from pathlib import Path
from threading import Lock, Thread


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "marketron_marketing"

RAG_TOP_K = 3
RAG_MAX_CHARS = 8000


# ============================================================
# LAZY RAG RESOURCE
# ============================================================

_vector_store = None
_resource_lock = Lock()
_warmup_started = False


def get_vector_store():
    """
    Load the embedding model and Chroma vector store lazily.

    The resource is created only once per Python process.
    A lock prevents simultaneous initialization if the
    background warm-up and a user request happen together.
    """

    global _vector_store

    if _vector_store is not None:
        return _vector_store

    with _resource_lock:

        # Another thread may have finished while we waited.
        if _vector_store is not None:
            return _vector_store

        # Heavy imports happen ONLY when RAG is actually needed.
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    return _vector_store


# ============================================================
# BACKGROUND WARM-UP
# ============================================================

def _warmup_rag():
    """
    Background worker.

    It intentionally does not use Streamlit APIs.
    """

    try:
        get_vector_store()
    except Exception as exc:
        print(
            f"RAG warm-up failed: {exc}",
            flush=True
        )


def start_rag_warmup():
    """
    Start the RAG initialization once in the background.

    The function returns immediately, so the main Streamlit
    execution is not blocked.
    """

    global _warmup_started

    with _resource_lock:

        if _warmup_started:
            return

        _warmup_started = True

    thread = Thread(
        target=_warmup_rag,
        name="marketron-rag-warmup",
        daemon=True,
    )

    thread.start()


# ============================================================
# MARKETING KNOWLEDGE RETRIEVAL
# ============================================================

def retrieve_marketing_context(query: str) -> str:
    """
    Retrieve a small, relevant set of marketing knowledge
    for the supplied query.

    Returns:
        Compact text suitable for LLM context.
    """

    if not query or not query.strip():
        return ""

    vector_store = get_vector_store()

    documents = vector_store.max_marginal_relevance_search(
        query,
        k=RAG_TOP_K,
        fetch_k=10,
        lambda_mult=0.65
    )

    context_parts = []
    total_chars = 0

    for document in documents:

        text = document.page_content.strip()

        if not text:
            continue

        remaining_chars = RAG_MAX_CHARS - total_chars

        if remaining_chars <= 0:
            break

        chunk = text[:remaining_chars]

        context_parts.append(chunk)

        total_chars += len(chunk)

        print(
            f"RAG source: "
            f"{document.metadata.get('source', 'Unknown')}",
            flush=True
        )

    return "\n\n---\n\n".join(context_parts)
