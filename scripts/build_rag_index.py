import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES_PATH = PROJECT_ROOT / "knowledge_base" / "sources.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "marketron_marketing"

MAX_ARTICLES_PER_SOURCE = 8
REQUEST_TIMEOUT = 15

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/151.0 Safari/537.36"
    )
}


# ============================================================
# LOAD SOURCE REGISTRY
# ============================================================

def load_sources():
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(
            f"Could not find sources file:\n{SOURCES_PATH}"
        )

    with open(SOURCES_PATH, "r", encoding="utf-8") as file:
        sources = json.load(file)

    if not isinstance(sources, list):
        raise ValueError("sources.json must contain a list.")

    return sources


# ============================================================
# URL HELPERS
# ============================================================

def normalize_url(url: str) -> str:
    """
    Remove fragments and trailing slashes so duplicate URLs
    are less likely to enter the index.
    """
    parsed = urlparse(url)

    clean = parsed._replace(
        fragment="",
        query=""
    ).geturl()

    return clean.rstrip("/")


def is_same_domain(base_url: str, candidate_url: str) -> bool:
    base_domain = urlparse(base_url).netloc.lower()
    candidate_domain = urlparse(candidate_url).netloc.lower()

    return candidate_domain == base_domain


def looks_like_marketing_article(url: str) -> bool:
    """
    Lightweight filter to avoid obvious navigation,
    login, asset, media, and utility pages.
    """

    blocked_patterns = [
        "/login",
        "/signup",
        "/contact",
        "/about",
        "/privacy",
        "/terms",
        "/cookie",
        "/search",
        "/tag/",
        "/author/",
        "/category/",
        "/feed",
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".mp4",
    ]

    lowered = url.lower()

    return not any(
        pattern in lowered
        for pattern in blocked_patterns
    )


# ============================================================
# DISCOVER ARTICLES
# ============================================================

def discover_article_urls(source_url: str) -> list[str]:
    """
    The source registry now contains curated article URLs.
    No crawling is performed.
    """
    return [source_url]


# ============================================================
# EXTRACT PAGE CONTENT
# ============================================================

def extract_page_text(url: str) -> str:
    """
    Extract visible article text from a webpage.
    """

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Remove non-content elements.
    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "form"
        ]
    ):
        element.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# BUILD DOCUMENTS
# ============================================================

def build_documents(sources):
    documents = []
    seen_urls = set()

    for source in sources:

        source_name = source["name"]
        source_url = source["url"]
        source_category = source.get(
            "category",
            "marketing"
        )

        print(
            f"\nProcessing source: {source_name}"
        )

        try:
            article_urls = discover_article_urls(
                source_url
            )
        except Exception as exc:
            print(
                f"  Could not discover pages: {exc}"
            )
            continue

        print(
            f"  Found {len(article_urls)} candidate pages."
        )

        for article_url in article_urls:

            if article_url in seen_urls:
                continue

            try:
                text = extract_page_text(
                    article_url
                )

                if len(text) < 500:
                    print(
                        f"  Skipping short page: {article_url}"
                    )
                    continue

                document = Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "category": source_category,
                        "url": article_url
                    }
                )

                documents.append(document)
                seen_urls.add(article_url)

                print(
                    f"  Added: {article_url}"
                )

            except Exception as exc:
                print(
                    f"  Failed: {article_url}\n"
                    f"  Reason: {exc}"
                )

    return documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# ============================================================
# BUILD CHROMA INDEX
# ============================================================

def build_chroma_index(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # Rebuild from scratch so old chunks are not duplicated.
    try:
        vector_store.delete_collection()
    except Exception:
        pass

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    vector_store.add_documents(chunks)

    return vector_store


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MARKETRON RAG INDEX BUILDER")
    print("=" * 60)

    sources = load_sources()

    print(
        f"Loaded {len(sources)} source definitions."
    )

    documents = build_documents(
        sources
    )

    if not documents:
        raise RuntimeError(
            "No usable documents were collected."
        )

    print(
        f"\nCollected {len(documents)} source pages."
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    build_chroma_index(
        chunks
    )

    print("\n" + "=" * 60)
    print("RAG INDEX CREATED SUCCESSFULLY")
    print(f"Location: {CHROMA_DIR}")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()