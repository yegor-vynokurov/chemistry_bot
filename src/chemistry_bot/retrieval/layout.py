"""Shared repository paths for the current Introductory Chemistry workflow."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
DATA_ROOT = REPO_ROOT / "data"
CONFIG_ROOT = REPO_ROOT / "config"
PROMPT_GARDEN_ROOT = REPO_ROOT / "prompt_garden"

INTROCHEM_CORPUS_ID = "introductory_chemistry"
INTROCHEM_RAW_ROOT = DATA_ROOT / "raw" / INTROCHEM_CORPUS_ID
INTROCHEM_PRIMARY_HTML = (
    INTROCHEM_RAW_ROOT
    / "Introductory-Chemistry-1st-Canadian-Edition-1695676481.html"
)
INTROCHEM_PRIMARY_XML = (
    INTROCHEM_RAW_ROOT
    / "Introductory-Chemistry-1st-Canadian-Edition-1695676494.xml"
)
INTROCHEM_NORMALIZED_ROOT = DATA_ROOT / "normalized" / INTROCHEM_CORPUS_ID
INTROCHEM_RAG_ROOT = DATA_ROOT / "rag" / INTROCHEM_CORPUS_ID
INTROCHEM_CHROMA_DIR = DATA_ROOT / "indexes" / "introductory_chemistry_chroma"
INTROCHEM_COLLECTION_NAME = "introchem_theory_v1"
INTROCHEM_EMBEDDING_MODEL = "embeddinggemma"
INTROCHEM_RETRIEVAL_FIXTURES = (
    CONFIG_ROOT / "introchem_retrieval_queries.jsonl"
)

INTROCHEM_PARSE_SCRIPT = REPO_ROOT / "src" / "parse_introchem_xhtml_v3.py"
INTROCHEM_BUILD_CHUNKS_SCRIPT = (
    REPO_ROOT / "src" / "build_introchem_rag_chunks_v3.py"
)
INTROCHEM_VECTOR_SEARCH_SCRIPT = (
    REPO_ROOT / "src" / "introchem_vector_search.py"
)

OPENSCIED_RAW_ROOT = DATA_ROOT / "raw" / "OpenSciEd"


def introchem_chapter_output_dir(chapter_number: int) -> Path:
    """Return the normalized output directory for a chapter."""

    return INTROCHEM_NORMALIZED_ROOT / f"chapter_{chapter_number:02d}_v3"


def introchem_layout_summary() -> dict[str, str]:
    """Return the current IntroChem path map in a log-friendly form."""

    return {
        "repo_root": str(REPO_ROOT),
        "prompt_garden_root": str(PROMPT_GARDEN_ROOT),
        "config_root": str(CONFIG_ROOT),
        "data_root": str(DATA_ROOT),
        "introchem_primary_html": str(INTROCHEM_PRIMARY_HTML),
        "introchem_primary_xml": str(INTROCHEM_PRIMARY_XML),
        "introchem_normalized_root": str(INTROCHEM_NORMALIZED_ROOT),
        "introchem_rag_root": str(INTROCHEM_RAG_ROOT),
        "introchem_chroma_dir": str(INTROCHEM_CHROMA_DIR),
        "introchem_collection_name": INTROCHEM_COLLECTION_NAME,
        "introchem_embedding_model": INTROCHEM_EMBEDDING_MODEL,
        "introchem_retrieval_fixtures": str(INTROCHEM_RETRIEVAL_FIXTURES),
        "openscied_raw_root": str(OPENSCIED_RAW_ROOT),
    }


__all__ = [
    "CONFIG_ROOT",
    "DATA_ROOT",
    "INTROCHEM_BUILD_CHUNKS_SCRIPT",
    "INTROCHEM_CHROMA_DIR",
    "INTROCHEM_COLLECTION_NAME",
    "INTROCHEM_CORPUS_ID",
    "INTROCHEM_EMBEDDING_MODEL",
    "INTROCHEM_NORMALIZED_ROOT",
    "INTROCHEM_PARSE_SCRIPT",
    "INTROCHEM_PRIMARY_HTML",
    "INTROCHEM_PRIMARY_XML",
    "INTROCHEM_RAG_ROOT",
    "INTROCHEM_RETRIEVAL_FIXTURES",
    "INTROCHEM_VECTOR_SEARCH_SCRIPT",
    "OPENSCIED_RAW_ROOT",
    "PROMPT_GARDEN_ROOT",
    "REPO_ROOT",
    "SRC_ROOT",
    "introchem_chapter_output_dir",
    "introchem_layout_summary",
]
