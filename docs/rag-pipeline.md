# RAG Pipeline

This document describes the current low-level workflow for turning the textbook corpus into a local retrieval index.

The repository now also has a public one-command wrapper for this flow:

```powershell
.\.venv\Scripts\python.exe scripts\build_introchem_rag_bundle.py --dry-run
```

The manual steps below remain useful because they show the current implementation details behind that wrapper.

## Corpus

Current target corpus:

- `Introductory Chemistry, 1st Canadian Edition`

The current parser and chunking flow are tuned for this book.

## Environment

Windows PowerShell example:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

RAG-related dependencies:

```powershell
pip install -r requirements-rag.txt
```

## Preferred Public Wrapper

For ordinary rebuilds, use the wrapper first:

```powershell
.\.venv\Scripts\python.exe scripts\build_introchem_rag_bundle.py
```

Useful variants:

```powershell
.\.venv\Scripts\python.exe scripts\build_introchem_rag_bundle.py --dry-run
.\.venv\Scripts\python.exe scripts\build_introchem_rag_bundle.py --chapters 1-3 --rebuild-index
.\.venv\Scripts\python.exe scripts\build_introchem_rag_bundle.py --skip-parse --skip-index
```

## Step 1. Parse The Textbook

Convert the source XHTML book into normalized chapter folders.

Example:

```powershell
python src/parse_introchem_xhtml_v3.py `
  --input "data/raw/introductory_chemistry/Introductory-Chemistry-1st-Canadian-Edition-1695676481.html" `
  --chapter 18 `
  --output "data/normalized/introductory_chemistry/chapter_18_v3"
```

## Step 2. Build RAG Chunks

Create chunk records from the normalized chapter outputs.

```powershell
python src/build_introchem_rag_chunks_v3.py `
  --normalized-root "data/normalized/introductory_chemistry" `
  --output "data/rag/introductory_chemistry"
```

## Step 3. Build The Vector Index

Build or rebuild the Chroma index with Ollama embeddings.

```powershell
python src/introchem_vector_search.py build `
  --chunks "data/rag/introductory_chemistry/rag_chunks.jsonl" `
  --db "data/indexes/introductory_chemistry_chroma" `
  --collection "introchem_theory_v1" `
  --model "embeddinggemma" `
  --rebuild
```

## Step 4. Query The Index

Single question:

```powershell
python src/introchem_vector_search.py search `
  --query "What is a chemical equation and why must it be balanced?" `
  -k 5
```

Interactive mode:

```powershell
python src/introchem_vector_search.py interactive
```

Example queries:

- `What is oxidation?`
- `How is an ionic bond formed?`
- `What is a reducing agent?`

## Step 5. Retrieval Review

Run fixture-based retrieval checks:

```powershell
python src/introchem_vector_search.py batch `
  --tests "config/introchem_retrieval_queries.jsonl" `
  --output "data/rag/introductory_chemistry/retrieval_review.md" `
  -k 5
```

## Useful Variants

Limit to one chapter:

```powershell
python src/introchem_vector_search.py search `
  --query "How do oxidation numbers change?" `
  --chapter 14
```

Theory-only retrieval:

```powershell
python src/introchem_vector_search.py search `
  --query "What is a reducing agent?" `
  --retrieval-group theory
```

Include non-default retrieval groups:

```powershell
python src/introchem_vector_search.py search `
  --query "Answer to the self-test about balancing ammonia formation" `
  --include-nondefault
```

Index metadata:

```powershell
python src/introchem_vector_search.py info
```

## Notes

- The current workflow is intentionally explicit so the internal parse, chunk, and index steps remain understandable.
- The repository now exposes the same flow through `scripts/build_introchem_rag_bundle.py`.
- The smoke-test plan will later reuse lightweight tracked fixtures such as `config/introchem_retrieval_queries.jsonl`.
