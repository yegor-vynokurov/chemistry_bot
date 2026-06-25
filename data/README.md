# Data Directory

This directory contains local corpus inputs and generated retrieval artifacts.

Most of `data/` remains ignored in Git.
The repository strategy is:

- track reproducibility and documentation
- avoid committing bulky generated artifacts by default

## Main Structure

- `raw/`
  Local source materials.

- `raw/introductory_chemistry/`
  Current primary source textbook files for the verified IntroChem RAG workflow.

- `raw/OpenSciEd/`
  Additional local curriculum materials kept as separate corpus inputs and not part of the main verified RAG path.

- `normalized/introductory_chemistry/`
  Parsed chapter outputs from the current XHTML parser.

- `rag/introductory_chemistry/`
  Chunk records, chunk reports, and retrieval-review outputs.

- `indexes/introductory_chemistry_chroma/`
  Local Chroma persistence files and manifest for the current textbook index.

## Rebuild Workflow

The main public rebuild command is:

```powershell
.\.venv\Scripts\python.exe scripts\build_introchem_rag_bundle.py
```

Use `--dry-run` first if you want to inspect the command chain without rebuilding.

## Tracking Policy

Tracked here:

- this `README.md`

Ignored here by default:

- raw corpus files
- normalized chapter outputs
- generated RAG chunks
- Chroma index files

The current project treats reproducibility as more important than committing all derived artifacts.
