"""Build, inspect, and query the local Introductory Chemistry Chroma index.

This module is the low-level retrieval/index CLI behind the current public
wrapper scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


INDEX_SCHEMA_VERSION = "introchem.chroma_index.v1"
INDEXER_VERSION = "introchem_chroma_indexer_v1"

DEFAULT_MODEL = "embeddinggemma"
DEFAULT_COLLECTION = "introchem_theory_v1"
DEFAULT_BASE_URL = "http://localhost:11434"


def load_integrations():
    try:
        import chromadb
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_ollama import OllamaEmbeddings
    except ImportError as error:
        missing = getattr(error, "name", None) or str(error)
        raise SystemExit(
            "Не установлены зависимости для векторного индекса.\n\n"
            "Установите их командой:\n"
            "  pip install -U langchain-ollama langchain-chroma chromadb\n\n"
            f"Не найден модуль: {missing}"
        ) from error

    return chromadb, Chroma, Document, OllamaEmbeddings


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Некорректный JSONL: {path}, строка {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Ожидался JSON-объект: {path}, строка {line_number}"
                )
            yield record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_collection_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    if not value:
        raise ValueError("Имя коллекции стало пустым после нормализации.")
    return value


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep only Chroma-safe primitive metadata values."""
    result: dict[str, str | int | float | bool] = {}

    for key, value in metadata.items():
        key = str(key)

        if value is None:
            continue

        if isinstance(value, bool):
            result[key] = value
        elif isinstance(value, int):
            result[key] = value
        elif isinstance(value, float):
            result[key] = value
        elif isinstance(value, str):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            # Arrays are supported by newer Chroma versions, but JSON strings
            # are more portable between versions and sufficient for provenance.
            result[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, dict):
            result[key] = json.dumps(value, ensure_ascii=False)
        else:
            result[key] = str(value)

    return result


def validate_chunk(record: dict[str, Any], line_number: int) -> None:
    required = ("id", "text", "embedding_text", "metadata")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(
            f"Чанк #{line_number} не содержит обязательные поля: {missing}"
        )

    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError(f"Чанк #{line_number}: поле id должно быть непустой строкой.")

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"Чанк #{line_number}: поле text должно быть непустой строкой.")

    if (
        not isinstance(record["embedding_text"], str)
        or not record["embedding_text"].strip()
    ):
        raise ValueError(
            f"Чанк #{line_number}: поле embedding_text должно быть непустой строкой."
        )

    if not isinstance(record["metadata"], dict):
        raise ValueError(f"Чанк #{line_number}: metadata должно быть объектом.")


def load_chunks(
    path: Path,
    only_default: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}

    for line_number, record in enumerate(read_jsonl(path), start=1):
        validate_chunk(record, line_number)

        chunk_id = record["id"]
        if chunk_id in by_id:
            raise ValueError(f"Повторяющийся chunk id: {chunk_id}")

        default_retrieval = bool(
            record.get("metadata", {}).get("default_retrieval", True)
        )
        by_id[chunk_id] = record

        if only_default and not default_retrieval:
            continue

        records.append(record)

    if not records:
        raise ValueError("После фильтрации не осталось чанков для индексации.")

    return records, by_id


def batches(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше нуля.")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ollama_json(base_url: str, endpoint: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + endpoint
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as error:
        raise ConnectionError(
            f"Не удалось подключиться к Ollama по адресу {base_url}.\n"
            "Убедитесь, что Ollama запущена."
        ) from error

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Неожиданный ответ Ollama от {url}")
    return data


def model_is_available(base_url: str, model: str) -> tuple[bool, list[str]]:
    data = ollama_json(base_url, "/api/tags")
    names = [
        str(item.get("name") or item.get("model") or "")
        for item in data.get("models", [])
        if isinstance(item, dict)
    ]

    requested_base = model.split(":", 1)[0]
    available = any(
        name == model
        or name.split(":", 1)[0] == requested_base
        for name in names
    )
    return available, sorted(name for name in names if name)


def create_embeddings(model: str, base_url: str):
    _, _, _, OllamaEmbeddings = load_integrations()
    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )


def collection_names(client: Any) -> set[str]:
    result: set[str] = set()
    for item in client.list_collections():
        if isinstance(item, str):
            result.add(item)
        else:
            name = getattr(item, "name", None)
            if name:
                result.add(str(name))
    return result


def manifest_path(db_path: Path, collection_name: str) -> Path:
    return db_path / f"{collection_name}.manifest.json"


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_manifest(db_path: Path, collection_name: str) -> dict[str, Any]:
    path = manifest_path(db_path, collection_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден manifest индекса: {path}\n"
            "Сначала выполните команду build."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Некорректный manifest: {path}")
    return data


def make_documents(records: Sequence[dict[str, Any]], Document: Any):
    documents = []
    ids = []

    for record in records:
        chunk_id = record["id"]
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "chunk_id": chunk_id,
                "parent_section_id": str(
                    record.get("parent_section_id")
                    or metadata.get("parent_section_id")
                    or ""
                ),
                "parent_block_locator": str(
                    record.get("parent_block_locator") or ""
                ),
                "content_sha256": str(record.get("content_sha256") or ""),
                "embedding_sha256": str(record.get("embedding_sha256") or ""),
                "schema_version": str(record.get("schema_version") or ""),
            }
        )

        documents.append(
            Document(
                id=chunk_id,
                page_content=record["embedding_text"],
                metadata=sanitize_metadata(metadata),
            )
        )
        ids.append(chunk_id)

    return documents, ids


def command_build(args: argparse.Namespace) -> None:
    chromadb, Chroma, Document, _ = load_integrations()

    chunks_path = Path(args.chunks).resolve()
    db_path = Path(args.db).resolve()
    collection_name = sanitize_collection_name(args.collection)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Не найден файл чанков: {chunks_path}")

    available, model_names = model_is_available(args.base_url, args.model)
    if not available:
        visible = "\n".join(f"  - {name}" for name in model_names[:20])
        raise SystemExit(
            f"Embedding-модель {args.model!r} не установлена в Ollama.\n\n"
            f"Установите её:\n  ollama pull {args.model}\n\n"
            f"Доступные модели:\n{visible or '  (список пуст)'}"
        )

    records, all_by_id = load_chunks(
        chunks_path,
        only_default=args.only_default,
    )

    ids_seen = {record["id"] for record in records}
    default_count = sum(
        bool(record.get("metadata", {}).get("default_retrieval", True))
        for record in records
    )

    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    existing = collection_names(client)

    if collection_name in existing:
        if not args.rebuild:
            raise SystemExit(
                f"Коллекция {collection_name!r} уже существует в {db_path}.\n"
                "Используйте --rebuild для её полного пересоздания."
            )
        client.delete_collection(collection_name)

    embeddings = create_embeddings(args.model, args.base_url)
    probe = embeddings.embed_query(
        "Chemistry studies matter, its properties, and its transformations."
    )
    embedding_dimension = len(probe)

    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={
            "description": "Introductory Chemistry normalized RAG chunks",
            "embedding_model": args.model,
            "indexer_version": INDEXER_VERSION,
        },
    )

    documents, document_ids = make_documents(records, Document)

    total_batches = (len(documents) + args.batch_size - 1) // args.batch_size
    indexed = 0

    for batch_number, start in enumerate(
        range(0, len(documents), args.batch_size),
        start=1,
    ):
        docs_batch = documents[start : start + args.batch_size]
        ids_batch = document_ids[start : start + args.batch_size]

        vector_store.add_documents(
            documents=docs_batch,
            ids=ids_batch,
        )
        indexed += len(docs_batch)
        print(
            f"[{batch_number}/{total_batches}] "
            f"Добавлено {indexed}/{len(documents)} чанков"
        )

    collection_count = client.get_collection(collection_name).count()
    if collection_count != len(documents):
        raise RuntimeError(
            "Число записей в Chroma не совпадает с ожидаемым: "
            f"{collection_count} != {len(documents)}"
        )

    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "indexer_version": INDEXER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "embedding_model": args.model,
        "embedding_dimension": embedding_dimension,
        "ollama_base_url": args.base_url,
        "chunks_path": str(chunks_path),
        "chunks_sha256": sha256_file(chunks_path),
        "source_chunk_count": len(all_by_id),
        "indexed_chunk_count": len(records),
        "indexed_default_chunk_count": default_count,
        "only_default_indexed": bool(args.only_default),
        "batch_size": args.batch_size,
        "chroma_count": collection_count,
    }
    save_json(manifest_path(db_path, collection_name), manifest)

    print()
    print("Векторный индекс создан.")
    print(f"Коллекция:          {collection_name}")
    print(f"Embedding-модель:   {args.model}")
    print(f"Размер вектора:     {embedding_dimension}")
    print(f"Чанков в индексе:   {collection_count}")
    print(f"Папка Chroma:       {db_path}")
    print(f"Manifest:           {manifest_path(db_path, collection_name)}")


def build_where_filter(args: argparse.Namespace) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []

    if not args.include_nondefault:
        clauses.append({"default_retrieval": True})

    if args.chapter is not None:
        clauses.append({"chapter_number": args.chapter})

    if args.retrieval_group:
        clauses.append({"retrieval_group": args.retrieval_group})

    if args.chunk_kind:
        clauses.append({"chunk_kind": args.chunk_kind})

    if args.language:
        clauses.append({"language": args.language})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def source_records_from_manifest(
    manifest: dict[str, Any],
    explicit_chunks: str | None,
) -> dict[str, dict[str, Any]]:
    path_value = explicit_chunks or manifest.get("chunks_path")
    if not path_value:
        return {}

    path = Path(path_value)
    if not path.exists():
        print(
            f"Предупреждение: исходный rag_chunks.jsonl не найден: {path}\n"
            "Будет показан embedding_text из Chroma.",
            file=sys.stderr,
        )
        return {}

    _, by_id = load_chunks(path, only_default=False)
    return by_id


def open_vector_store(
    db_path: Path,
    collection_name: str,
    model: str,
    base_url: str,
):
    chromadb, Chroma, _, _ = load_integrations()
    client = chromadb.PersistentClient(path=str(db_path))

    if collection_name not in collection_names(client):
        raise FileNotFoundError(
            f"Коллекция {collection_name!r} не найдена в {db_path}"
        )

    embeddings = create_embeddings(model, base_url)
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return client, vector_store


def result_to_dict(
    rank: int,
    document: Any,
    score: float,
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    chunk_id = str(document.metadata.get("chunk_id") or document.id or "")
    source_record = source_by_id.get(chunk_id)
    source_text = (
        str(source_record.get("text") or "")
        if source_record
        else document.page_content
    )

    return {
        "rank": rank,
        "score": float(score),
        "chunk_id": chunk_id,
        "text": source_text,
        "embedding_text": document.page_content,
        "metadata": dict(document.metadata),
    }


def run_search(
    args: argparse.Namespace,
    query: str,
) -> list[dict[str, Any]]:
    db_path = Path(args.db).resolve()
    collection_name = sanitize_collection_name(args.collection)
    manifest = load_manifest(db_path, collection_name)

    manifest_model = str(manifest.get("embedding_model") or "")
    model = args.model or manifest_model
    if not model:
        raise ValueError("В manifest не указана embedding-модель.")

    base_url = args.base_url or str(
        manifest.get("ollama_base_url") or DEFAULT_BASE_URL
    )

    available, _ = model_is_available(base_url, model)
    if not available:
        raise SystemExit(
            f"Embedding-модель {model!r} не установлена.\n"
            f"Выполните: ollama pull {model}"
        )

    _, vector_store = open_vector_store(
        db_path,
        collection_name,
        model,
        base_url,
    )

    source_by_id = source_records_from_manifest(
        manifest,
        explicit_chunks=args.chunks,
    )
    where_filter = build_where_filter(args)

    kwargs: dict[str, Any] = {
        "query": query,
        "k": args.k,
    }
    if where_filter is not None:
        kwargs["filter"] = where_filter

    raw_results = vector_store.similarity_search_with_score(**kwargs)
    return [
        result_to_dict(
            rank=rank,
            document=document,
            score=score,
            source_by_id=source_by_id,
        )
        for rank, (document, score) in enumerate(raw_results, start=1)
    ]


def print_results(
    query: str,
    results: Sequence[dict[str, Any]],
    max_chars: int,
) -> None:
    print()
    print("=" * 88)
    print(f"QUERY: {query}")
    print("=" * 88)

    if not results:
        print("Результатов нет.")
        return

    for result in results:
        metadata = result["metadata"]
        text = result["text"].strip()
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n…"

        print()
        print(
            f"[{result['rank']}] score={result['score']:.6f}  "
            f"chapter={metadata.get('chapter_number')}  "
            f"section={metadata.get('section_number')}  "
            f"group={metadata.get('retrieval_group')}  "
            f"kind={metadata.get('chunk_kind')}"
        )
        print(f"chunk_id: {result['chunk_id']}")
        print(f"title:    {metadata.get('section_title', '')}")
        print("-" * 88)
        print(text)


def command_search(args: argparse.Namespace) -> None:
    results = run_search(args, args.query)
    print_results(args.query, results, args.max_chars)

    if args.save_results:
        output = Path(args.save_results)
        save_json(
            output,
            {
                "query": args.query,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
        )
        print(f"\nРезультаты сохранены: {output.resolve()}")


def command_interactive(args: argparse.Namespace) -> None:
    print("Интерактивный поиск по Introductory Chemistry.")
    print("Пустая строка или /exit завершает работу.")

    while True:
        try:
            query = input("\nЗапрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() in {"/exit", "exit", "выход"}:
            break

        results = run_search(args, query)
        print_results(query, results, args.max_chars)


def load_test_queries(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, record in enumerate(read_jsonl(path), start=1):
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Тестовый запрос #{line_number} не содержит непустое поле query."
            )
        result.append(record)
    return result


def command_batch(args: argparse.Namespace) -> None:
    tests_path = Path(args.tests).resolve()
    tests = load_test_queries(tests_path)
    output_path = Path(args.output).resolve()

    lines = [
        "# Retrieval review",
        "",
        f"- Created: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Collection: `{args.collection}`",
        f"- Tests: `{len(tests)}`",
        f"- Top-k: `{args.k}`",
        "",
    ]
    json_results: list[dict[str, Any]] = []

    for test_index, test in enumerate(tests, start=1):
        query = test["query"]
        results = run_search(args, query)
        json_results.append(
            {
                "id": test.get("id", f"query_{test_index}"),
                "query": query,
                "expected_topics": test.get("expected_topics", []),
                "results": results,
            }
        )

        lines.extend(
            [
                f"## {test_index}. {query}",
                "",
            ]
        )

        if test.get("expected_topics"):
            lines.append(
                "**Expected topics:** "
                + ", ".join(map(str, test["expected_topics"]))
            )
            lines.append("")

        for result in results:
            metadata = result["metadata"]
            text = result["text"].strip()
            if args.max_chars > 0 and len(text) > args.max_chars:
                text = text[: args.max_chars].rstrip() + "\n…"

            lines.extend(
                [
                    (
                        f"### Rank {result['rank']} · "
                        f"score `{result['score']:.6f}` · "
                        f"chapter `{metadata.get('chapter_number')}` · "
                        f"group `{metadata.get('retrieval_group')}`"
                    ),
                    "",
                    f"- Chunk: `{result['chunk_id']}`",
                    f"- Section: `{metadata.get('section_number')}` "
                    f"{metadata.get('section_title', '')}",
                    "",
                    text,
                    "",
                ]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    json_path = output_path.with_suffix(".json")
    save_json(
        json_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tests_path": str(tests_path),
            "collection": args.collection,
            "top_k": args.k,
            "queries": json_results,
        },
    )

    print(f"Markdown review: {output_path}")
    print(f"JSON results:    {json_path}")


def command_info(args: argparse.Namespace) -> None:
    chromadb, _, _, _ = load_integrations()

    db_path = Path(args.db).resolve()
    collection_name = sanitize_collection_name(args.collection)
    manifest = load_manifest(db_path, collection_name)

    client = chromadb.PersistentClient(path=str(db_path))
    count = client.get_collection(collection_name).count()

    print(json.dumps(
        {
            **manifest,
            "current_chroma_count": count,
            "manifest_path": str(manifest_path(db_path, collection_name)),
        },
        ensure_ascii=False,
        indent=2,
    ))


def add_shared_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default="data/indexes/introductory_chemistry_chroma",
        help="Папка постоянной базы Chroma.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Имя коллекции Chroma.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding-модель. По умолчанию берётся из manifest.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Адрес Ollama. По умолчанию берётся из manifest.",
    )
    parser.add_argument(
        "--chunks",
        default=None,
        help="Необязательный путь к rag_chunks.jsonl для показа исходного text.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Число результатов.",
    )
    parser.add_argument(
        "--include-nondefault",
        action="store_true",
        help="Также искать среди ответов, self-test и других отключённых чанков.",
    )
    parser.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Ограничить поиск номером главы.",
    )
    parser.add_argument(
        "--retrieval-group",
        default=None,
        help="Ограничить поиск retrieval_group.",
    )
    parser.add_argument(
        "--chunk-kind",
        default=None,
        help="Ограничить поиск chunk_kind.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Ограничить поиск языком metadata.language.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1400,
        help="Максимум символов одного результата в отчёте. 0 = без ограничения.",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Создание постоянного Chroma-индекса из Introductory Chemistry "
            "RAG chunks и проверка семантического поиска."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Построить постоянный векторный индекс.",
    )
    build_parser.add_argument(
        "--chunks",
        default="data/rag/introductory_chemistry/rag_chunks.jsonl",
        help="Путь к rag_chunks.jsonl.",
    )
    build_parser.add_argument(
        "--db",
        default="data/indexes/introductory_chemistry_chroma",
        help="Папка постоянной базы Chroma.",
    )
    build_parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Имя коллекции Chroma.",
    )
    build_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Embedding-модель Ollama.",
    )
    build_parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Адрес локальной Ollama.",
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер пакета при индексации.",
    )
    build_parser.add_argument(
        "--only-default",
        action="store_true",
        help="Индексировать только чанки default_retrieval=true.",
    )
    build_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Удалить существующую коллекцию и построить её заново.",
    )
    build_parser.set_defaults(func=command_build)

    search_parser = subparsers.add_parser(
        "search",
        help="Выполнить один семантический поиск.",
    )
    add_shared_search_arguments(search_parser)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument(
        "--save-results",
        default=None,
        help="Сохранить результаты в JSON.",
    )
    search_parser.set_defaults(func=command_search)

    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Запустить интерактивный поиск.",
    )
    add_shared_search_arguments(interactive_parser)
    interactive_parser.set_defaults(func=command_interactive)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Выполнить набор тестовых запросов и создать review.",
    )
    add_shared_search_arguments(batch_parser)
    batch_parser.add_argument(
        "--tests",
        default="config/introchem_retrieval_queries.jsonl",
        help="JSONL с тестовыми запросами.",
    )
    batch_parser.add_argument(
        "--output",
        default="data/rag/introductory_chemistry/retrieval_review.md",
        help="Выходной Markdown review.",
    )
    batch_parser.set_defaults(func=command_batch)

    info_parser = subparsers.add_parser(
        "info",
        help="Показать manifest и число записей в индексе.",
    )
    info_parser.add_argument(
        "--db",
        default="data/indexes/introductory_chemistry_chroma",
    )
    info_parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
    )
    info_parser.set_defaults(func=command_info)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, ConnectionError) as error:
        print(f"\nОшибка: {error}\n", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
