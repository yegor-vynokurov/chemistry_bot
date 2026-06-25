"""Build RAG chunks from normalized Introductory Chemistry chapter outputs.

This is the current low-level chunking stage behind the public corpus build
wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "introchem.rag_chunks.v1"
CHUNKER_VERSION = "introchem_structure_chunker_v3"

SPECIAL_BLOCK_TYPES = {
    "worked_example",
    "exercise_set",
    "learning_objectives",
    "key_takeaways",
    "feature",
}

DEFAULT_EXCLUDED_GROUPS = {
    "assessment_answer",
    "self_test_answer",
    "attribution",
}

# Normalized XHTML occasionally contains malformed HTML list-start values such as
# `7""`. Keep a transparent audit trail instead of crashing or silently losing
# the intended numbering.
LIST_START_REPAIRS: list[dict[str, Any]] = []


@dataclass(frozen=True)
class ChunkConfig:
    target_chars: int = 1800
    max_chars: int = 3000
    overlap_chars: int = 180
    min_chars: int = 80

    def validate(self) -> None:
        if self.target_chars <= 0:
            raise ValueError("target_chars must be > 0")
        if self.max_chars < self.target_chars:
            raise ValueError("max_chars must be >= target_chars")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("overlap_chars must be >= 0 and < max_chars")
        if self.min_chars < 0:
            raise ValueError("min_chars must be >= 0")


@dataclass
class SemanticUnit:
    locator: str
    text: str
    chunk_kind: str
    retrieval_group: str
    block_start: int
    block_end: int
    block_type: str
    title: str | None = None
    part: str | None = None
    item_index: int | None = None
    paired_text: str | None = None
    instruction_text: str | None = None
    default_retrieval: bool = True
    node_path: str | None = None
    formula_count: int = 0
    media_count: int = 0


def normalize_space(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "part"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {error}") from error


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def find_sections_files(root: Path) -> list[Path]:
    candidates = sorted(root.rglob("sections.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No sections.jsonl files found under {root}. "
            "Point --normalized-root to the directory containing parsed chapter folders."
        )
    return candidates


def parse_list_start(value: Any, node_path: str | None = None) -> int:
    """Return a safe integer for an HTML ``ol[start]`` value.

    Pressbooks exports can contain malformed attribute values such as ``7""``.
    A strict ``int(value)`` call aborts the entire corpus build in that case.
    This function accepts valid integers directly, repairs quote-like debris,
    and records every repair in ``LIST_START_REPAIRS`` for later inspection.
    """
    if value in (None, ""):
        return 1

    if isinstance(value, int) and not isinstance(value, bool):
        return value

    raw = str(value).strip()

    if re.fullmatch(r"[+-]?\d+", raw):
        return int(raw)

    # First try the conservative repair: remove only surrounding quote marks.
    quote_cleaned = raw.strip().strip("\"'").strip()
    if re.fullmatch(r"[+-]?\d+", quote_cleaned):
        parsed = int(quote_cleaned)
        repair_method = "trim_surrounding_quotes"
    else:
        # Last-resort recovery for malformed HTML attributes. HTML ordered-list
        # start values are integers, so the first integer token is the only
        # useful semantic payload here.
        match = re.search(r"[+-]?\d+", raw)
        if match:
            parsed = int(match.group(0))
            repair_method = "extract_first_integer"
        else:
            parsed = 1
            repair_method = "fallback_to_one"

    repair = {
        "node_path": node_path,
        "raw_value": raw,
        "normalized_value": parsed,
        "method": repair_method,
    }
    if repair not in LIST_START_REPAIRS:
        LIST_START_REPAIRS.append(repair)

    return parsed


def actual_list_items(node: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    first_number = parse_list_start(
        node.get("start"),
        node_path=str(node.get("node_path") or "") or None,
    )
    result: list[tuple[int, dict[str, Any]]] = []
    for offset, item in enumerate(node.get("items") or []):
        result.append((first_number + offset, item))
    return result


def render_list_item(item: dict[str, Any]) -> str:
    lead = normalize_space(str(item.get("lead_text") or ""))
    full = normalize_space(str(item.get("text") or ""))
    children = item.get("children") or []

    parts: list[str] = []
    if lead:
        parts.append(lead)

    # The full item text can duplicate lead + child paragraphs. Prefer explicit children.
    for child in children:
        text = normalize_space(str(child.get("text") or ""))
        if text and text not in parts:
            parts.append(text)

    if not children and full and full != lead:
        parts.append(full)

    return "\n\n".join(parts).strip()


def split_children_by_heading(structure: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] = {"heading": None, "children": []}

    for child in structure.get("children") or []:
        if child.get("node_type") == "heading":
            if current["heading"] is not None or current["children"]:
                groups.append(current)
            current = {
                "heading": normalize_space(str(child.get("text") or "")) or None,
                "children": [],
            }
        else:
            current["children"].append(child)

    if current["heading"] is not None or current["children"]:
        groups.append(current)

    return groups


def render_nodes(
    nodes: Sequence[dict[str, Any]],
    *,
    include_media: bool = True,
) -> str:
    parts: list[str] = []
    for node in nodes:
        if not include_media and node.get("node_type") in {"figure", "media"}:
            continue
        node_type = node.get("node_type")
        if node_type == "list":
            ordered = bool(node.get("ordered"))
            for number, item in actual_list_items(node):
                item_text = render_list_item(item)
                if item_text:
                    prefix = f"{number}." if ordered else "-"
                    indented = item_text.replace("\n", "\n   ")
                    parts.append(f"{prefix} {indented}")
        else:
            text = normalize_space(str(node.get("text") or ""))
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def make_unit(
    *,
    block: dict[str, Any],
    locator: str,
    text: str,
    chunk_kind: str,
    retrieval_group: str,
    title: str | None = None,
    part: str | None = None,
    item_index: int | None = None,
    paired_text: str | None = None,
    instruction_text: str | None = None,
    default_retrieval: bool = True,
    node_path: str | None = None,
) -> SemanticUnit | None:
    text = normalize_space(text)
    if not text:
        return None
    return SemanticUnit(
        locator=locator,
        text=text,
        chunk_kind=chunk_kind,
        retrieval_group=retrieval_group,
        block_start=int(block.get("block_index") or 0),
        block_end=int(block.get("block_index") or 0),
        block_type=str(block.get("block_type") or "other"),
        title=title,
        part=part,
        item_index=item_index,
        paired_text=normalize_space(paired_text or "") or None,
        instruction_text=normalize_space(instruction_text or "") or None,
        default_retrieval=default_retrieval,
        node_path=node_path,
        formula_count=len(block.get("formulae") or []),
        media_count=len(block.get("media") or []),
    )


def units_from_worked_example(block: dict[str, Any]) -> list[SemanticUnit]:
    structure = block.get("structure") or {}
    title = normalize_space(str(structure.get("title") or "Worked example"))
    groups = split_children_by_heading(structure)
    result: list[SemanticUnit] = []
    problem_map: dict[int, str] = {}

    for group in groups:
        heading = normalize_space(str(group.get("heading") or "Content"))
        heading_key = heading.lower()
        nodes = group.get("children") or []
        lists = [node for node in nodes if node.get("node_type") == "list"]
        non_lists = [node for node in nodes if node.get("node_type") != "list"]
        intro_text = render_nodes(non_lists)

        if heading_key in {"problem", "problems"}:
            if intro_text:
                unit = make_unit(
                    block=block,
                    locator=f"example-{slug(title)}-problems-intro",
                    text=intro_text,
                    chunk_kind="worked_example_problem_intro",
                    retrieval_group="worked_example",
                    title=title,
                    part="problems",
                    node_path=structure.get("node_path"),
                )
                if unit:
                    result.append(unit)
            for list_node in lists:
                for number, item in actual_list_items(list_node):
                    item_text = render_list_item(item)
                    problem_map[number] = item_text
                    unit = make_unit(
                        block=block,
                        locator=f"example-{slug(title)}-problem-{number}",
                        text=item_text,
                        chunk_kind="worked_example_problem",
                        retrieval_group="worked_example",
                        title=title,
                        part="problem",
                        item_index=number,
                        instruction_text=intro_text,
                        node_path=list_node.get("node_path"),
                    )
                    if unit:
                        result.append(unit)
            continue

        if heading_key in {"solution", "solutions"}:
            if intro_text:
                unit = make_unit(
                    block=block,
                    locator=f"example-{slug(title)}-solutions-intro",
                    text=intro_text,
                    chunk_kind="worked_example_solution_intro",
                    retrieval_group="worked_example",
                    title=title,
                    part="solutions",
                    node_path=structure.get("node_path"),
                )
                if unit:
                    result.append(unit)
            for list_node in lists:
                for number, item in actual_list_items(list_node):
                    item_text = render_list_item(item)
                    unit = make_unit(
                        block=block,
                        locator=f"example-{slug(title)}-solution-{number}",
                        text=item_text,
                        chunk_kind="worked_example_solution",
                        retrieval_group="worked_example",
                        title=title,
                        part="solution",
                        item_index=number,
                        paired_text=problem_map.get(number),
                        node_path=list_node.get("node_path"),
                    )
                    if unit:
                        result.append(unit)
            continue

        if heading_key in {"test yourself", "test yourselfs"}:
            text = render_nodes(nodes)
            unit = make_unit(
                block=block,
                locator=f"example-{slug(title)}-self-test",
                text=text,
                chunk_kind="self_test_question",
                retrieval_group="self_test",
                title=title,
                part="test_yourself",
                default_retrieval=False,
                node_path=structure.get("node_path"),
            )
            if unit:
                result.append(unit)
            continue

        if heading_key in {"answer", "answers"}:
            text = render_nodes(nodes, include_media=False)
            unit = make_unit(
                block=block,
                locator=f"example-{slug(title)}-self-test-answer",
                text=text,
                chunk_kind="self_test_answer",
                retrieval_group="self_test_answer",
                title=title,
                part="answer",
                default_retrieval=False,
                node_path=structure.get("node_path"),
            )
            if unit:
                result.append(unit)
            continue

        text = render_nodes(nodes)
        unit = make_unit(
            block=block,
            locator=f"example-{slug(title)}-{slug(heading)}",
            text=text,
            chunk_kind="worked_example_part",
            retrieval_group="worked_example",
            title=title,
            part=heading_key or "content",
            node_path=structure.get("node_path"),
        )
        if unit:
            result.append(unit)

    if not result:
        fallback = make_unit(
            block=block,
            locator=f"example-{slug(title)}",
            text=str(block.get("text") or ""),
            chunk_kind="worked_example",
            retrieval_group="worked_example",
            title=title,
            default_retrieval=False,
            node_path=structure.get("node_path"),
        )
        if fallback:
            result.append(fallback)

    return result


def units_from_exercise_set(block: dict[str, Any]) -> list[SemanticUnit]:
    structure = block.get("structure") or {}
    title = normalize_space(str(structure.get("title") or "Exercises"))
    groups = split_children_by_heading(structure)
    result: list[SemanticUnit] = []
    question_map: dict[int, str] = {}

    for group in groups:
        heading = normalize_space(str(group.get("heading") or "Content"))
        heading_key = heading.lower()
        nodes = group.get("children") or []
        lists = [node for node in nodes if node.get("node_type") == "list"]
        non_lists = [node for node in nodes if node.get("node_type") != "list"]
        intro_text = render_nodes(non_lists)

        if heading_key in {"question", "questions", "exercise", "exercises"}:
            if intro_text:
                unit = make_unit(
                    block=block,
                    locator=f"exercise-{block.get('block_index')}-questions-intro",
                    text=intro_text,
                    chunk_kind="assessment_question_intro",
                    retrieval_group="assessment_question",
                    title=title,
                    part="questions",
                    default_retrieval=False,
                    node_path=structure.get("node_path"),
                )
                if unit:
                    result.append(unit)
            for list_node in lists:
                for number, item in actual_list_items(list_node):
                    item_text = render_list_item(item)
                    question_map[number] = item_text
                    unit = make_unit(
                        block=block,
                        locator=f"exercise-{block.get('block_index')}-question-{number}",
                        text=item_text,
                        chunk_kind="assessment_question",
                        retrieval_group="assessment_question",
                        title=title,
                        part="question",
                        item_index=number,
                        instruction_text=intro_text,
                        default_retrieval=False,
                        node_path=list_node.get("node_path"),
                    )
                    if unit:
                        result.append(unit)
            continue

        if heading_key in {"answer", "answers", "solution", "solutions"}:
            if intro_text:
                unit = make_unit(
                    block=block,
                    locator=f"exercise-{block.get('block_index')}-answers-intro",
                    text=intro_text,
                    chunk_kind="assessment_answer_intro",
                    retrieval_group="assessment_answer",
                    title=title,
                    part="answers",
                    default_retrieval=False,
                    node_path=structure.get("node_path"),
                )
                if unit:
                    result.append(unit)
            for list_node in lists:
                for number, item in actual_list_items(list_node):
                    item_text = render_list_item(item)
                    unit = make_unit(
                        block=block,
                        locator=f"exercise-{block.get('block_index')}-answer-{number}",
                        text=item_text,
                        chunk_kind="assessment_answer",
                        retrieval_group="assessment_answer",
                        title=title,
                        part="answer",
                        item_index=number,
                        paired_text=question_map.get(number),
                        default_retrieval=False,
                        node_path=list_node.get("node_path"),
                    )
                    if unit:
                        result.append(unit)
            continue

        text = render_nodes(nodes)
        unit = make_unit(
            block=block,
            locator=f"exercise-{block.get('block_index')}-{slug(heading)}",
            text=text,
            chunk_kind="assessment_part",
            retrieval_group="assessment_question",
            title=title,
            part=heading_key or "content",
            default_retrieval=False,
            node_path=structure.get("node_path"),
        )
        if unit:
            result.append(unit)

    if not result:
        fallback = make_unit(
            block=block,
            locator=f"exercise-{block.get('block_index')}",
            text=str(block.get("text") or ""),
            chunk_kind="assessment",
            retrieval_group="assessment_question",
            title=title,
            node_path=structure.get("node_path"),
        )
        if fallback:
            result.append(fallback)

    return result


def units_from_simple_special(block: dict[str, Any]) -> list[SemanticUnit]:
    block_type = str(block.get("block_type") or "other")
    structure = block.get("structure") or {}
    title = normalize_space(str(structure.get("title") or "")) or None
    text = str(block.get("text") or "")

    group_map = {
        "learning_objectives": "learning_objective",
        "key_takeaways": "summary",
        "feature": "feature",
    }
    retrieval_group = group_map.get(block_type, "theory")
    unit = make_unit(
        block=block,
        locator=f"block-{block.get('block_index')}-{block_type}",
        text=text,
        chunk_kind=block_type,
        retrieval_group=retrieval_group,
        title=title,
        node_path=structure.get("node_path"),
    )
    return [unit] if unit else []


def split_long_text(text: str, config: ChunkConfig) -> list[str]:
    text = normalize_space(text)
    if len(text) <= config.max_chars:
        return [text]

    # Structure-aware fallback: paragraphs, then lines, then sentences, then words.
    separators: list[str | re.Pattern[str]] = [
        "\n\n",
        "\n",
        re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])"),
        " ",
    ]

    def split_with_separator(value: str, separator: str | re.Pattern[str]) -> list[str]:
        if isinstance(separator, str):
            return [part.strip() for part in value.split(separator) if part.strip()]
        return [part.strip() for part in separator.split(value) if part.strip()]

    pieces = [text]
    for separator in separators:
        next_pieces: list[str] = []
        for piece in pieces:
            if len(piece) <= config.max_chars:
                next_pieces.append(piece)
            else:
                split_parts = split_with_separator(piece, separator)
                next_pieces.extend(split_parts if len(split_parts) > 1 else [piece])
        pieces = next_pieces

    # Hard fallback for a single unbroken token/string.
    hard_pieces: list[str] = []
    for piece in pieces:
        if len(piece) <= config.max_chars:
            hard_pieces.append(piece)
        else:
            step = max(1, config.max_chars - config.overlap_chars)
            for start in range(0, len(piece), step):
                part = piece[start : start + config.max_chars].strip()
                if part:
                    hard_pieces.append(part)
                if start + config.max_chars >= len(piece):
                    break

    chunks: list[str] = []
    buffer = ""
    for piece in hard_pieces:
        separator = "\n\n" if buffer else ""
        candidate = f"{buffer}{separator}{piece}".strip()
        if buffer and len(candidate) > config.target_chars:
            chunks.append(buffer)
            overlap = buffer[-config.overlap_chars :].strip() if config.overlap_chars else ""
            buffer = f"{overlap}\n\n{piece}".strip() if overlap else piece
            if len(buffer) > config.max_chars:
                chunks.append(buffer[: config.max_chars].strip())
                buffer = buffer[config.max_chars - config.overlap_chars :].strip()
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)

    return [chunk for chunk in chunks if chunk]


def ordinary_units(section: dict[str, Any]) -> list[SemanticUnit]:
    result: list[SemanticUnit] = []
    attribution_mode = False

    for block in section.get("blocks") or []:
        block_type = str(block.get("block_type") or "other")
        if block_type in SPECIAL_BLOCK_TYPES:
            attribution_mode = False
            if block_type == "worked_example":
                result.extend(units_from_worked_example(block))
            elif block_type == "exercise_set":
                result.extend(units_from_exercise_set(block))
            else:
                result.extend(units_from_simple_special(block))
            continue

        text = normalize_space(str(block.get("text") or ""))
        if not text:
            continue

        if block_type == "heading" and text.lower() == "media attributions":
            attribution_mode = True

        if attribution_mode:
            retrieval_group = "attribution"
            chunk_kind = "attribution"
            default_retrieval = False
        elif block_type == "formula_or_equation":
            retrieval_group = "theory"
            chunk_kind = "equation_context"
            default_retrieval = True
        elif block_type == "table":
            retrieval_group = "reference_table"
            chunk_kind = "table"
            default_retrieval = True
        elif block_type == "figure":
            retrieval_group = "figure"
            chunk_kind = "figure_caption"
            default_retrieval = True
        else:
            retrieval_group = "theory"
            chunk_kind = "theory"
            default_retrieval = True

        structure = block.get("structure") or {}
        unit = make_unit(
            block=block,
            locator=f"block-{block.get('block_index')}-{block_type}",
            text=text,
            chunk_kind=chunk_kind,
            retrieval_group=retrieval_group,
            default_retrieval=default_retrieval,
            node_path=structure.get("node_path"),
        )
        if unit:
            result.append(unit)

    return result


def merge_ordinary_units(units: list[SemanticUnit], config: ChunkConfig) -> list[SemanticUnit]:
    """Merge adjacent generic theory units, but never merge special semantic units."""
    mergeable_kinds = {"theory", "equation_context", "figure_caption"}
    result: list[SemanticUnit] = []
    buffer: list[SemanticUnit] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n\n".join(unit.text for unit in buffer).strip()
        first = buffer[0]
        last = buffer[-1]
        result.append(
            SemanticUnit(
                locator=f"blocks-{first.block_start}-{last.block_end}",
                text=text,
                chunk_kind="theory",
                retrieval_group=first.retrieval_group,
                block_start=first.block_start,
                block_end=last.block_end,
                block_type="mixed" if len({u.block_type for u in buffer}) > 1 else first.block_type,
                default_retrieval=all(u.default_retrieval for u in buffer),
                formula_count=sum(u.formula_count for u in buffer),
                media_count=sum(u.media_count for u in buffer),
            )
        )
        buffer = []

    for unit in units:
        if unit.chunk_kind not in mergeable_kinds or unit.retrieval_group != "theory":
            flush()
            result.append(unit)
            continue

        if not buffer:
            buffer.append(unit)
            continue

        candidate_length = sum(len(item.text) for item in buffer) + 2 * len(buffer) + len(unit.text)
        if candidate_length <= config.target_chars:
            buffer.append(unit)
        else:
            flush()
            buffer.append(unit)

    flush()
    return result


def embedding_prefix(section: dict[str, Any], unit: SemanticUnit) -> str:
    lines = [
        f"Book: {section.get('attribution', {}).get('title', 'Introductory Chemistry')}",
        f"Chapter {section.get('chapter_number')}: {section.get('chapter_title')}",
        f"Section {section.get('derived_section_number')}: {section.get('section_title')}",
        f"Content type: {unit.chunk_kind.replace('_', ' ')}",
    ]
    if unit.title:
        lines.append(f"Item: {unit.title}")
    if unit.part:
        lines.append(f"Part: {unit.part.replace('_', ' ')}")
    if unit.item_index is not None:
        lines.append(f"Number: {unit.item_index}")
    if unit.instruction_text:
        lines.append(f"Instruction: {unit.instruction_text}")
    if unit.paired_text:
        label = "Related problem" if "solution" in unit.chunk_kind else "Related question"
        lines.append(f"{label}: {unit.paired_text}")
    return "\n".join(lines)


def build_chunk_record(
    section: dict[str, Any],
    unit: SemanticUnit,
    text: str,
    piece_index: int,
    piece_count: int,
) -> dict[str, Any]:
    chunk_id = f"{section['id']}__{unit.locator}"
    if piece_count > 1:
        chunk_id += f"__part-{piece_index:02d}"

    prefix = embedding_prefix(section, unit)
    embedding_text = f"{prefix}\n\n{text}".strip()
    source_url = f"https://opentextbc.ca/introductorychemistry/#{section.get('source_html_id')}"

    metadata = {
        "source_id": section.get("source_id"),
        "source_file": section.get("source_file"),
        "source_format": section.get("source_format"),
        "source_html_id": section.get("source_html_id"),
        "source_url": source_url,
        "language": section.get("language"),
        "document_type": section.get("document_type"),
        "chapter_number": section.get("chapter_number"),
        "chapter_title": section.get("chapter_title"),
        "section_number": section.get("derived_section_number"),
        "section_title": section.get("section_title"),
        "parent_section_id": section.get("id"),
        "block_start": unit.block_start,
        "block_end": unit.block_end,
        "block_type": unit.block_type,
        "chunk_kind": unit.chunk_kind,
        "retrieval_group": unit.retrieval_group,
        "item_title": unit.title or "",
        "part": unit.part or "",
        "item_index": unit.item_index if unit.item_index is not None else -1,
        "has_instruction": bool(unit.instruction_text),
        "has_paired_text": bool(unit.paired_text),
        "piece_index": piece_index,
        "piece_count": piece_count,
        "default_retrieval": unit.default_retrieval,
        "formula_count": unit.formula_count,
        "media_count": unit.media_count,
        "license": section.get("license"),
        "parser_version": section.get("parser_version"),
        "chunker_version": CHUNKER_VERSION,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "id": chunk_id,
        "parent_section_id": section.get("id"),
        "parent_block_locator": unit.locator,
        "text": text,
        "embedding_text": embedding_text,
        "metadata": metadata,
        "content_sha256": sha256_text(text),
        "embedding_sha256": sha256_text(embedding_text),
    }


def chunks_from_section(section: dict[str, Any], config: ChunkConfig) -> list[dict[str, Any]]:
    units = merge_ordinary_units(ordinary_units(section), config)
    chunks: list[dict[str, Any]] = []

    for unit in units:
        pieces = split_long_text(unit.text, config)
        for index, piece in enumerate(pieces, start=1):
            chunks.append(build_chunk_record(section, unit, piece, index, len(pieces)))

    return chunks


def validate_chunks(chunks: Sequence[dict[str, Any]], config: ChunkConfig) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    ids = [str(chunk.get("id")) for chunk in chunks]
    counts = Counter(ids)
    duplicate_ids = sorted(chunk_id for chunk_id, count in counts.items() if count > 1)
    if duplicate_ids:
        errors.append(f"Duplicate chunk IDs: {duplicate_ids[:10]}")

    for chunk in chunks:
        chunk_id = chunk.get("id")
        text = str(chunk.get("text") or "")
        metadata = chunk.get("metadata") or {}
        if not text.strip():
            errors.append(f"Empty text: {chunk_id}")
        if len(text) > config.max_chars + config.overlap_chars:
            errors.append(
                f"Chunk exceeds expected maximum ({len(text)} chars): {chunk_id}"
            )
        for field in (
            "source_id",
            "chapter_number",
            "section_title",
            "parent_section_id",
            "chunk_kind",
            "retrieval_group",
        ):
            if metadata.get(field) in (None, ""):
                errors.append(f"Missing metadata.{field}: {chunk_id}")
        embedding_text = str(chunk.get("embedding_text") or "")
        if len(embedding_text) < config.min_chars:
            warnings.append(
                f"Very short embedding text ({len(embedding_text)} chars): {chunk_id}"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "ok": not errors,
        "chunk_count": len(chunks),
        "duplicate_id_count": len(duplicate_ids),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def build_report(
    sections_files: Sequence[Path],
    sections: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    config: ChunkConfig,
) -> dict[str, Any]:
    lengths = [len(str(chunk.get("text") or "")) for chunk in chunks]
    kinds = Counter(chunk["metadata"]["chunk_kind"] for chunk in chunks)
    groups = Counter(chunk["metadata"]["retrieval_group"] for chunk in chunks)
    chapters = Counter(int(chunk["metadata"]["chapter_number"]) for chunk in chunks)

    return {
        "schema_version": SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "input_files": [str(path) for path in sections_files],
        "input_section_count": len(sections),
        "chapter_count": len(chapters),
        "chunk_count": len(chunks),
        "list_start_repair_count": len(LIST_START_REPAIRS),
        "list_start_repairs": LIST_START_REPAIRS,
        "default_retrieval_chunk_count": sum(
            1 for chunk in chunks if chunk["metadata"]["default_retrieval"]
        ),
        "excluded_from_default_count": sum(
            1 for chunk in chunks if not chunk["metadata"]["default_retrieval"]
        ),
        "config": {
            "target_chars": config.target_chars,
            "max_chars": config.max_chars,
            "overlap_chars": config.overlap_chars,
            "min_chars": config.min_chars,
        },
        "lengths": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        },
        "chunks_by_kind": dict(sorted(kinds.items())),
        "chunks_by_retrieval_group": dict(sorted(groups.items())),
        "chunks_by_chapter": {str(key): value for key, value in sorted(chapters.items())},
    }


def write_review(path: Path, chunks: Sequence[dict[str, Any]], max_per_group: int = 4) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk["metadata"]["chunk_kind"])].append(chunk)

    lines = [
        "# RAG chunks review",
        "",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Chunker: `{CHUNKER_VERSION}`",
        f"- Total chunks: **{len(chunks)}**",
        "",
        "This file shows a small sample for each chunk kind. The complete data is in `rag_chunks.jsonl`.",
        "",
    ]

    for kind in sorted(grouped):
        items = grouped[kind]
        lines.extend([f"## `{kind}`", "", f"Count: **{len(items)}**", ""])
        for chunk in items[:max_per_group]:
            meta = chunk["metadata"]
            lines.extend(
                [
                    f"### `{chunk['id']}`",
                    "",
                    f"- Chapter: {meta['chapter_number']} | Section: {meta['section_number']} {meta['section_title']}",
                    f"- Retrieval group: `{meta['retrieval_group']}`",
                    f"- Default retrieval: `{meta['default_retrieval']}`",
                    f"- Characters: {len(chunk['text'])}",
                    "",
                    "```text",
                    chunk["embedding_text"][:2400],
                    "```",
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build structure-aware RAG chunks from normalized Introductory Chemistry "
            "sections.jsonl files."
        )
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        required=True,
        help="Directory containing all normalized chapter folders (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for rag_chunks.jsonl and review reports.",
    )
    parser.add_argument("--target-chars", type=int, default=1800)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--overlap-chars", type=int, default=180)
    parser.add_argument("--min-chars", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    LIST_START_REPAIRS.clear()
    args = parse_args()
    config = ChunkConfig(
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        min_chars=args.min_chars,
    )
    config.validate()

    normalized_root = args.normalized_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sections_files = find_sections_files(normalized_root)
    sections: list[dict[str, Any]] = []
    for sections_file in sections_files:
        sections.extend(read_jsonl(sections_file))

    # Stable ordering across reruns.
    sections.sort(
        key=lambda record: (
            int(record.get("chapter_number") or 0),
            int(record.get("section_index") or 0),
            str(record.get("id") or ""),
        )
    )

    chunks: list[dict[str, Any]] = []
    for section in sections:
        chunks.extend(chunks_from_section(section, config))

    chunks.sort(
        key=lambda record: (
            int(record["metadata"].get("chapter_number") or 0),
            str(record["metadata"].get("section_number") or ""),
            int(record["metadata"].get("block_start") or 0),
            str(record.get("id") or ""),
        )
    )

    validation = validate_chunks(chunks, config)
    report = build_report(sections_files, sections, chunks, config)

    validation["list_start_repair_count"] = len(LIST_START_REPAIRS)
    validation["list_start_repairs"] = LIST_START_REPAIRS
    if LIST_START_REPAIRS:
        validation["warnings"].append(
            f"Repaired {len(LIST_START_REPAIRS)} malformed ordered-list start value(s). "
            "See list_start_repairs for details."
        )
        validation["warning_count"] = len(validation["warnings"])

    write_jsonl(output / "rag_chunks.jsonl", chunks)
    (output / "chunk_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "chunk_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_review(output / "review_chunks.md", chunks)

    print(f"Chunker version: {CHUNKER_VERSION}")
    print(f"Input sections files: {len(sections_files)}")
    print(f"Input sections: {len(sections)}")
    print(f"RAG chunks: {len(chunks)}")
    print(f"Default retrieval chunks: {report['default_retrieval_chunk_count']}")
    print(f"Repaired list starts: {report['list_start_repair_count']}")
    print(f"Output: {output}")
    print(f"Validation: {'OK' if validation['ok'] else 'FAILED'}")

    if not validation["ok"]:
        preview = "\n".join(validation["errors"][:20])
        raise SystemExit(f"Chunk validation failed:\n{preview}")


if __name__ == "__main__":
    main()
