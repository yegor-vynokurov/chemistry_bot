from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lxml import etree


XHTML_NS = "http://www.w3.org/1999/xhtml"
NS = {"x": XHTML_NS}
SCHEMA_VERSION = "introchem.normalized.v3"
PARSER_VERSION = "introchem_xhtml_v3"

SUBSCRIPT_MAP = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
SUPERSCRIPT_MAP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


BLOCK_LEVEL_TAGS = {
    "address", "article", "aside", "blockquote", "div", "dl", "fieldset",
    "figure", "figcaption", "footer", "form", "h1", "h2", "h3", "h4",
    "h5", "h6", "header", "hr", "main", "nav", "ol", "p", "pre",
    "section", "table", "ul",
}


@dataclass(frozen=True)
class BookMetadata:
    title: str
    language: str
    authors: list[str]
    publisher: str | None
    publication_year: str | None
    license_code: str | None
    source_id: str


def local_name(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    return etree.QName(element).localname


def normalized_space(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def meta_values(root: etree._Element, name: str) -> list[str]:
    values = root.xpath(
        f'.//x:head/x:meta[@name="{name}"]/@content',
        namespaces=NS,
    )
    return [normalized_space(value) for value in values if normalized_space(value)]


def read_book_metadata(root: etree._Element) -> BookMetadata:
    title = (meta_values(root, "pb-title") or ["Untitled"])[0]
    language = (meta_values(root, "pb-language") or ["en"])[0]
    authors = meta_values(root, "pb-authors")
    publisher = (meta_values(root, "pb-publisher") or [None])[0]
    publication_year = (meta_values(root, "pb-copyright-year") or [None])[0]
    license_code = (meta_values(root, "pb-book-license") or [None])[0]

    return BookMetadata(
        title=title,
        language=language,
        authors=authors,
        publisher=publisher,
        publication_year=publication_year,
        license_code=license_code,
        source_id="introchem_canadian_1e",
    )


def unicode_script(text: str, *, superscript: bool) -> str:
    mapping = SUPERSCRIPT_MAP if superscript else SUBSCRIPT_MAP
    converted = text.translate(mapping)
    # Keep original text when it contains unsupported characters.
    supported = set(mapping.keys())
    if all(ord(char) in supported or char.isspace() for char in text):
        return converted
    return f"^({text})" if superscript else f"_({text})"


def render_text(element: etree._Element) -> str:
    """Render XHTML into readable text while preserving subscripts and formulas."""

    pieces: list[str] = []

    def visit(node: etree._Element) -> None:
        if node.text:
            pieces.append(node.text)

        for child in node:
            if not isinstance(child.tag, str):
                if child.tail:
                    pieces.append(child.tail)
                continue
            tag = local_name(child)

            if tag == "sub":
                pieces.append(unicode_script("".join(child.itertext()), superscript=False))
            elif tag == "sup":
                pieces.append(unicode_script("".join(child.itertext()), superscript=True))
            elif tag == "img":
                alt = normalized_space(child.get("alt"))
                if alt:
                    if "quicklatex" in (child.get("class") or ""):
                        pieces.append(f" [FORMULA: {alt}] ")
                    else:
                        pieces.append(f" [IMAGE: {alt}] ")
            elif tag == "br":
                pieces.append("\n")
            else:
                visit(child)

            if child.tail:
                pieces.append(child.tail)

    visit(element)
    text = "".join(pieces)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def serialize_inner_html(element: etree._Element) -> str:
    clone = deepcopy(element)
    return etree.tostring(clone, encoding="unicode", method="xml")


def extract_formulae(element: etree._Element) -> list[dict[str, Any]]:
    formulae: list[dict[str, Any]] = []

    for image in element.xpath('.//x:img[contains(@class, "quicklatex")]', namespaces=NS):
        alt = normalized_space(image.get("alt"))
        if not alt:
            continue
        formulae.append(
            {
                "kind": "latex_image",
                "latex": alt,
                "source": image.get("src"),
            }
        )

    for sub in element.xpath(".//x:sub", namespaces=NS):
        value = normalized_space("".join(sub.itertext()))
        if value:
            formulae.append({"kind": "subscript", "value": value})

    for sup in element.xpath(".//x:sup", namespaces=NS):
        value = normalized_space("".join(sup.itertext()))
        if value:
            formulae.append({"kind": "superscript", "value": value})

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in formulae:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def extract_media(element: etree._Element) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    for image in element.xpath(".//x:img", namespaces=NS):
        if "quicklatex" in (image.get("class") or ""):
            continue

        caption = ""
        parent = image.getparent()
        while parent is not None and local_name(parent) != "div":
            parent = parent.getparent()
        if parent is not None:
            captions = parent.xpath(
                './/x:div[contains(@class, "wp-caption-text")]',
                namespaces=NS,
            )
            if captions:
                caption = normalized_space(render_text(captions[0]))

        media.append(
            {
                "kind": "image",
                "src": image.get("src"),
                "alt": normalized_space(image.get("alt")),
                "caption": caption or None,
                "width": image.get("width"),
                "height": image.get("height"),
            }
        )
    return media


def extract_links(element: etree._Element) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for anchor in element.xpath(".//x:a[@href]", namespaces=NS):
        links.append(
            {
                "text": normalized_space(render_text(anchor)),
                "href": anchor.get("href") or "",
                "class": normalized_space(anchor.get("class")),
            }
        )
    return links


def table_to_matrix(table: etree._Element) -> dict[str, Any]:
    caption_nodes = table.xpath("./x:caption", namespaces=NS)
    caption = normalized_space(render_text(caption_nodes[0])) if caption_nodes else None
    rows: list[list[str]] = []

    for row in table.xpath(".//x:tr", namespaces=NS):
        cells = row.xpath("./x:th | ./x:td", namespaces=NS)
        rows.append([normalized_space(render_text(cell)) for cell in cells])

    return {
        "caption": caption,
        "rows": rows,
    }


def classify_textbox(classes: set[str]) -> str:
    mapping = {
        "textbox--learning-objectives": "learning_objectives",
        "textbox--examples": "worked_example",
        "textbox--key-takeaways": "key_takeaways",
        "textbox--exercises": "exercise_set",
    }
    for class_name, content_type in mapping.items():
        if class_name in classes:
            return content_type
    if "shaded" in classes:
        return "feature"
    return "textbox"


def render_inline_prefix(element: etree._Element) -> str:
    """Render inline content before the first nested block-level child.

    Useful for list items such as ``<li>Lead text<p>More details</p></li>``.
    """
    pieces: list[str] = []
    if element.text:
        pieces.append(element.text)

    for child in element:
        if not isinstance(child.tag, str):
            continue
        if local_name(child) in BLOCK_LEVEL_TAGS:
            break

        tag = local_name(child)
        if tag == "sub":
            pieces.append(
                unicode_script("".join(child.itertext()), superscript=False)
            )
        elif tag == "sup":
            pieces.append(
                unicode_script("".join(child.itertext()), superscript=True)
            )
        elif tag == "img":
            alt = normalized_space(child.get("alt"))
            if alt:
                if "quicklatex" in (child.get("class") or ""):
                    pieces.append(f" [FORMULA: {alt}] ")
                else:
                    pieces.append(f" [IMAGE: {alt}] ")
        else:
            clone = deepcopy(child)
            clone.tail = None
            pieces.append(render_text(clone))
        if child.tail:
            pieces.append(child.tail)

    return normalized_space("".join(pieces))


def semantic_children(element: etree._Element) -> list[etree._Element]:
    """Return immediate child elements that carry document structure."""
    return [
        child
        for child in element
        if isinstance(child.tag, str) and local_name(child) in BLOCK_LEVEL_TAGS
    ]


def parse_list_item(element: etree._Element, item_index: int, path: str) -> dict[str, Any]:
    children = [
        parse_semantic_node(child, f"{path}.{child_index}")
        for child_index, child in enumerate(semantic_children(element), start=1)
    ]
    return {
        "item_index": item_index,
        "lead_text": render_inline_prefix(element),
        "text": render_text(element),
        "children": children,
        "html": serialize_inner_html(element),
        "formulae": extract_formulae(element),
        "media": extract_media(element),
        "links": extract_links(element),
    }


def parse_semantic_node(element: etree._Element, path: str) -> dict[str, Any]:
    """Parse a nested XHTML element without flattening its instructional hierarchy."""
    tag = local_name(element)
    classes = set((element.get("class") or "").split())
    node_type = classify_block(element)

    node: dict[str, Any] = {
        "node_path": path,
        "node_type": node_type,
        "html_tag": tag,
        "html_class": normalized_space(element.get("class")),
        "text": render_text(element),
        "html": serialize_inner_html(element),
        "formulae": extract_formulae(element),
        "media": extract_media(element),
        "links": extract_links(element),
    }

    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        node["heading_level"] = int(tag[1])

    if tag in {"ol", "ul"}:
        node["ordered"] = tag == "ol"
        node["start"] = element.get("start")
        node["list_style"] = element.get("type")
        node["items"] = [
            parse_list_item(item, item_index, f"{path}.item{item_index}")
            for item_index, item in enumerate(
                element.xpath("./x:li", namespaces=NS),
                start=1,
            )
        ]
        return node

    if tag == "table":
        node["table"] = table_to_matrix(element)
        return node

    if tag == "div" and "textbox" in classes:
        title_nodes = element.xpath(
            './x:div[contains(@class, "textbox__header")]'
            '//x:*[contains(@class, "textbox__title")][1]',
            namespaces=NS,
        )
        node["title"] = (
            normalized_space(render_text(title_nodes[0])) if title_nodes else None
        )
        content_nodes = element.xpath(
            './x:div[contains(@class, "textbox__content")]',
            namespaces=NS,
        )
        content_root = content_nodes[0] if content_nodes else element
        node["children"] = [
            parse_semantic_node(child, f"{path}.{child_index}")
            for child_index, child in enumerate(
                semantic_children(content_root),
                start=1,
            )
        ]
        return node

    child_elements = semantic_children(element)
    if child_elements and tag not in {"p"}:
        node["children"] = [
            parse_semantic_node(child, f"{path}.{child_index}")
            for child_index, child in enumerate(child_elements, start=1)
        ]

    return node


def semantic_node_to_text(node: dict[str, Any], indent: int = 0) -> str:
    """Serialize a semantic node to readable plain text with structural line breaks."""
    prefix = " " * indent
    node_type = node.get("node_type")
    parts: list[str] = []

    if node.get("title"):
        parts.append(prefix + str(node["title"]))

    if node_type == "heading":
        parts.append(prefix + node.get("text", ""))
        return "\n".join(part for part in parts if part.strip())

    if node_type in {"paragraph", "formula_or_equation"}:
        text = node.get("text", "")
        if text:
            parts.append(prefix + text)
        return "\n".join(part for part in parts if part.strip())

    if "items" in node:
        ordered = bool(node.get("ordered"))
        for item in node.get("items", []):
            marker = f"{item['item_index']}." if ordered else "-"
            lead = item.get("lead_text") or ""
            if lead:
                parts.append(prefix + marker + " " + lead)
            elif not item.get("children"):
                parts.append(prefix + marker + " " + item.get("text", ""))
            else:
                parts.append(prefix + marker)
            for child in item.get("children", []):
                child_text = semantic_node_to_text(child, indent=indent + 3)
                if child_text:
                    parts.append(child_text)
        return "\n".join(part for part in parts if part.strip())

    if node_type == "table" and node.get("table"):
        table = node["table"]
        if table.get("caption"):
            parts.append(prefix + "Table: " + table["caption"])
        for row in table.get("rows", []):
            parts.append(prefix + " | ".join(row))
        return "\n".join(part for part in parts if part.strip())

    children = node.get("children", [])
    if children:
        for child in children:
            child_text = semantic_node_to_text(child, indent=indent)
            if child_text:
                parts.append(child_text)
        return "\n\n".join(part for part in parts if part.strip())

    text = node.get("text", "")
    if text:
        parts.append(prefix + text)
    return "\n".join(part for part in parts if part.strip())



def classify_block(element: etree._Element) -> str:
    tag = local_name(element)
    classes = set((element.get("class") or "").split())

    if tag == "div" and "textbox" in classes:
        return classify_textbox(classes)
    if tag == "table":
        return "table"
    if tag in {"ul", "ol"}:
        return "list"
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return "heading"
    if tag == "div" and "wp-caption" in classes:
        return "figure"
    if tag == "p":
        if element.xpath('.//x:img[contains(@class, "quicklatex")]', namespaces=NS):
            return "formula_or_equation"
        return "paragraph"
    return "other"


def parse_block(element: etree._Element, index: int) -> dict[str, Any]:
    semantic = parse_semantic_node(element, str(index))
    block: dict[str, Any] = {
        "block_index": index,
        "block_type": semantic["node_type"],
        "html_tag": semantic["html_tag"],
        "html_class": semantic["html_class"],
        "text": semantic_node_to_text(semantic),
        "flattened_text": semantic["text"],
        "html": semantic["html"],
        "formulae": semantic["formulae"],
        "media": semantic["media"],
        "links": semantic["links"],
        "structure": semantic,
    }

    tables = element.xpath("self::x:table | .//x:table", namespaces=NS)
    if tables:
        block["tables"] = [table_to_matrix(table) for table in tables]

    return block


def find_part_wrapper(root: etree._Element, chapter_number: int) -> etree._Element:
    prefix = f"part-chapter-{chapter_number}-"
    candidates = root.xpath(
        './/x:div[contains(concat(" ", normalize-space(@class), " "), " part-wrapper ")]',
        namespaces=NS,
    )
    for candidate in candidates:
        if (candidate.get("id") or "").startswith(prefix):
            return candidate

    available = []
    for candidate in candidates:
        candidate_id = candidate.get("id") or ""
        match = re.match(r"part-chapter-(\d+)-", candidate_id)
        if match:
            available.append(int(match.group(1)))
    raise ValueError(
        f"Chapter {chapter_number} was not found. Available chapters: {sorted(set(available))}"
    )


def parse_chapter(
    root: etree._Element,
    source_path: Path,
    chapter_number: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    book = read_book_metadata(root)
    wrapper = find_part_wrapper(root, chapter_number)

    part_nodes = wrapper.xpath(
        './x:div[contains(concat(" ", normalize-space(@class), " "), " part ")]',
        namespaces=NS,
    )
    if not part_nodes:
        raise ValueError(f"Chapter {chapter_number} has no Pressbooks part node.")

    part = part_nodes[0]
    title_nodes = part.xpath('.//x:h1[contains(@class, "part-title")][1]', namespaces=NS)
    chapter_title = (
        normalized_space(render_text(title_nodes[0]))
        if title_nodes
        else f"Chapter {chapter_number}"
    )

    intro_nodes = part.xpath('./x:div[contains(@class, "part-ugc")]', namespaces=NS)
    chapter_intro = render_text(intro_nodes[0]) if intro_nodes else ""

    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_rel = source_path.name

    sections: list[dict[str, Any]] = []
    chapter_nodes = wrapper.xpath(
        './x:div[contains(concat(" ", normalize-space(@class), " "), " chapter ")]',
        namespaces=NS,
    )

    for section_index, chapter_node in enumerate(chapter_nodes, start=1):
        source_html_id = chapter_node.get("id") or f"chapter-{chapter_number}-{section_index}"
        section_id = f"introchem_ch{chapter_number:02d}_{source_html_id.removeprefix('chapter-')}"
        section_title = normalized_space(chapter_node.get("title"))

        source_number_nodes = chapter_node.xpath(
            './x:div[contains(@class, "chapter-title-wrap")]/x:p[contains(@class, "chapter-number")]',
            namespaces=NS,
        )
        source_section_number = (
            normalized_space(render_text(source_number_nodes[0]))
            if source_number_nodes
            else None
        )

        content_nodes = chapter_node.xpath(
            './x:div[contains(@class, "chapter-ugc")]',
            namespaces=NS,
        )
        if not content_nodes:
            blocks: list[dict[str, Any]] = []
        else:
            blocks = [
                parse_block(element, block_index)
                for block_index, element in enumerate(content_nodes[0], start=1)
            ]

        full_text = "\n\n".join(
            block["text"] for block in blocks if block["text"]
        ).strip()

        block_type_counts: dict[str, int] = {}
        for block in blocks:
            block_type_counts[block["block_type"]] = (
                block_type_counts.get(block["block_type"], 0) + 1
            )

        section_record = {
            "schema_version": SCHEMA_VERSION,
            "id": section_id,
            "parent_id": f"introchem_ch{chapter_number:02d}",
            "source_id": book.source_id,
            "source_file": source_rel,
            "source_format": "pressbooks_xhtml",
            "source_html_id": source_html_id,
            "source_section_number": source_section_number,
            "language": book.language,
            "document_type": "textbook_section",
            "content_type": "mixed",
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "section_index": section_index,
            "derived_section_number": f"{chapter_number}.{section_index}",
            "section_title": section_title,
            "text": full_text,
            "blocks": blocks,
            "block_type_counts": block_type_counts,
            "license": book.license_code,
            "attribution": {
                "title": book.title,
                "authors": book.authors,
                "publisher": book.publisher,
                "publication_year": book.publication_year,
            },
            "parser_version": PARSER_VERSION,
            "source_sha256": source_hash,
            "content_sha256": sha256_text(full_text),
            "review_status": "unreviewed",
        }
        sections.append(section_record)

    chapter_record = {
        "schema_version": SCHEMA_VERSION,
        "id": f"introchem_ch{chapter_number:02d}",
        "source_id": book.source_id,
        "source_file": source_rel,
        "source_format": "pressbooks_xhtml",
        "language": book.language,
        "document_type": "textbook_chapter",
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "chapter_intro": chapter_intro,
        "section_ids": [section["id"] for section in sections],
        "section_count": len(sections),
        "license": book.license_code,
        "attribution": {
            "title": book.title,
            "authors": book.authors,
            "publisher": book.publisher,
            "publication_year": book.publication_year,
        },
        "parser_version": PARSER_VERSION,
        "source_sha256": source_hash,
        "review_status": "unreviewed",
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "source_file": source_rel,
        "source_sha256": source_hash,
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "section_count": len(sections),
        "total_blocks": sum(len(section["blocks"]) for section in sections),
        "total_characters": sum(len(section["text"]) for section in sections),
        "total_formula_records": sum(
            len(block["formulae"])
            for section in sections
            for block in section["blocks"]
        ),
        "total_media_records": sum(
            len(block["media"])
            for section in sections
            for block in section["blocks"]
        ),
        "total_tables": sum(
            len(block.get("tables", []))
            for section in sections
            for block in section["blocks"]
        ),
        "sections": [
            {
                "id": section["id"],
                "title": section["section_title"],
                "characters": len(section["text"]),
                "blocks": len(section["blocks"]),
                "block_type_counts": section["block_type_counts"],
            }
            for section in sections
        ],
    }

    return chapter_record, sections, report


def indent_markdown(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else prefix.rstrip() for line in text.splitlines())


def semantic_node_to_markdown(node: dict[str, Any], depth: int = 4) -> list[str]:
    node_type = node.get("node_type")
    text = node.get("text", "").strip()
    lines: list[str] = []

    if node_type == "heading":
        level = min(6, max(depth, depth + int(node.get("heading_level", 1)) - 1))
        return [f"{'#' * level} {text}", ""]

    if node_type in {"paragraph", "formula_or_equation"}:
        return [text or "_[empty paragraph]_", ""]

    if node_type in {"list", "ordered_list", "unordered_list"} or "items" in node:
        ordered = bool(node.get("ordered"))
        for item in node.get("items", []):
            marker = f"{item['item_index']}." if ordered else "-"
            lead = item.get("lead_text") or ""
            if lead:
                lines.append(f"{marker} {lead}")
            elif not item.get("children"):
                lines.append(f"{marker} {item.get('text', '')}")
            else:
                lines.append(marker)

            for child in item.get("children", []):
                rendered = semantic_node_to_markdown(child, depth=min(depth + 1, 6))
                body = "\n".join(rendered).rstrip()
                if body:
                    lines.append(indent_markdown(body, "   "))
        lines.append("")
        return lines

    if node_type == "table" and node.get("table"):
        table = node["table"]
        if table.get("caption"):
            lines.append(f"**Table:** {table['caption']}")
        for row in table.get("rows", []):
            lines.append(" | ".join(row))
        lines.append("")
        return lines

    title = node.get("title")
    if title:
        lines.extend([f"**{title}**", ""])

    children = node.get("children", [])
    if children:
        for child in children:
            lines.extend(semantic_node_to_markdown(child, depth=depth))
        return lines

    return [text or "_[empty structured node]_", ""]


def make_review_markdown(
    chapter: dict[str, Any],
    sections: list[dict[str, Any]],
    report: dict[str, Any],
) -> str:
    lines = [
        f"# Parse review: {chapter['chapter_title']}",
        "",
        f"- Source: `{chapter['source_file']}`",
        f"- Parser: `{chapter['parser_version']}`",
        f"- Sections: {chapter['section_count']}",
        f"- Total blocks: {report['total_blocks']}",
        f"- Formula records: {report['total_formula_records']}",
        f"- Media records: {report['total_media_records']}",
        f"- Tables: {report['total_tables']}",
        "",
        "This file is for human review. The canonical data are in `chapter.json` and `sections.jsonl`.",
        "Nested headings, paragraphs, lists, and list items are rendered separately.",
        "",
    ]

    for section in sections:
        lines.extend(
            [
                f"## {section['derived_section_number']} {section['section_title']}",
                "",
                f"- ID: `{section['id']}`",
                f"- Source HTML ID: `{section['source_html_id']}`",
                f"- Source section number: `{section['source_section_number']}`",
                f"- Blocks: {len(section['blocks'])}",
                f"- Characters: {len(section['text'])}",
                f"- Block types: `{json.dumps(section['block_type_counts'], ensure_ascii=False)}`",
                "",
            ]
        )

        for block in section["blocks"]:
            lines.extend(
                [
                    f"### Block {block['block_index']}: `{block['block_type']}`",
                    "",
                ]
            )
            lines.extend(semantic_node_to_markdown(block["structure"], depth=4))

    return "\n".join(lines).rstrip() + "\n"


def validate_review_structure(
    sections: list[dict[str, Any]],
    markdown: str,
) -> dict[str, Any]:
    """Validate that instructional textboxes remain visibly structured.

    The check is deliberately performed on the generated Markdown, not only on
    the JSON tree. This catches regressions where the canonical structure is
    present but ``review.md`` accidentally flattens it for human inspection.
    """
    checked_examples: list[dict[str, Any]] = []
    errors: list[str] = []

    for section in sections:
        for block in section.get("blocks", []):
            if block.get("block_type") != "worked_example":
                continue

            structure = block.get("structure", {})
            title = normalized_space(structure.get("title"))
            heading_texts = [
                normalized_space(child.get("text"))
                for child in structure.get("children", [])
                if child.get("node_type") == "heading"
            ]

            if title:
                title_marker = f"**{title}**"
                if title_marker not in markdown:
                    errors.append(f"Missing worked-example title in review: {title_marker}")

            for heading in heading_texts:
                # Nested Pressbooks h1 starts at Markdown h4 in review.md.
                heading_pattern = re.compile(
                    rf"^#{{4,6}}\s+{re.escape(heading)}\s*$",
                    flags=re.MULTILINE,
                )
                if not heading_pattern.search(markdown):
                    errors.append(
                        f"Missing nested heading in review for {title or '[untitled]'}: {heading}"
                    )

            checked_examples.append(
                {
                    "section_id": section.get("id"),
                    "block_index": block.get("block_index"),
                    "title": title or None,
                    "headings": heading_texts,
                }
            )

    # A focused regression test for the first balancing example in chapter 4.
    if any(item.get("title") == "Example 4.1" for item in checked_examples):
        expected_sequence = re.compile(
            r"\*\*Example 4\.1\*\*\s+"
            r"####\s+Problems\s+"
            r"Write and balance the chemical equation for each given chemical reaction\.",
            flags=re.MULTILINE,
        )
        if not expected_sequence.search(markdown):
            errors.append(
                "Regression check failed: Example 4.1 is not rendered as "
                "title -> Problems heading -> instruction paragraph."
            )

    result = {
        "parser_version": PARSER_VERSION,
        "worked_examples_checked": len(checked_examples),
        "examples": checked_examples,
        "ok": not errors,
        "errors": errors,
    }

    if errors:
        raise RuntimeError(
            "Review structure validation failed:\n- " + "\n- ".join(errors)
        )

    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one chapter from a Pressbooks XHTML export into a reviewable "
            "canonical JSON/JSONL representation."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to XHTML/HTML export")
    parser.add_argument("--chapter", type=int, default=4, help="Chapter number to parse")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for normalized files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    parser = etree.XMLParser(
        recover=True,
        huge_tree=True,
        resolve_entities=False,
        no_network=True,
    )
    tree = etree.parse(str(input_path), parser)
    root = tree.getroot()

    chapter, sections, report = parse_chapter(
        root=root,
        source_path=input_path,
        chapter_number=args.chapter,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "chapter.json", chapter)
    write_jsonl(output_dir / "sections.jsonl", sections)
    write_json(output_dir / "parse_report.json", report)
    review_markdown = make_review_markdown(chapter, sections, report)
    review_validation = validate_review_structure(sections, review_markdown)
    (output_dir / "review.md").write_text(
        review_markdown,
        encoding="utf-8",
    )
    write_json(output_dir / "review_validation.json", review_validation)

    print(f"Parser version: {PARSER_VERSION}")
    print(f"Review structure check: OK ({review_validation['worked_examples_checked']} examples)")
    print(f"Parsed: {chapter['chapter_title']}")
    print(f"Sections: {report['section_count']}")
    print(f"Blocks: {report['total_blocks']}")
    print(f"Formula records: {report['total_formula_records']}")
    print(f"Media records: {report['total_media_records']}")
    print(f"Tables: {report['total_tables']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
