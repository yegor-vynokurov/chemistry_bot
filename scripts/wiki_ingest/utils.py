from __future__ import annotations

import re
import shutil
import unicodedata
from datetime import date
from pathlib import Path
from typing import Iterable

from charset_normalizer import from_bytes

SUPPORTED_EXTENSIONS = {".pdf", ".mhtml", ".mht", ".html", ".htm", ".md", ".txt"}


def today_iso() -> str:
    return date.today().isoformat()


def replacement_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count("\ufffd") / max(1, len(text))


def cyrillic_score(text: str) -> int:
    return sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch.lower() in "іїєґ")


def latin_score(text: str) -> int:
    return sum(1 for ch in text if "a" <= ch.lower() <= "z")


def weird_char_score(text: str) -> int:
    """Penalty for characters often produced by wrong single-byte decoders."""
    if not text:
        return 0
    bad = text.count("\ufffd")
    bad += text.count("\x80") + text.count("\x81") + text.count("\x8d") + text.count("\x8f") + text.count("\x90") + text.count("\x9d")
    bad += text.count("Â") + text.count("Ã") + text.count("Ð") + text.count("Ñ")
    # NBSP is normal in HTML, but lots of NBSP inside words is suspicious.
    bad += min(text.count("\xa0"), 50)
    return bad


def looks_utf8_as_iso_8859_5(text: str) -> bool:
    # UTF-8 Cyrillic bytes decoded as iso-8859-5 often looks like:
    # а аЕаГбаЕбб...
    if not text:
        return False
    sample = text[:2000]
    patterns = (
        "аЕ", "аГ", "аИ", "аО", "аН", "аЛ", "аА",
        "б", "б", "б", "б", "а", "а", "аЅ",
    )
    return sum(sample.count(p) for p in patterns) >= 5


def looks_mojibake(text: str) -> bool:
    if not text:
        return False
    if replacement_ratio(text) > 0.002:
        return True
    if looks_utf8_as_iso_8859_5(text):
        return True
    markers = ("Ð", "Ñ", "Рќ", "Р°", "Рµ", "СЃ", "â€”", "â€“", "â€œ", "â€")
    return sum(text.count(m) for m in markers) >= 3


def _strict_decode(raw: bytes, enc: str) -> str | None:
    try:
        return raw.decode(enc, errors="strict")
    except (UnicodeDecodeError, LookupError):
        return None


def decode_bytes_best(raw: bytes, declared_encoding: str | None = None) -> str:
    """Decode bytes for HTML/MHTML/text.

    v1.5.1 could accidentally prefer iso-8859-5 for valid UTF-8 Russian text,
    producing strings like "аЕаГб..." instead of "Регр...".

    v1.5.2 rule:
    1. If declared encoding decodes strictly and does not look broken, use it.
    2. If UTF-8 decodes strictly and does not look broken, use UTF-8.
    3. Otherwise rank legacy encodings by fewer replacement/weird chars and more Cyrillic.
    """
    declared = (declared_encoding or "").strip().strip('"').strip("'")

    # Some pages lie with "charset=iso-8859-1" while carrying UTF-8 bytes,
    # so only trust declared encoding when decoded text does not look broken.
    if declared:
        text = _strict_decode(raw, declared)
        if text is not None and not looks_mojibake(text):
            return text

    utf8 = _strict_decode(raw, "utf-8")
    if utf8 is not None and not looks_mojibake(utf8):
        return utf8

    utf8_sig = _strict_decode(raw, "utf-8-sig")
    if utf8_sig is not None and not looks_mojibake(utf8_sig):
        return utf8_sig

    candidates: list[tuple[str, str]] = []
    encs = []
    if declared:
        encs.append(declared)
    encs.extend([
        "windows-1251",
        "cp1251",
        "koi8-r",
        "windows-1252",
        "latin-1",
        # Keep iso-8859-5 as last-resort only. It can decode UTF-8 bytes into fake Cyrillic.
        "iso-8859-5",
    ])

    seen = set()
    for enc in encs:
        if not enc or enc.lower() in seen:
            continue
        seen.add(enc.lower())
        try:
            candidates.append((enc, raw.decode(enc, errors="replace")))
        except LookupError:
            continue

    try:
        result = from_bytes(raw).best()
        if result is not None:
            candidates.append(("charset-normalizer", str(result)))
    except Exception:
        pass

    if not candidates:
        return raw.decode("utf-8", errors="replace")

    def score(item: tuple[str, str]) -> tuple[int, float, int, int]:
        enc, text = item
        iso_penalty = 1000 if enc.lower() == "iso-8859-5" and looks_utf8_as_iso_8859_5(text) else 0
        # replacement/weird first; then prefer more Cyrillic; then longer text.
        return (
            iso_penalty + weird_char_score(text),
            replacement_ratio(text),
            -cyrillic_score(text),
            -len(text),
        )

    best_enc, best_text = sorted(candidates, key=score)[0]

    # Rescue common mojibake: UTF-8 bytes decoded as CP1251 gives "Рµ",
    # UTF-8 bytes decoded as ISO-8859-5 gives "аЕ".
    if looks_mojibake(best_text):
        utf8_replace = raw.decode("utf-8", errors="replace")
        if replacement_ratio(utf8_replace) <= replacement_ratio(best_text) and cyrillic_score(utf8_replace) >= cyrillic_score(best_text) * 0.6:
            best_text = utf8_replace

    return best_text


def read_text_file(path: Path) -> str:
    return decode_bytes_best(path.read_bytes())


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_for_prompt(text: str, max_chars: int = 14000) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.55)
    middle = int(max_chars * 0.20)
    tail = max_chars - head - middle - 300
    mid_start = max(0, len(text) // 2 - middle // 2)
    mid_end = min(len(text), mid_start + middle)
    return text[:head] + "\n\n[... middle excerpt ...]\n\n" + text[mid_start:mid_end] + "\n\n[... end excerpt ...]\n\n" + text[-tail:]


def ascii_slug(value: str, max_len: int = 96) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        value = "untitled-source"
    return value[:max_len].strip("-")


def filename_stem_slug(path: Path) -> str:
    stem = path.stem.replace(" ", "-")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem)
    stem = stem.strip("._-")
    return ascii_slug(stem.replace("_", "-")).replace("-", "_")


def detect_language(text: str) -> str:
    sample = (text or "")[:8000]
    if not sample.strip():
        return "unknown"
    cyr = cyrillic_score(sample)
    uk_specific = sum(1 for ch in sample if ch.lower() in "іїєґ")
    latin = latin_score(sample)
    if cyr == 0 and latin == 0:
        return "unknown"
    if cyr > latin * 0.4 and latin > cyr * 0.4:
        return "mixed"
    if cyr > latin:
        return "uk" if uk_specific >= max(3, cyr * 0.015) else "ru"
    return "en"


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def iter_supported_files(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def ensure_unique_path(path: Path, overwrite: bool = False) -> Path:
    if overwrite or not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def move_to_skipped_area(path: Path, vault_root: Path, skipped_root: Path, reason: str, dry_run: bool = False) -> Path:
    try:
        rel = path.resolve().relative_to(vault_root.resolve())
    except ValueError:
        rel = Path(path.name)
    target = ensure_unique_path(skipped_root / reason / rel)
    if dry_run:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


def yaml_scalar(value: object) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = re.sub(r"[\r\n]+", " ", s)
    return f'"{s}"'


def yaml_list(values: Iterable[str], indent: int = 2, quote_links: bool = False) -> str:
    prefix = " " * indent
    lines: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        if quote_links or value.startswith("[[") or ":" in value or value.startswith(("-", "[", "{", "#")):
            lines.append(f"{prefix}- {yaml_scalar(value)}")
        else:
            lines.append(f"{prefix}- {value}")
    return "\n".join(lines) if lines else f"{prefix}[]"


def unique_keep_order(values: Iterable[str], max_items: int = 20) -> list[str]:
    out: list[str] = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= max_items:
            break
    return out
