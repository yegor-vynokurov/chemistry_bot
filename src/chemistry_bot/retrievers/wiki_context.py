"""Wikipedia context enrichment for the chemistry bot.

This module is intentionally not an agent tool. It is a small, controlled
retrieval layer that can add short Wikipedia snippets to the prompt as
illustrative background. The final answer must still obey the bot's school
context and safety rules.

The first retrieval path uses LangChain's WikipediaRetriever, because it fits
LangChain/RAG experiments nicely. If that path fails with a JSONDecodeError or
another HTTP/API problem, the module falls back to a small direct MediaWiki API
client with an explicit User-Agent header. This avoids a common failure mode
where Wikimedia returns an empty or non-JSON response and the underlying
library raises: "Expecting value: line 1 column 1 (char 0)".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any


@dataclass
class WikiContextConfig:
    """Settings for controlled Wikipedia enrichment."""

    enabled: bool = True
    # The chatbot answers in English, so English Wikipedia is the default.
    # The CLI command `/wiki lang ru` or `/wiki lang uk` can still switch it.
    lang: str = "en"
    top_k_results: int = 2
    doc_content_chars_max: int = 1600
    formatted_context_chars_max: int = 3600
    query_suffix: str = "chemistry"

    # Keep LangChain in the pipeline, but use direct MediaWiki API as a fallback
    # when LangChain's wrapper or the underlying wikipedia package fails.
    use_langchain_retriever: bool = True
    use_mediawiki_fallback: bool = True

    # Wikimedia asks API clients to identify themselves. For a local learning
    # project this generic value is enough for experiments; for a public/server
    # deployment, replace it with an app name plus contact URL or email.
    user_agent: str = (
        "chemistry-bot/0.1 "
        "(local educational project; contact: replace-with-your-email-or-github)"
    )
    request_timeout_seconds: float = 15.0

    # Auto mode retrieves Wikipedia only when a question looks like it needs
    # a real-world reference, an encyclopedic definition, history, or a reaction
    # example. The /wiki once command can still force retrieval for the next turn.
    auto_mode: bool = True

    # Queries that ask for quantities, protocols, or home synthesis should not
    # use Wikipedia snippets as extra context. The answer can still discuss the
    # idea safely, but not by importing procedural details from the web.
    blocked_patterns: tuple[str, ...] = (
        # English patterns.
        r"\b(at\s+home|home\s+experiment|by\s+myself|without\s+a\s+teacher)\b",
        r"\b(recipe|protocol|procedure|step[-\s]?by[-\s]?step|instructions?|how\s+to\s+make)\b",
        r"\b(how\s+much|dosage|ratio|proportion|concentration|volume|mass|amount)\b",
        r"\b(ml|millilitre|milliliter|liter|litre|gram|grams|molar|mol/l|%)\b",
        r"\b(synthesize|manufacture|prepare\s+at\s+home|make\s+at\s+home)\b",
        r"\b(explosive|gunpowder|detonator|poison|toxic\s+gas)\b",
        # Russian/Ukrainian fallback patterns, useful if the user tests the bot
        # in another language while Wikipedia remains English.
        r"\b(в\s+домашних\s+условиях|дома|самостоятельно|без\s+учителя)\b",
        r"\b(рецепт|протокол|методик[ауи]|инструкц|пошагов)\b",
        r"\b(сколько|дозиров|пропорц|концентрац|об[ъь]ем|масса)\b",
        r"\b(мл|литр|грамм|моль/л|процент)\b",
        r"\b(синтезировать|изготовить|сделать|получить\s+дома)\b",
        r"\b(взрывчат|порох|детонатор|яд|отравляющ)\b",
    )

    retrieval_hints: tuple[str, ...] = (
        # English hints.
        "example",
        "examples",
        "illustration",
        "reaction",
        "equation",
        "obtaining",
        "preparation",
        "produced",
        "formed",
        "substance",
        "element",
        "compound",
        "oxygen",
        "hydrogen",
        "ammonia",
        "carbon dioxide",
        "peroxide",
        "oxide",
        "acid",
        "base",
        "salt",
        "redox",
        "oxidation",
        "reduction",
        "history",
        "discovered",
        "who discovered",
        "what is",
        "difference between",
        "why",
        # Russian fallback hints.
        "пример",
        "примеры",
        "иллюстрац",
        "реакц",
        "уравнен",
        "получени",
        "образуется",
        "кислород",
        "водород",
        "пероксид",
        "оксид",
        "кислота",
    )

    unsafe_detail_patterns: tuple[str, ...] = (
        r"\b\d+[\.,]?\d*\s?(ml|l|g|mg|kg|mol|mol/l|m|%)\b",
        r"\b\d+[\.,]?\d*\s?(°c|celsius|kelvin|k)\b",
        r"\b(heat(?:ed)?\s+to|cool(?:ed)?\s+to|add(?:ed)?\s+dropwise)\b",
        r"\b(flask|test\s+tube|burner|stand|funnel|gas\s+delivery)\b.*\b(heat|add|mix)\b",
        r"\b\d+[\.,]?\d*\s?(мл|л|г|мг|кг|моль|моль/л|%)\b",
        r"\b\d+[\.,]?\d*\s?(°c|кельвин)\b",
        r"\b(нагревают?\s+до|охлаждают?\s+до|добавляют?\s+по\s+каплям)\b",
        r"\b(колба|пробирка|горелка|штатив|воронка|газоотводн)\b.*\b(нагре|добав|смеш)\b",
    )


@dataclass
class _WikiDoc:
    """Small Document-like container used by the MediaWiki fallback."""

    page_content: str
    metadata: dict[str, str]


@dataclass
class WikipediaContextBuilder:
    """Builds a safe prompt block from WikipediaRetriever results."""

    config: WikiContextConfig = field(default_factory=WikiContextConfig)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__),
        repr=False,
    )
    _retriever: Any | None = field(default=None, init=False, repr=False)

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled

    def set_language(self, lang: str) -> None:
        lang = lang.strip().lower()
        if not lang:
            return
        self.config.lang = lang
        # Keep the query suffix aligned with the default language.
        if lang == "en":
            self.config.query_suffix = "chemistry"
        elif lang == "ru":
            self.config.query_suffix = "химия"
        elif lang == "uk":
            self.config.query_suffix = "хімія"
        self._retriever = None

    def status(self) -> str:
        mode = "auto" if self.config.auto_mode else "always"
        enabled = "on" if self.config.enabled else "off"
        retriever = "langchain+fallback" if self.config.use_langchain_retriever else "mediawiki"
        return (
            f"Wikipedia: {enabled}, lang={self.config.lang}, "
            f"mode={mode}, top_k={self.config.top_k_results}, retriever={retriever}"
        )

    def should_retrieve(self, question: str, *, force: bool = False) -> tuple[bool, str]:
        text = question.strip().lower()

        if not self.config.enabled:
            return False, "Wikipedia context is disabled."

        if len(text) < 4:
            return False, "Question is too short for Wikipedia retrieval."

        for pattern in self.config.blocked_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return (
                    False,
                    "Query looks procedural or quantity-focused; Wikipedia context skipped for safety.",
                )

        if force:
            return True, "Forced by /wiki once."

        if not self.config.auto_mode:
            return True, "Wikipedia context is set to always-on mode."

        if any(hint in text for hint in self.config.retrieval_hints):
            return True, "Question matches encyclopedia/reaction-example hints."

        return False, "Question does not need Wikipedia context in auto mode."

    def build(self, question: str, *, force: bool = False) -> str:
        """Return a formatted Wikipedia context block, or an empty string."""

        allowed, reason = self.should_retrieve(question, force=force)
        if not allowed:
            self.logger.info("Wikipedia context skipped: %s", reason)
            return ""

        query = self._make_query(question)
        errors: list[str] = []
        docs: list[Any] = []

        if self.config.use_langchain_retriever:
            try:
                docs = self._get_retriever().invoke(query)
            except Exception as error:  # noqa: BLE001 - fallback should handle this
                errors.append(f"LangChain WikipediaRetriever failed: {error}")
                self.logger.warning(
                    "LangChain WikipediaRetriever failed; trying MediaWiki fallback: %s",
                    error,
                    exc_info=True,
                )

        if not docs and self.config.use_mediawiki_fallback:
            try:
                docs = self._retrieve_mediawiki_docs(query)
            except Exception as error:  # noqa: BLE001 - retrieval should not break chat
                errors.append(f"MediaWiki fallback failed: {error}")
                self.logger.exception("MediaWiki fallback retrieval failed")

        if not docs:
            if errors:
                return (
                    "Wikipedia context was requested, but retrieval failed. "
                    + " | ".join(errors)
                )
            return ""

        return self._format_docs(docs)

    def _get_retriever(self):
        if self._retriever is None:
            try:
                from langchain_community.retrievers import WikipediaRetriever
            except ImportError as error:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "Install Wikipedia dependencies: "
                    "pip install -U langchain-community wikipedia"
                ) from error

            self._retriever = WikipediaRetriever(
                top_k_results=self.config.top_k_results,
                lang=self.config.lang,
                doc_content_chars_max=self.config.doc_content_chars_max,
            )

        return self._retriever

    def _make_query(self, question: str) -> str:
        query = question.strip()
        suffix = self.config.query_suffix.strip()
        if suffix and suffix.lower() not in query.lower():
            query = f"{query} {suffix}"
        return query

    def _retrieve_mediawiki_docs(self, query: str) -> list[_WikiDoc]:
        try:
            import requests
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("Install requests: pip install requests") from error

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept": "application/json",
            }
        )

        api_url = f"https://{self.config.lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": str(self.config.top_k_results),
            "utf8": "1",
        }

        search_json = self._get_json(session, api_url, search_params)
        results = search_json.get("query", {}).get("search", [])

        docs: list[_WikiDoc] = []
        for result in results[: self.config.top_k_results]:
            title = str(result.get("title", "")).strip()
            if not title:
                continue

            extract_params = {
                "action": "query",
                "format": "json",
                "prop": "extracts|info",
                "titles": title,
                "explaintext": "1",
                "exintro": "1",
                "redirects": "1",
                "inprop": "url",
                "utf8": "1",
            }
            page_json = self._get_json(session, api_url, extract_params)
            pages = page_json.get("query", {}).get("pages", {})
            if not isinstance(pages, dict):
                continue

            for page in pages.values():
                if not isinstance(page, dict):
                    continue
                extract = str(page.get("extract", "")).strip()
                if not extract:
                    # Fallback to the search snippet, stripped from tiny HTML bits.
                    extract = self._strip_html(str(result.get("snippet", ""))).strip()
                if not extract:
                    continue

                docs.append(
                    _WikiDoc(
                        page_content=extract[: self.config.doc_content_chars_max],
                        metadata={
                            "title": str(page.get("title", title)),
                            "source": str(page.get("fullurl", "")),
                        },
                    )
                )
                break

        return docs

    def _get_json(self, session: Any, url: str, params: dict[str, str]) -> dict[str, Any]:
        response = session.get(
            url,
            params=params,
            timeout=self.config.request_timeout_seconds,
        )

        content_type = response.headers.get("content-type", "")
        body_preview = response.text[:300].replace("\n", " ").strip()

        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code} from MediaWiki API; "
                f"content-type={content_type!r}; body={body_preview!r}"
            )

        try:
            data = response.json()
        except ValueError as error:
            raise RuntimeError(
                "MediaWiki API did not return valid JSON; "
                f"content-type={content_type!r}; body={body_preview!r}"
            ) from error

        if not isinstance(data, dict):
            raise RuntimeError("MediaWiki API returned JSON, but not an object.")

        if "error" in data:
            raise RuntimeError(f"MediaWiki API error: {data['error']}")

        return data

    def _format_docs(self, docs: list[Any]) -> str:
        chunks: list[str] = [
            "Additional reference context from Wikipedia "
            "(for illustrations only, not a laboratory procedure):",
            "Rules for using this block:",
            "- Use it only as background for explaining concepts and examples of real reactions.",
            "- Do not turn it into a recipe, protocol, synthesis instruction, or home experiment.",
            "- Do not add exact masses, volumes, concentrations, temperatures, heating times, or step-by-step operations.",
            "- If the student asks how to perform something practically, answer safely at the level of a school demonstration idea.",
        ]

        used = 0
        for index, doc in enumerate(docs, start=1):
            title = str(getattr(doc, "metadata", {}).get("title", "Wikipedia article"))
            source = str(getattr(doc, "metadata", {}).get("source", ""))
            text = self._sanitize_text(str(getattr(doc, "page_content", "")))
            if not text:
                continue

            block = f"\n[Wiki {index}] {title}\nSource: {source}\n{text}"
            if used + len(block) > self.config.formatted_context_chars_max:
                remaining = self.config.formatted_context_chars_max - used
                if remaining <= 250:
                    break
                block = block[:remaining].rsplit(" ", 1)[0] + "..."

            chunks.append(block)
            used += len(block)

        # Header has 6 chunks. If no article block was added, return nothing.
        if len(chunks) <= 6:
            return ""

        return "\n".join(chunks)

    def _sanitize_text(self, text: str) -> str:
        text = self._strip_html(text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        safe_sentences: list[str] = []

        for sentence in sentences:
            sentence_l = sentence.lower()
            if any(
                re.search(pattern, sentence_l, flags=re.IGNORECASE)
                for pattern in self.config.unsafe_detail_patterns
            ):
                continue
            safe_sentences.append(sentence)

        return " ".join(safe_sentences).strip()

    def _strip_html(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("&quot;", '"')
        text = text.replace("&#039;", "'")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        return text
