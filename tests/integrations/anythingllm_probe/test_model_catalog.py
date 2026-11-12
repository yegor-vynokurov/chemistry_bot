from __future__ import annotations

import json
import os
import sys
import unittest
from threading import Lock
from unittest.mock import patch
from urllib.error import URLError

from chemistry_bot.integrations.anythingllm_probe import model_catalog
from chemistry_bot.integrations.anythingllm_probe import __main__ as probe_main
from chemistry_bot.integrations.anythingllm_probe.model_catalog import (
    OllamaCatalogUnavailable,
    OllamaModelCatalog,
    TutorModelUnavailable,
    load_tutor_model_configuration,
)
from chemistry_bot.tutor_graph.graph import TutorGraphConfig
from chemistry_bot.integrations.anythingllm_probe.tutor_bridge import TutorGraphBridge


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class OllamaModelCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = OllamaModelCatalog(
            base_url="http://ollama:11434/",
            allowed_models=("phi4-mini:latest", "qwen3.5:9b-q4_K_M"),
        )

    def test_loads_shared_tutor_and_support_model_roster(self) -> None:
        tutor_models, support_models = load_tutor_model_configuration()

        self.assertIn("phi4-mini:latest", tutor_models)
        self.assertIn("qwen3.5:9b-q4_K_M", tutor_models)
        self.assertIn("embeddinggemma:latest", support_models)

    @patch.object(
        model_catalog,
        "urlopen",
        return_value=_FakeResponse(
            {
                "models": [
                    {"name": "qwen3.5:9b-q4_K_M"},
                    {"model": "phi4-mini:latest"},
                    {"model": "embeddinggemma:latest"},
                    {"model": "unsupported:latest"},
                ]
            }
        ),
    )
    def test_lists_only_installed_allowlisted_tutor_models(self, _urlopen: object) -> None:
        self.assertEqual(
            self.catalog.installed_tutor_models(),
            ("phi4-mini:latest", "qwen3.5:9b-q4_K_M"),
        )
        self.assertEqual(
            [item["id"] for item in self.catalog.openai_model_list()["data"]],
            ["phi4-mini:latest", "qwen3.5:9b-q4_K_M"],
        )

    @patch.object(
        model_catalog,
        "urlopen",
        return_value=_FakeResponse({"models": [{"model": "phi4-mini:latest"}]}),
    )
    def test_resolves_legacy_id_to_startup_default(self, _urlopen: object) -> None:
        self.assertEqual(
            self.catalog.resolve_requested_model(
                "chemistry-tutor",
                default_model="phi4-mini:latest",
                legacy_model_id="chemistry-tutor",
            ),
            "phi4-mini:latest",
        )

    def test_rejects_unconfigured_selection_without_calling_backend(self) -> None:
        with patch.object(model_catalog, "urlopen") as urlopen:
            with self.assertRaises(TutorModelUnavailable):
                self.catalog.resolve_requested_model(
                    "unknown:latest", default_model="phi4-mini:latest"
                )
        urlopen.assert_not_called()

    @patch.object(
        model_catalog,
        "urlopen",
        return_value=_FakeResponse({"models": [{"model": "phi4-mini:latest"}]}),
    )
    def test_rejects_allowlisted_but_uninstalled_selection(self, _urlopen: object) -> None:
        with self.assertRaises(TutorModelUnavailable):
            self.catalog.resolve_requested_model(
                "qwen3.5:9b-q4_K_M", default_model="phi4-mini:latest"
            )

    @patch.object(model_catalog, "urlopen", side_effect=URLError("offline"))
    def test_reports_unavailable_ollama_without_fallback(self, _urlopen: object) -> None:
        with self.assertRaises(OllamaCatalogUnavailable):
            self.catalog.installed_tutor_models()


class EntrypointModelCatalogSettingsTests(unittest.TestCase):
    @patch.object(probe_main.uvicorn, "run")
    @patch.object(probe_main, "create_app")
    def test_main_forwards_active_backend_and_model_roster(
        self, create_app: object, _uvicorn_run: object
    ) -> None:
        """Give the picker catalog the same runtime backend and roster as Compose."""
        with (
            patch.dict(
                os.environ,
                {
                    "OLLAMA_HOST": "http://ollama:11434",
                    "CHEM_TUTOR_MODEL_OPTIONS": "phi4-mini:latest",
                },
            ),
            patch.object(sys, "argv", ["chemistry-tutor"]),
        ):
            probe_main.main()

        settings = create_app.call_args.args[0]  # type: ignore[attr-defined]
        self.assertEqual(settings.ollama_base_url, "http://ollama:11434")
        self.assertEqual(settings.tutor_model_options, ("phi4-mini:latest",))


class TutorGraphModelSelectionTests(unittest.TestCase):
    def test_selected_graph_varies_only_tutor_generation_model_and_is_cached(self) -> None:
        original = TutorGraphConfig(
            model_name="phi4-mini:latest",
            classifier_model_name="phi4-mini:latest",
            turn_relation_model_name="phi4-mini:latest",
            followup_model_name="phi4-mini:latest",
            safety_embedding_model_name="embeddinggemma:latest",
        )
        bridge = TutorGraphBridge.__new__(TutorGraphBridge)
        bridge.graph_config = original
        bridge._graphs = {original.model_name: object()}
        bridge._graph_lock = Lock()

        with patch(
            "chemistry_bot.integrations.anythingllm_probe.tutor_bridge.build_tutor_graph",
            side_effect=lambda config: config,
        ) as build_graph:
            selected = bridge._graph_for_model("qwen3.5:9b-q4_K_M")
            repeated = bridge._graph_for_model("qwen3.5:9b-q4_K_M")

        self.assertIs(selected, repeated)
        self.assertEqual(selected.model_name, "qwen3.5:9b-q4_K_M")
        self.assertEqual(selected.classifier_model_name, original.classifier_model_name)
        self.assertEqual(selected.turn_relation_model_name, original.turn_relation_model_name)
        self.assertEqual(selected.followup_model_name, original.followup_model_name)
        self.assertEqual(selected.safety_embedding_model_name, original.safety_embedding_model_name)
        self.assertEqual(original.model_name, "phi4-mini:latest")
        build_graph.assert_called_once()


if __name__ == "__main__":
    unittest.main()
