from __future__ import annotations

import importlib.util
import unittest


_HAS_BRIDGE_RUNTIME = all(
    importlib.util.find_spec(name) is not None
    for name in ("fastapi", "langchain_core", "langchain_ollama", "langgraph")
)


@unittest.skipUnless(
    _HAS_BRIDGE_RUNTIME,
    "Full I0.3 API tests require the tutor graph runtime dependencies.",
)
class AnythingLLMLangGraphBridgeImportTests(unittest.TestCase):
    def test_app_module_imports_with_full_runtime(self) -> None:
        from chemistry_bot.integrations.anythingllm_probe.app import (
            DEFAULT_MODEL_ID,
            ProbeSettings,
        )

        settings = ProbeSettings()
        self.assertEqual(DEFAULT_MODEL_ID, "chemistry-tutor")
        self.assertEqual(settings.model_id, "chemistry-tutor")
        self.assertTrue(settings.resolve_conversations)
        self.assertEqual(settings.user_document_policy, "classifier")


if __name__ == "__main__":
    unittest.main()
