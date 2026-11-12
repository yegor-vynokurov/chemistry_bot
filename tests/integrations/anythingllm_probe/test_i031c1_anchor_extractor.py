from __future__ import annotations

import unittest

from chemistry_bot.integrations.anythingllm_probe.rag_relevance_gate import (
    ChunkAnchorMatch,
    QueryAnchor,
    extract_query_anchors,
    match_chunk_anchors,
    normalize_chem_term,
)


def _kinds(anchors: tuple[QueryAnchor, ...]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for anchor in anchors:
        result.setdefault(anchor.family, set()).add(anchor.kind)
    return result


class AnchorExtractorTests(unittest.TestCase):
    def test_normalize_chem_term_handles_case_subscripts_and_punctuation(self) -> None:
        self.assertEqual(normalize_chem_term("KMnO₄⁻"), "kmno4")
        self.assertEqual(normalize_chem_term("self-indicator"), "self indicator")
        self.assertEqual(normalize_chem_term("Sodium   Thiosulfate"), "sodium thiosulfate")

    def test_extract_query_anchors_normalizes_thiosulphate_variants(self) -> None:
        anchors = extract_query_anchors("Why do we need tiosulphate for titration?")
        by_family = _kinds(anchors)
        self.assertIn("thiosulphate", by_family)
        self.assertIn("target_anchor", by_family["thiosulphate"])
        self.assertIn("titration", by_family)
        self.assertIn("generic_topic_anchor", by_family["titration"])

    def test_extract_query_anchors_normalizes_us_spelling(self) -> None:
        anchors = extract_query_anchors("What does sodium thiosulfate do in iodometry?")
        by_family = _kinds(anchors)
        self.assertIn("thiosulphate", by_family)
        self.assertIn("target_anchor", by_family["thiosulphate"])
        self.assertIn("supporting_anchor", by_family["thiosulphate"])

    def test_extract_query_anchors_normalizes_permanganate_formula(self) -> None:
        anchors = extract_query_anchors("Why is KMnO4 useful in titration?")
        by_family = _kinds(anchors)
        self.assertIn("permanganate", by_family)
        self.assertIn("target_anchor", by_family["permanganate"])

    def test_titration_alone_is_generic_topic_only(self) -> None:
        anchors = extract_query_anchors("What is titration?")
        self.assertEqual(tuple(anchor.kind for anchor in anchors), ("generic_topic_anchor",))
        self.assertEqual(anchors[0].canonical, "titration")

    def test_chunk_with_only_titration_is_generic_only_for_thiosulphate_question(self) -> None:
        anchors = extract_query_anchors("Why do we need thiosulphate for titration?")
        match = match_chunk_anchors(
            "A titration should have a clear endpoint and a rapid reaction.",
            anchors,
        )
        self.assertIsInstance(match, ChunkAnchorMatch)
        self.assertTrue(match.anchor_required)
        self.assertFalse(match.target_anchor_matched)
        self.assertFalse(match.supporting_anchor_matched)
        self.assertTrue(match.generic_topic_matched)
        self.assertTrue(match.generic_only)
        self.assertIn("thiosulphate", match.missing_required_anchors)

    def test_chunk_with_iodometric_context_counts_as_supporting_thiosulphate_evidence(self) -> None:
        anchors = extract_query_anchors("Why do we need thiosulphate for titration?")
        match = match_chunk_anchors(
            "In iodometric analysis iodine is reduced by the titrant at the endpoint.",
            anchors,
        )
        self.assertTrue(match.anchor_required)
        self.assertFalse(match.target_anchor_matched)
        self.assertTrue(match.supporting_anchor_matched)
        self.assertFalse(match.generic_only)
        self.assertNotIn("thiosulphate", match.missing_required_anchors)
        self.assertIn("supporting:thiosulphate", match.matched_anchors)

    def test_chunk_with_self_indicator_and_pink_endpoint_supports_permanganate(self) -> None:
        anchors = extract_query_anchors("Why is potassium permanganate useful in titration?")
        match = match_chunk_anchors(
            "Potassium permanganate acts as a self-indicator because a faint pink endpoint remains.",
            anchors,
        )
        self.assertTrue(match.target_anchor_matched)
        self.assertTrue(match.supporting_anchor_matched)
        self.assertFalse(match.generic_only)
        self.assertIn("target:permanganate", match.matched_anchors)
        self.assertIn("supporting:permanganate", match.matched_anchors)

    def test_formula_variants_match_permanganate_chunks(self) -> None:
        anchors = extract_query_anchors("Why use KMnO4 in redox titration?")
        match = match_chunk_anchors("The MnO₄⁻ ion is reduced to Mn2+ in acidic solution.", anchors)
        self.assertTrue(match.target_anchor_matched)
        self.assertIn("target:permanganate", match.matched_anchors)

    def test_atom_ion_anchors_are_word_bound_aware(self) -> None:
        anchors = extract_query_anchors("What is an ion?")
        self.assertIn("atom_ion", _kinds(anchors))
        # "solution" contains the letters i-o-n but should not match the ion anchor.
        no_match = match_chunk_anchors("This solution is used in titration.", anchors)
        self.assertFalse(no_match.target_anchor_matched)
        self.assertIn("atom_ion", no_match.missing_required_anchors)
        yes_match = match_chunk_anchors("An ion is an atom or molecule with a net charge.", anchors)
        self.assertTrue(yes_match.target_anchor_matched)
        self.assertTrue(yes_match.supporting_anchor_matched)

    def test_as_dict_shapes_are_stable_for_logs(self) -> None:
        anchors = extract_query_anchors("Why do we need thiosulfate for titration?")
        anchor_payload = anchors[0].as_dict()
        self.assertIn("canonical", anchor_payload)
        self.assertIn("matched_texts", anchor_payload)
        match_payload = match_chunk_anchors("iodometric titration with iodine", anchors).as_dict()
        self.assertIn("matched_anchors", match_payload)
        self.assertIn("missing_required_anchors", match_payload)


if __name__ == "__main__":
    unittest.main()
