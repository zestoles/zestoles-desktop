"""What JARVIS is allowed to know about a person.

This is the layer where being wrong is least visible and least forgivable: a
made-up fact about someone is not an error they can see, it is something the
machine acts on quietly for months. So the rules are structural and tested as
rules, not as prompt wording.

Three of them carry the weight:

- **inference is never durable.** "Sen stresli birisin" is a guess however
  fluently a model produced it, and no amount of confidence promotes it.
- **consent is required, and it means they said yes.** Not that it seemed
  useful, not that they mentioned it -- yes.
- **forgetting deletes.** No tombstone, no hidden flag, no archive the model can
  still read.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.memory.profile import (  # noqa: E402
    Confidence,
    Consent,
    Preference,
    Profile,
    Source,
    accept,
)


class TestTheGate(unittest.TestCase):
    """`accept` is pure, so the rule can be tested without a database."""

    def item(self, **kwargs):
        base = {"subject": "çalışma saati", "detail": "geceleri çalışıyor",
                "source": Source.USER, "confidence": Confidence.HIGH,
                "consent": Consent.GRANTED}
        base.update(kwargs)
        return Preference(**base)

    def test_a_stated_preference_with_consent_is_accepted(self):
        ok, why = accept(self.item())
        self.assertTrue(ok, why)

    def test_an_inference_is_never_durable(self):
        """The rule this whole module exists for."""
        ok, why = accept(self.item(source=Source.INFERRED))
        self.assertFalse(ok)
        self.assertIn("çıkarım", why)

    def test_an_inference_is_refused_even_with_consent_and_high_confidence(self):
        ok, _why = accept(self.item(source=Source.INFERRED,
                                    consent=Consent.GRANTED,
                                    confidence=Confidence.HIGH))
        self.assertFalse(ok, "izin bir çıkarımı olguya çevirmez")

    def test_without_consent_nothing_is_stored(self):
        for consent in (Consent.UNKNOWN, Consent.REFUSED):
            with self.subTest(consent=consent):
                ok, why = accept(self.item(consent=consent))
                self.assertFalse(ok)
                self.assertIn("onaylamadı", why)

    def test_empty_fields_are_refused(self):
        self.assertFalse(accept(self.item(subject="  "))[0])
        self.assertFalse(accept(self.item(detail=""))[0])

    def test_unknown_vocabulary_is_refused(self):
        self.assertFalse(accept(self.item(source="uydurma"))[0])
        self.assertFalse(accept(self.item(confidence="epey"))[0])
        self.assertFalse(accept(self.item(consent="belki"))[0])

    def test_a_tool_measurement_may_be_kept_with_consent(self):
        ok, why = accept(self.item(source=Source.TOOL))
        self.assertTrue(ok, why)


class ProfileCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.profile = Profile(Path(self._tmp.name) / "profil.db")

    def tearDown(self):
        self._tmp.cleanup()

    def remember(self, subject="çalışma saati", detail="geceleri çalışıyor",
                 **kwargs):
        kwargs.setdefault("consent", Consent.GRANTED)
        return self.profile.remember(subject, detail, **kwargs)


class TestRemembering(ProfileCase):
    def test_it_survives_a_new_profile_object(self):
        """The point of durable memory: a new session still knows."""
        self.remember()
        again = Profile(self.profile.db_path)
        self.assertEqual(len(again.facts()), 1)
        self.assertIn("geceleri", again.facts()[0].detail)

    def test_refused_items_are_not_stored(self):
        item, why = self.profile.remember("ruh hali", "stresli biri",
                                          source=Source.INFERRED,
                                          consent=Consent.GRANTED)
        self.assertIsNone(item)
        self.assertTrue(why)
        self.assertEqual(self.profile.count(), 0)

    def test_the_same_subject_updates_rather_than_duplicating(self):
        self.remember(detail="geceleri çalışıyor")
        self.remember(detail="artık sabahları çalışıyor")
        items = self.profile.facts()
        self.assertEqual(len(items), 1)
        self.assertIn("sabahları", items[0].detail)

    def test_long_input_is_trimmed_rather_than_refused(self):
        item, _why = self.remember(subject="k" * 300, detail="d" * 2000)
        self.assertIsNotNone(item)
        self.assertLessEqual(len(item.subject), 60)
        self.assertLessEqual(len(item.detail), 400)

    def test_whitespace_is_normalised_so_forgetting_can_find_it(self):
        self.remember(subject="  çalışma   saati \n")
        self.assertEqual(self.profile.facts()[0].subject, "çalışma saati")


class TestRecall(ProfileCase):
    def test_nothing_known_is_an_empty_list_not_an_error(self):
        self.assertEqual(self.profile.facts(), [])
        self.assertEqual(self.profile.summary(), "")

    def test_every_line_carries_where_it_came_from(self):
        """A preference repeated back without its provenance is indistinguishable
        from something the user just said."""
        self.remember()
        line = self.profile.facts()[0].as_line()
        self.assertIn("senin söylediğin", line)

    def test_a_tool_measurement_says_it_was_measured(self):
        self.remember(subject="python sürümü", detail="3.14.6",
                      source=Source.TOOL)
        self.assertIn("ölçülen", self.profile.facts()[0].as_line())

    def test_the_summary_tells_the_model_these_are_not_new_words(self):
        self.remember()
        summary = self.profile.summary()
        self.assertIn("şimdi söylediği değil", summary)

    def test_the_newest_comes_first(self):
        self.remember(subject="birinci", detail="a")
        self.remember(subject="ikinci", detail="b")
        self.assertEqual(self.profile.facts()[0].subject, "ikinci")


class TestForgetting(ProfileCase):
    def test_forgetting_removes_it(self):
        self.remember()
        self.assertEqual(self.profile.forget("çalışma saati"), 1)
        self.assertEqual(self.profile.facts(), [])

    def test_it_is_gone_from_storage_not_merely_hidden(self):
        """No tombstone, no archive: "unut" has to mean gone."""
        self.remember()
        self.profile.forget("çalışma saati")
        self.assertEqual(self.profile.count(), 0)
        self.assertEqual(len(Profile(self.profile.db_path).recall()), 0)

    def test_a_partial_name_finds_it(self):
        """The user says "çalışma saatimi unut", not the exact stored subject."""
        self.remember(subject="çalışma saati")
        self.assertEqual(self.profile.forget("çalışma"), 1)

    def test_forgetting_something_unknown_removes_nothing(self):
        self.remember()
        self.assertEqual(self.profile.forget("hiç kaydedilmemiş şey"), 0)
        self.assertEqual(self.profile.count(), 1)

    def test_forgetting_nothing_is_not_a_crash(self):
        self.assertEqual(self.profile.forget(""), 0)
        self.assertEqual(self.profile.forget("   "), 0)


class TestTheTools(unittest.TestCase):
    """The model reaches this through tools, so the gate has to hold there too."""

    def setUp(self):
        from jarvis.tools import hafiza

        self._tmp = tempfile.TemporaryDirectory()
        self.profile = Profile(Path(self._tmp.name) / "profil.db")
        hafiza.provide_profile(self.profile)
        self.hafiza = hafiza

    def tearDown(self):
        self.hafiza.provide_profile(None)
        self._tmp.cleanup()

    def run_tool(self, name, **arguments):
        from jarvis import tools

        return tools.run(name, workspace=None, confirmed=False, **arguments)

    def test_remembering_without_consent_fails_and_says_to_ask(self):
        result = self.run_tool("memory.remember", konu="çalışma saati",
                               ayrinti="geceleri", onay=False)
        self.assertFalse(result.ok)
        self.assertIn("onaylamadı", result.error)
        self.assertEqual(self.profile.count(), 0)

    def test_remembering_with_consent_works(self):
        result = self.run_tool("memory.remember", konu="çalışma saati",
                               ayrinti="geceleri çalışıyor", onay=True)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(self.profile.count(), 1)

    def test_the_model_cannot_store_its_own_conclusion(self):
        result = self.run_tool("memory.remember", konu="kişilik",
                               ayrinti="stresli biri", onay=True,
                               kaynak="cikarim")
        self.assertFalse(result.ok)
        self.assertEqual(self.profile.count(), 0)

    def test_recall_lists_what_is_known(self):
        self.run_tool("memory.remember", konu="dil", ayrinti="Türkçe konuşur",
                      onay=True)
        result = self.run_tool("memory.recall")
        self.assertTrue(result.ok)
        self.assertIn("Türkçe", result.output)

    def test_recall_with_nothing_known_says_so(self):
        result = self.run_tool("memory.recall")
        self.assertTrue(result.ok)
        self.assertIn("kayıtlı bir şey yok", result.output)

    def test_forget_removes_it(self):
        self.run_tool("memory.remember", konu="dil", ayrinti="Türkçe", onay=True)
        result = self.run_tool("memory.forget", konu="dil")
        self.assertTrue(result.ok)
        self.assertEqual(self.profile.count(), 0)

    def test_forgetting_something_unknown_is_an_honest_failure(self):
        result = self.run_tool("memory.forget", konu="olmayan")
        self.assertFalse(result.ok)
        self.assertIn("yok", result.error)

    def test_the_memory_tools_are_low_risk(self):
        """Putting a confirmation card in front of writing a preference would
        train the user to click through confirmations, which is how the ones
        that matter stop being read."""
        from jarvis import tools

        for name in ("memory.remember", "memory.recall", "memory.forget"):
            with self.subTest(name=name):
                self.assertEqual(tools.get(name).risk, "low")

    def test_without_a_profile_the_tools_say_so(self):
        self.hafiza.provide_profile(None)
        result = self.run_tool("memory.recall")
        self.assertFalse(result.ok)
        self.assertIn("açık değil", result.error)


if __name__ == "__main__":
    unittest.main()
