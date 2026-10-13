"""Offline Phase 2 tests: no live AvalAI calls or credentials needed."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent import KnowledgeStore, MathTutorAgent


class FakeChatModel:
    def __init__(self, concepts=None):
        self.concepts = [] if concepts is None else concepts
        self.answer_calls = 0
        self.prompts = []
        self.diagnostic_calls = 0
        self.grading_calls = 0

    def invoke(self, messages):
        system = messages[0].content
        user = messages[-1].content
        self.prompts.append((system, user))
        if system.startswith("You analyze requests"):
            text = user.lower()
            if "حل کن" in text:
                intent = "solve"
            elif "توضیح بده" in text:
                intent = "explain"
            elif "ترجمه کن" in text:
                intent = "translate"
            else:
                intent = "clarify"
            return SimpleNamespace(content=json.dumps({
                "intent": intent,
                "math_topic": "test_topic",
                "explanation_depth": "technical" if "فنی" in text else "simple",
            }))
        if system.startswith("Extract the FEW"):
            return SimpleNamespace(content=json.dumps({"concept_ids": self.concepts}))
        if system.startswith("Create ONE simple"):
            self.diagnostic_calls += 1
            return SimpleNamespace(content=json.dumps({
                "question": "اگر 1 + 1 = 2 باشد، پاسخ چیست؟",
                "expected_answer": "2",
            }))
        if system.startswith("Grade the learner's answer"):
            self.grading_calls += 1
            return SimpleNamespace(content=json.dumps({
                "is_correct": "Learner response: 2" in user
            }))
        self.answer_calls += 1
        return SimpleNamespace(content="روش حل، مثال ساده، و پاسخ آزمایشی.")


class MathTutorGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "knowledge.json"

    def make_agent(self, concepts=None):
        model = FakeChatModel(concepts)
        return MathTutorAgent(model=model, knowledge_path=self.path), model

    def finish_question(self, agent, question, answers):
        thread_id, result = agent.start_question(question)
        i = 0
        while "__interrupt__" in result:
            self.assertLess(i, len(answers), "Missing diagnostic answer")
            self.assertTrue(result["__interrupt__"][0].value["question"])
            result = agent.resume_question(thread_id, answers[i])
            i += 1
        return result, i

    def test_routing_unchanged(self):
        agent, model = self.make_agent()
        self.assertEqual(agent.ask("x^2 = 4 را حل کن")["intent"], "solve")
        self.assertEqual(agent.ask("مشتق چیست؟ فنی توضیح بده")["explanation_depth"], "technical")
        self.assertEqual(agent.ask("integral را ترجمه کن")["intent"], "translate")
        self.assertEqual(model.answer_calls, 3)

    def test_clarify_does_not_assess(self):
        agent, model = self.make_agent(["algebra.factoring"])
        result = agent.ask("سلام")
        self.assertEqual(result["intent"], "clarify")
        self.assertEqual(model.diagnostic_calls, 0)
        self.assertEqual(model.answer_calls, 0)

    def test_correct_diagnostic_persists(self):
        agent, model = self.make_agent(["algebra.factoring"])
        result, n = self.finish_question(agent, "معادله را حل کن", ["2"])
        self.assertEqual(n, 1)
        self.assertEqual(result["assessment_count"], 1)
        self.assertEqual(result["prerequisites"][0]["status"], "known")
        self.assertEqual(json.loads(self.path.read_text())["known_concepts"], ["algebra.factoring"])
        self.assertEqual(model.grading_calls, 1)

    def test_failed_diagnostic_teaches_without_persisting(self):
        agent, model = self.make_agent(["algebra.zero_product_rule"])
        result, n = self.finish_question(agent, "معادله را حل کن", ["غلط"])
        self.assertEqual(n, 1)
        self.assertFalse(self.path.exists())
        self.assertEqual(result["prerequisites"][0]["status"], "needs_teaching")
        self.assertTrue(result["prerequisites"][0]["needs_teaching"])
        answer_system = model.prompts[-1][0]
        self.assertIn("Zero-product rule", answer_system)
        self.assertIn("Needs teaching:", answer_system)

    def test_skip_does_not_call_grader(self):
        agent, model = self.make_agent(["algebra.factoring"])
        result, _ = self.finish_question(agent, "معادله را حل کن", ["/skip"])
        self.assertEqual(result["prerequisites"][0]["assessment_result"], "failed")
        self.assertEqual(model.grading_calls, 0)
        self.assertEqual(agent.known_concepts(), [])

    def test_known_concept_is_not_retested(self):
        store = KnowledgeStore(self.path)
        store.mark_known("algebra.factoring")
        agent, model = self.make_agent(["algebra.factoring"])
        result = agent.ask("x^2 را حل کن")
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(model.diagnostic_calls, 0)
        self.assertEqual(result["prerequisites"][0]["status"], "known")

    def test_known_concept_survives_restarting_agent(self):
        one, _ = self.make_agent(["algebra.factoring"])
        self.finish_question(one, "معادله را حل کن", ["2"])
        two, model = self.make_agent(["algebra.factoring"])
        result = two.ask("معادله را حل کن")
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(model.diagnostic_calls, 0)

    def test_limit_two_diagnostics(self):
        ids = ["algebra.factoring", "algebra.polynomials", "algebra.zero_product_rule"]
        agent, model = self.make_agent(ids)
        result, n = self.finish_question(agent, "معادله را حل کن", ["/skip", "/skip"])
        self.assertEqual(n, 2)
        self.assertEqual(model.diagnostic_calls, 2)
        self.assertEqual(result["assessment_count"], 2)
        self.assertEqual(result["prerequisites"][2]["status"], "unchecked")

    def test_invalid_concept_ids_are_filtered(self):
        agent, model = self.make_agent(["unknown.made_up", "algebra.factoring", "algebra.factoring"])
        result, n = self.finish_question(agent, "معادله را حل کن", ["/skip"])
        self.assertEqual(n, 1)
        self.assertEqual(len(result["prerequisites"]), 1)
        self.assertEqual(model.diagnostic_calls, 1)

    def test_corrupted_json_is_not_overwritten(self):
        self.path.write_text("{broken json", encoding="utf-8")
        store = KnowledgeStore(self.path)
        with self.assertRaises(RuntimeError):
            store.mark_known("algebra.factoring")
        self.assertEqual(self.path.read_text(), "{broken json")

    def test_invalid_schema_is_rejected(self):
        self.path.write_text('{"known_concepts": "string"}', encoding="utf-8")
        with self.assertRaises(RuntimeError):
            KnowledgeStore(self.path).load()

    def test_separate_question_state(self):
        agent, _ = self.make_agent()
        first = agent.ask("معادله را حل کن")
        first["prerequisites"].append({"concept_id": "algebra.factoring"})
        second = agent.ask("مشتق را توضیح بده")
        self.assertEqual(second["prerequisites"], [])
        self.assertEqual(second["assessment_count"], 0)

    def test_resume_rejects_unknown_thread(self):
        agent, _ = self.make_agent()
        with self.assertRaisesRegex(ValueError, "No pending diagnostic"):
            agent.resume_question("invalid", "2")

    def test_avalai_gateway_settings_are_preserved(self):
        fake = FakeChatModel()
        settings = {
            "AVALAI_API_KEY": "unit-test-key",
            "AVALAI_BASE_URL": "https://api.avalai.ir/v1",
            "LLM_MODEL": "kimi-k2.6",
        }
        with patch.dict("os.environ", settings):
            with patch("agent.ChatOpenAI", return_value=fake) as client:
                agent = MathTutorAgent(knowledge_path=self.path)
                self.assertIsNotNone(agent.graph)
                client.assert_called_once_with(
                    model="kimi-k2.6",
                    api_key="unit-test-key",
                    base_url="https://api.avalai.ir/v1",
                    timeout=90,
                    max_retries=2,
                )

    def test_invalid_router_json_fails_explicitly(self):
        class InvalidRouter(FakeChatModel):
            def invoke(self, messages):
                if messages[0].content.startswith("You analyze requests"):
                    return SimpleNamespace(content="not-json")
                return super().invoke(messages)
        agent = MathTutorAgent(model=InvalidRouter(), knowledge_path=self.path)
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            agent.ask("معادله را حل کن")

    def test_reject_empty_question(self):
        agent, _ = self.make_agent()
        with self.assertRaises(ValueError):
            agent.ask("  ")


if __name__ == "__main__":
    unittest.main()
