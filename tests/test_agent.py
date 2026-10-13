"""Offline graph tests: no API key, network, or billing required."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent import MathTutorAgent


class FakeChatModel:
    def __init__(self):
        self.answer_calls = 0

    def invoke(self, messages):
        question = messages[-1].content.lower()
        is_router = "Classify the user's CURRENT message" in messages[0].content
        if is_router:
            if "حل کن" in question:
                intent = "solve"
            elif "توضیح بده" in question:
                intent = "explain"
            elif "ترجمه کن" in question:
                intent = "translate"
            else:
                intent = "clarify"
            depth = "technical" if "فنی" in question else "simple"
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "intent": intent,
                        "math_topic": "test_topic",
                        "explanation_depth": depth,
                    }
                )
            )
        self.answer_calls += 1
        return SimpleNamespace(content="روش حل، مثال ساده، و پاسخ آزمایشی.")


class MathTutorGraphTests(unittest.TestCase):
    def setUp(self):
        self.fake_model = FakeChatModel()
        self.agent = MathTutorAgent(model=self.fake_model)

    def test_solver_route(self):
        result = self.agent.ask("x^2 - 5x + 6 = 0 را حل کن")
        self.assertEqual(result["intent"], "solve")
        self.assertEqual(result["verification_status"], "not_verified")
        self.assertIsNone(result["sympy_result"])
        self.assertIn("پاسخ آزمایشی", result["final_answer"])
        self.assertEqual(self.fake_model.answer_calls, 1)

    def test_explainer_route_and_technical_depth(self):
        result = self.agent.ask("مشتق چیست؟ فنی توضیح بده")
        self.assertEqual(result["intent"], "explain")
        self.assertEqual(result["explanation_depth"], "technical")

    def test_translator_route(self):
        result = self.agent.ask("integral را ترجمه کن")
        self.assertEqual(result["intent"], "translate")
        self.assertTrue(result["final_answer"])

    def test_clarify_route_does_not_call_answer_model(self):
        result = self.agent.ask("سلام")
        self.assertEqual(result["intent"], "clarify")
        self.assertIn("دقیق‌تر", result["final_answer"])
        self.assertEqual(self.fake_model.answer_calls, 0)

    def test_new_question_starts_with_fresh_question_state(self):
        first = self.agent.ask("x = 2 را حل کن")
        first["prerequisites"].append(
            {
                "concept_id": "test.concept",
                "name": "Test",
                "status": "known",
                "assessment_result": "passed",
                "needs_teaching": False,
            }
        )
        second = self.agent.ask("مشتق را توضیح بده")
        self.assertEqual(second["prerequisites"], [])
        self.assertEqual(second["assessment_count"], 0)

    def test_reject_empty_input(self):
        with self.assertRaises(ValueError):
            self.agent.ask("   ")

    def test_avalai_client_uses_configured_endpoint_and_model(self):
        settings = {
            "AVALAI_API_KEY": "unit-test-key",
            "AVALAI_BASE_URL": "https://api.avalai.ir/v1",
            "LLM_MODEL": "kimi-k2.6",
        }
        with patch.dict("os.environ", settings):
            with patch("agent.ChatOpenAI", return_value=self.fake_model) as client:
                initialized_agent = MathTutorAgent()
                self.assertIsNotNone(initialized_agent.graph)
                client.assert_called_once_with(
                    model="kimi-k2.6",
                    api_key="unit-test-key",
                    base_url="https://api.avalai.ir/v1",
                    timeout=90,
                    max_retries=2,
                )

    def test_router_rejects_invalid_json(self):
        class InvalidJSONModel(FakeChatModel):
            def invoke(self, messages):
                if "Classify the user's CURRENT message" in messages[0].content:
                    return SimpleNamespace(content="not-json")
                return super().invoke(messages)

        agent = MathTutorAgent(model=InvalidJSONModel())
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            agent.ask("معادله را حل کن")


if __name__ == "__main__":
    unittest.main()
