import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent.gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def test_plain_text_email(self):
        body = "Are you available for a pricing review next week?"
        result = send_prospect_email.func(
            prospect={"name": "Prospect", "email": "prospect@example.com"},
            subject="Pricing review",
            body=body,
            runtime=SimpleNamespace(config={}),
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "prospect@example.com")
        self.assertEqual(result["body"], body)


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Requires OPENAI_API_KEY")
class AgentEmailTests(unittest.TestCase):
    def run_request(self, request):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            return run_agent(request, user_id="rep_sbrown")

    def test_document_requests_disclose_attachment_limitation(self):
        for document in ("pricing deck", "PDF", "contract"):
            with self.subTest(document=document):
                result = self.run_request(
                    f"Email LEAD-50003 to send over our {document}."
                )
                reply = result["reply"].lower()
                self.assertRegex(
                    reply,
                    r"(?:cannot|can't|unable|not supported|don't support).*attach"
                    r"|attach.*(?:cannot|can't|unable|not supported)"
                    r"|(?:shareable|provide|share|send).*link",
                )
                texts = [reply]
                for message in result["messages"]:
                    if message.type == "tool" and message.name == "send_prospect_email":
                        email = json.loads(message.content)
                        if email.get("status") == "sent":
                            texts.append(email["body"].lower())
                for text in texts:
                    self.assertNotRegex(
                        text,
                        r"(?:i've|i have|we've|we have) attached"
                        r"|please find.*attached"
                        r"|attached (?:the|our|your|a) "
                        r"|(?:is|was|has been) attached"
                        r"|sent (?:the|our|your|a) (?:pricing deck|pdf|contract)"
                        r"|(?:deck|pdf|contract) (?:was|has been) sent",
                    )

    def test_plain_text_request_sends_email(self):
        result = self.run_request(
            "Email LEAD-50003 asking about availability for a pricing review next week."
        )
        emails = [
            json.loads(message.content)
            for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]
        self.assertTrue(any(email.get("status") == "sent" for email in emails))
        self.assertTrue(result["reply"])
