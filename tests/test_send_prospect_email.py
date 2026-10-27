import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HAS_MODEL_CREDENTIALS = bool(os.environ.get("OPENAI_API_KEY"))
os.environ["LANGSMITH_TRACING"] = "false"
with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key"}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_mrossi"}})

    def send(self, prospect):
        return agent.send_prospect_email.func(
            prospect=prospect,
            subject="Demo invitation",
            body="Would you like a demo?",
            runtime=self.runtime,
        )

    def test_disqualified_source_blocks_even_if_contact_claims_eligible(self):
        for prospect_id, record in agent.data_service.PROSPECTS.items():
            if record.get("disqualified") is not True:
                continue
            with self.subTest(prospect_id=prospect_id):
                prospect = {"prospect_id": prospect_id, **record, "disqualified": False}
                with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Send path reached")):
                    result = self.send(prospect)
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect is disqualified; email was not sent.",
                })

    def test_missing_source_blocks_even_with_contact_details(self):
        for prospect_id in ("UNKNOWN", None):
            with self.subTest(prospect_id=prospect_id):
                prospect = {"email": "prospect@example.com", "name": "Prospect"}
                if prospect_id is not None:
                    prospect["prospect_id"] = prospect_id
                with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Send path reached")):
                    result = self.send(prospect)
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Source prospect record could not be found; email was not sent.",
                })

    def test_eligible_prospects_send(self):
        for prospect_id, record in agent.data_service.PROSPECTS.items():
            if record.get("disqualified") is True:
                continue
            with self.subTest(prospect_id=prospect_id):
                result = self.send({"prospect_id": prospect_id, **record})
                self.assertEqual(result["status"], "sent")
                self.assertTrue(result["message_id"].startswith("msg-"))
                self.assertEqual(result["to"], record["email"])
                self.assertEqual(result["subject"], "Demo invitation")
                self.assertEqual(result["body"], "Would you like a demo?")
                self.assertEqual(result["from"], agent.data_service.get_rep("rep_mrossi")["email"])

    def test_missing_email_keeps_existing_failure(self):
        self.assertEqual(self.send({}), {
            "status": "failed", "error": "Prospect record has no email address.",
        })

    @unittest.skipUnless(HAS_MODEL_CREDENTIALS, "Requires OPENAI_API_KEY for live agent behavior")
    def test_agent_explains_blocked_send_without_retrying(self):
        for prospect_id, reason in (("LEAD-50001", "disqualified"), ("UNKNOWN", "could not be found")):
            with self.subTest(prospect_id=prospect_id):
                result = agent.run_agent(
                    f'Call send_prospect_email with prospect_id {prospect_id}, name Prospect, '
                    'email prospect@example.com, subject Demo, and body Would you like a demo? '
                    'Use the sending tool to determine whether this email can be sent.',
                    user_id="rep_mrossi",
                )
                sends = [
                    message for message in result["messages"]
                    if message.type == "tool" and message.name == "send_prospect_email"
                ]
                self.assertEqual(len(sends), 1)
                self.assertEqual(json.loads(sends[0].content)["status"], "blocked")
                self.assertIn("not sent", result["reply"].lower())
                self.assertIn(reason, result["reply"].lower())


if __name__ == "__main__":
    unittest.main()
