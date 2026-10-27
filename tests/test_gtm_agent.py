import importlib
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class AgentTests(unittest.TestCase):
    def test_rep_and_email_tools_receive_runtime_metadata(self):
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "test-key",
            "LANGSMITH_TRACING": "false",
        }):
            agent = importlib.import_module("gtm_agent.gtm_agent")
            rep = agent.data_service.REP_IDS[0]
            prospect_id = next(iter(agent.data_service.PROSPECTS))
            prospect = agent.data_service.get_prospect_record(prospect_id)
            responses = [
                AIMessage(content="", tool_calls=[{
                    "name": "get_current_rep", "args": {}, "id": "rep",
                }]),
                AIMessage(content="", tool_calls=[{
                    "name": "get_prospect",
                    "args": {"prospect_id": prospect_id}, "id": "prospect",
                }]),
                AIMessage(content="", tool_calls=[{
                    "name": "send_prospect_email",
                    "args": {
                        "prospect": prospect,
                        "subject": "Hello", "body": "Let's connect.",
                    },
                    "id": "email",
                }]),
                AIMessage(content="Email sent."),
            ]
            calls = []

            def generate(messages, **kwargs):
                calls.append((messages, kwargs))
                return ChatResult(generations=[ChatGeneration(message=responses.pop(0))])

            with patch.object(ChatOpenAI, "_generate", side_effect=generate):
                result = agent.run_agent(
                    "Email this prospect", user_id=rep["rep_id"], thread_id="test-thread",
                )

        self.assertEqual(result["reply"], "Email sent.")
        results = {
            message.name: json.loads(message.content)
            for message in result["messages"] if isinstance(message, ToolMessage)
        }
        self.assertEqual(results["get_current_rep"], {"rep": rep, "found": True})
        email = results["send_prospect_email"]
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["from"], rep["email"])
        self.assertEqual(email["to"], prospect["email"])
        self.assertEqual(email["subject"], "Hello")
        self.assertEqual(email["body"], "Let's connect.")
        self.assertEqual(len(calls), 4)
        for messages, kwargs in calls:
            self.assertEqual(
                [m.content for m in messages if isinstance(m, SystemMessage)],
                [agent.SYSTEM_PROMPT],
            )
            schemas = {tool["function"]["name"]: tool["function"] for tool in kwargs["tools"]}
            self.assertEqual(set(schemas), {
                "lookup_offering", "build_prospect_profile", "get_prospect",
                "get_current_rep", "send_prospect_email", "score_prospect",
                "update_prospect_info",
            })
            for schema in schemas.values():
                self.assertNotIn("runtime", schema["parameters"]["properties"])


if __name__ == "__main__":
    unittest.main()
