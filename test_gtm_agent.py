import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class GTMAgentTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = next(iter(agent.data_service.OFFERINGS))

    def test_run_agent_includes_signed_in_rep_without_tool_round_trip(self):
        rep = agent.REP_IDS[-1]
        with patch.object(agent.gtm_agent, "invoke") as invoke:
            invoke.return_value = {"messages": [SimpleNamespace(content="Done")]}
            result = agent.run_agent("Email the prospect", user_id=rep["rep_id"])
        messages = invoke.call_args.args[0]["messages"]
        context = next(message["content"] for message in messages if message["role"] == "system")
        self.assertIn(rep["name"], context)
        self.assertIn(rep["email"], context)
        self.assertEqual(invoke.call_args.kwargs["config"]["metadata"]["user_id"], rep["rep_id"])
        self.assertEqual(result["reply"], "Done")

    def test_unknown_rep_does_not_invent_identity(self):
        with patch.object(agent.gtm_agent, "invoke") as invoke:
            invoke.return_value = {"messages": [SimpleNamespace(content="Done")]}
            agent.run_agent("Look up an offering", user_id="unknown")
        self.assertEqual(invoke.call_args.args[0]["messages"][0]["content"], "Signed-in rep: not found.")

    def test_score_loads_records_from_ids(self):
        expected = agent.ProspectScore(
            score=75, justification="A promising fit with missing technologies.",
            rubric_breakdown=agent.RubricBreakdown(revenue_fit=90, tech_stack_match=60, segment_fit=75),
        )
        with patch.object(agent, "_scoring_llm") as scorer:
            scorer.invoke.return_value = expected
            result = agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": self.offering_id})
        self.assertEqual(result, expected.model_dump())
        content = scorer.invoke.call_args.args[0][1]["content"]
        offering, profile = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering), agent.data_service.get_offering(self.offering_id))
        self.assertEqual(json.loads(profile), agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"])

    def test_missing_prospect(self):
        with patch.object(agent, "_scoring_llm") as scorer:
            result = agent.score_prospect.invoke({"prospect_id": "missing", "offering_id": self.offering_id})
        self.assertEqual(result, {"score": None, "error": "Cannot score without a valid prospect profile."})
        scorer.invoke.assert_not_called()
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {"prospect_profile": None, "found": False})

    def test_missing_or_invalid_offering(self):
        offering = agent.data_service.get_offering(self.offering_id)
        for record in (None, {}, {**offering, "required_tech_stack": []},
                       {**offering, "min_annual_revenue": None}, {**offering, "description": ""}):
            with self.subTest(record=record):
                with patch.object(agent.data_service, "get_offering", return_value=record), patch.object(agent, "_scoring_llm") as scorer:
                    result = agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": "missing"})
                self.assertEqual(result, {"score": None, "error": "Cannot score without a valid offering."})
                scorer.invoke.assert_not_called()

    def test_scoring_rereads_tech_stack_for_cached_profile(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"]
        stale = {**profile, "tech_stack": ["stale"]}
        agent.data_service.save_profile_to_db(self.prospect_id, stale)
        with patch.object(agent, "_scoring_llm") as scorer:
            agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": self.offering_id})
        content = scorer.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(content.split("\n\nProspect profile:\n")[1])
        self.assertEqual(scored_profile["tech_stack"], agent.data_service.fetch_tech_stack(self.prospect_id))
        self.assertEqual(stale["tech_stack"], ["stale"])


if __name__ == "__main__":
    unittest.main()
