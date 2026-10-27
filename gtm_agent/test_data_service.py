import json
import os
import unittest
from unittest.mock import Mock, patch

from . import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from . import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.dict(data_service.PROSPECTS, {
            "test-prospect": {
                "tech_stack": ["AWS"],
                "engagement_history": [],
                "account_details": {},
            },
        }, clear=True))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))

    def test_update_persists_returned_stack(self):
        result = data_service.update_prospect_info("test-prospect", "Kafka")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertEqual(result["tech_stack"], ["AWS", "Kafka"])
        self.assertEqual(data_service.fetch_tech_stack("test-prospect"), result["tech_stack"])
        data_service.update_prospect_info("test-prospect", "Kafka")
        self.assertEqual(data_service.fetch_tech_stack("test-prospect"), ["AWS", "Kafka"])

    def test_update_refreshes_cached_profile(self):
        before = gtm_agent.build_prospect_profile.invoke({"prospect_id": "test-prospect"})
        self.assertEqual(before["prospect_profile"]["tech_stack"], ["AWS"])

        data_service.update_prospect_info("test-prospect", "Kafka")
        after = gtm_agent.build_prospect_profile.invoke({"prospect_id": "test-prospect"})

        self.assertTrue(after["found"])
        self.assertEqual(after["prospect_profile"]["tech_stack"], ["AWS", "Kafka"])
        self.assertEqual(
            data_service.get_profile_from_db("test-prospect")["prospect_profile"],
            after["prospect_profile"],
        )

    def test_unknown_prospect_is_not_updated(self):
        result = data_service.update_prospect_info("unknown-prospect", "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertIsNone(data_service.get_prospect_record("unknown-prospect"))

    def test_score_receives_saved_stack_after_update(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": "test-prospect"})
        result = gtm_agent.update_prospect_info.invoke({
            "prospect_id": "test-prospect", "technology": "Kafka",
        })
        self.assertTrue(result["updated"])
        offering = {
            "required_tech_stack": ["AWS", "Kafka"],
            "min_annual_revenue": 0,
            "description": "Streaming infrastructure",
        }
        scoring_llm = Mock()
        scoring_llm.invoke.return_value.model_dump.return_value = {"score": 100}

        with patch.object(gtm_agent, "_scoring_llm", scoring_llm):
            score = gtm_agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"], "offering": offering,
            })

        self.assertEqual(score, {"score": 100})
        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(scored_profile["tech_stack"], ["AWS", "Kafka"])
