import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langsmith import Client, trace, tracing_context

with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.client = Mock(spec=Client)
        self.tracing = tracing_context(enabled=False, client=self.client)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.pid = next(iter(data_service.PROSPECTS))
        self.record = data_service.PROSPECTS[self.pid]
        self.offering = next(iter(data_service.OFFERINGS.values()))
        self.score = agent.ProspectScore(
            score=75,
            justification="Revenue and technology fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=60, segment_fit=75
            ),
        )

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for record in data_service.PROSPECTS.values():
            for key, secret in record["billing_qualification"].items():
                self.assertNotIn(key, serialized)
                self.assertNotIn(secret, serialized)

    def test_contact_allowlist_and_email_behavior(self):
        for pid, record in data_service.PROSPECTS.items():
            with self.subTest(pid=pid):
                with patch.dict(record, {"extra_enrichment": "not a contact field"}):
                    result = agent.get_prospect.invoke({"prospect_id": pid})
                expected = {"prospect_id": pid, **{
                    k: record[k] for k in ("name", "email", "disqualified")
                }}
                self.assertEqual(result, {"prospect": expected, "found": True})
                self.assert_no_billing(result)
                sent = agent.send_prospect_email.func(
                    expected, "Hello", "Following up", SimpleNamespace(config={}),
                    from_rep={"name": "Rep", "email": "rep@example.com"},
                )
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["to"], record["email"])
                self.assertEqual(sent["body"], "Following up")

    def test_missing_results(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"),
                         {"prospect_profile": None})

    def test_fresh_and_cached_profiles_through_scoring(self):
        for cached in (False, True):
            for pid, record in data_service.PROSPECTS.items():
                with self.subTest(cached=cached, pid=pid):
                    data_service._PROFILES.clear()
                    expected = {"prospect_id": pid, **{
                        k: v for k, v in record.items() if k != "billing_qualification"
                    }}
                    if cached:
                        data_service._PROFILES[pid] = {"prospect_id": pid, **record}
                    runs = []
                    with tracing_context(enabled=True, client=self.client):
                        with trace("profile flow", inputs={"prospect_id": pid}) as root:
                            result = agent.build_prospect_profile.invoke({"prospect_id": pid})
                            lookup = data_service.get_profile_from_db(
                                pid, langsmith_extra={"on_end": runs.append}
                            )
                    pending = list(root.child_runs)
                    while pending:
                        run = pending.pop()
                        self.assert_no_billing(run.inputs)
                        self.assert_no_billing(run.outputs)
                        pending.extend(run.child_runs)
                    self.assertEqual(result, {"prospect_profile": expected, "found": True})
                    self.assertEqual(lookup, {"prospect_profile": expected})
                    self.assertEqual(len(runs), 1)
                    self.assert_no_billing(runs[0].inputs)
                    self.assert_no_billing(runs[0].outputs)
                    if not cached:
                        self.assertEqual(data_service._PROFILES[pid], expected)
                    arguments = {"prospect_profile": result["prospect_profile"],
                                 "offering": self.offering}
                    self.assert_no_billing(arguments)
                    with patch.object(agent, "_scoring_llm") as model:
                        model.invoke.return_value = self.score
                        scored = agent.score_prospect.invoke(arguments)
                    self.assertEqual(scored, self.score.model_dump())
                    self.assert_no_billing(model.invoke.call_args.args[0])

    def test_scoring_uses_only_rubric_inputs_and_authoritative_stack(self):
        profile = {
            "prospect_id": self.pid, **self.record,
            "segment": "Enterprise", "tech_stack": ["invented technology"],
            "unrelated": "private notes",
            "account_details": [{
                **self.record["account_details"][0], "segment": "Enterprise",
                "billing_qualification": self.record["billing_qualification"],
                "internal_notes": "private account notes",
            }],
        }
        with patch.object(agent, "_scoring_llm") as model:
            model.invoke.return_value = self.score
            result = agent.score_prospect.invoke({
                "prospect_profile": profile, "offering": self.offering,
            })
        self.assertEqual(result, self.score.model_dump())
        messages = model.invoke.call_args.args[0]
        self.assert_no_billing(messages)
        scoring_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        self.assertEqual(scoring_profile, {
            "annual_revenue": self.record["annual_revenue"],
            "tech_stack": self.record["tech_stack"],
            "segment": "Enterprise",
            "account_details": [{
                k: v for k, v in profile["account_details"][0].items()
                if k in ("industry", "sub_industry", "founded_year", "segment")
            }],
        })
        self.assertEqual(profile["tech_stack"], ["invented technology"])

    def test_invalid_offering_still_returns_error(self):
        with patch.object(agent, "_scoring_llm") as model:
            for offering in (None, {}, {"required_tech_stack": ["AWS"]}):
                result = agent.score_prospect.invoke({
                    "prospect_profile": self.record, "offering": offering,
                })
                self.assertEqual(result, {
                    "score": None, "error": "Cannot score without a valid offering.",
                })
            model.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
