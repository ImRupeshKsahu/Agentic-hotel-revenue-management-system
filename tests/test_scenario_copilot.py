import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot_core.scenario_copilot import (
    ScenarioChatResponse,
    ScenarioChatContext,
    ScenarioConversationMemory,
    handle_scenario_chat,
    update_conversation_memory,
)


class ScenarioCopilotTests(unittest.TestCase):
    def _context(self):
        sep19_state = {
            "current_otb": 190,
            "raw_otb_occupancy": 190 / 237,
            "adjusted_otb": 184.0,
            "adjusted_otb_occupancy": 184 / 237,
            "expected_cancellations": 6.0,
            "historical_avg_otb": 150,
            "competitor_price": 150.0,
            "comp_low": 140.0,
            "comp_median": 150.0,
            "comp_high": 165.0,
            "sample_size": 5,
            "source_quality": "simulated",
            "market_regime": "normal_market",
            "market_as_of_timestamp": "2017-09-10T00:00:00",
            "booking_velocity": 1.15,
            "gross_pace_index": 1.15,
            "retained_pace_index": 1.10,
            "pickup_trend_index": 1.18,
            "pricing_pace_index": 1.16,
            "total_rooms": 237,
        }
        sep11_state = {
            **sep19_state,
            "current_otb": 184,
            "raw_otb_occupancy": 184 / 237,
            "adjusted_otb": 171.0,
            "adjusted_otb_occupancy": 171 / 237,
            "expected_cancellations": 13.41,
            "competitor_price": 162.16,
            "comp_low": 151.20,
            "comp_median": 162.16,
            "comp_high": 174.80,
            "market_regime": "sellout_regime",
            "booking_velocity": 1.05,
            "gross_pace_index": 1.05,
            "retained_pace_index": 1.02,
            "pickup_trend_index": 1.50,
            "pricing_pace_index": 1.05,
        }
        return ScenarioChatContext(
            target_date="2017-09-12",
            forecasted_occupancy=0.82,
            manual_demand_shock=0.0,
            current_state={
                "current_otb": 170,
                "raw_otb_occupancy": 170 / 237,
                "adjusted_otb": 160.0,
                "adjusted_otb_occupancy": 160 / 237,
                "expected_cancellations": 10.0,
                "historical_avg_otb": 140,
                "competitor_price": 140.0,
                "comp_low": 132.0,
                "comp_median": 140.0,
                "comp_high": 152.0,
                "sample_size": 5,
                "source_quality": "simulated",
                "market_regime": "normal_market",
                "market_as_of_timestamp": "2017-09-10T00:00:00",
                "booking_velocity": 1.08,
                "gross_pace_index": 1.08,
                "retained_pace_index": 1.04,
                "pickup_trend_index": 1.12,
                "pricing_pace_index": 1.09,
                "total_rooms": 237,
            },
            live_market_by_date={"2017-09-19": sep19_state, "2017-09-11": sep11_state},
            forecast_occupancy_by_date={"2017-09-19": 0.91, "2017-09-11": 0.904},
            horizon_records=[
                {
                    "date": "2017-09-11",
                    "recommended_adr": 165.0,
                    "raw_otb_occupancy": 184 / 237,
                    "adjusted_otb_occupancy": 171 / 237,
                    "forecasted_occupancy": 0.904,
                    "competitor_median": 162.16,
                    "review_status": "Review needed",
                    "manual_approval_required": False,
                    "sold_out": False,
                    "material_retention_gap": False,
                    "review_flags": ["High forecast demand and firm market context."],
                    "top_reasons": ["Strong expected demand with sellout market context."],
                    "revenue_upside": 700.0,
                    "expected_revenue": 39000.0,
                },
                {
                    "date": "2017-09-12",
                    "recommended_adr": 140.0,
                    "raw_otb_occupancy": 170 / 237,
                    "adjusted_otb_occupancy": 160 / 237,
                    "forecasted_occupancy": 0.82,
                    "competitor_median": 140.0,
                    "review_status": "No review",
                    "manual_approval_required": False,
                    "sold_out": False,
                    "material_retention_gap": False,
                    "review_flags": [],
                    "top_reasons": [],
                    "revenue_upside": 100.0,
                    "expected_revenue": 1000.0,
                },
                {
                    "date": "2017-09-19",
                    "recommended_adr": 155.0,
                    "raw_otb_occupancy": 190 / 237,
                    "adjusted_otb_occupancy": 184 / 237,
                    "forecasted_occupancy": 0.91,
                    "competitor_median": 150.0,
                    "review_status": "Review needed",
                    "manual_approval_required": True,
                    "sold_out": True,
                    "material_retention_gap": True,
                    "review_flags": ["Inventory is tight; confirm remaining-room strategy."],
                    "top_reasons": ["Demand is strong."],
                    "revenue_upside": 500.0,
                    "expected_revenue": 2000.0,
                },
            ],
            horizon_summary={"dates_evaluated": 2, "dates_needing_review": 1},
        )

    def _fake_result(self, **kwargs):
        return {
            "final_adr": 147.50,
            "pct_delta_from_reference": 4.25,
            "absolute_delta": 6.00,
            "competitor_gap_pct": 5.36,
            "pricing_pace_index": kwargs.get("pricing_pace_index", 1.0),
            "local_intel_applied_shock": kwargs.get("local_intel_applied_shock", 0.0),
            "local_intel_estimate": kwargs.get("local_intel_estimate", {}),
            "manual_event_text": kwargs.get("manual_event_text", ""),
            "ai_recommended_action": "Review Before Publishing",
            "ai_risk_level": "Medium",
        }

    def test_answers_scenario_data_question_from_context(self):
        response = handle_scenario_chat("What is booked and forecast occupancy?", self._context())

        self.assertIn("booked occupancy", response.answer)
        self.assertIn("forecast occupancy is 82.0%", response.answer)
        self.assertIn("expected cancellations", response.answer)
        self.assertIn("OTB snapshot", response.source_labels)

    def test_generic_concern_question_uses_horizon_risk_snapshot(self):
        response = handle_scenario_chat("Which date should be most concerning to me?", self._context())

        self.assertIn("2017-09-19", response.answer)
        self.assertIn("recommended ADR $155.00", response.answer)
        self.assertIn("Inventory is tight", response.answer)
        self.assertIn("30-day Scenario Lab risk snapshot", response.source_labels)

    def test_horizon_rank_followup_uses_prior_top_n_clarification(self):
        context = self._context()
        clarification = ScenarioChatResponse(
            answer="Which revenue basis should I use?",
            clarification_question="Which revenue basis should I use?",
            intent="data_question",
            domain="scenario_lab",
        )
        context.conversation_memory = update_conversation_memory(
            context.conversation_memory,
            "could you provide me top 10 highest revenue days",
            clarification,
        )

        response = handle_scenario_chat("recommended ADR revenue, consider next 30 days", context)

        self.assertIn("Top 3 dates by expected revenue at recommended ADR", response.answer)
        self.assertIn("\n1. 2017-09-11: $39,000.00", response.answer)
        self.assertIn("30-day Scenario Lab ranking", response.source_labels)
        self.assertNotIn("For 2017-09-12", response.answer)

    def test_explicit_highest_revenue_upside_defaults_to_one_and_ignores_prior_limit(self):
        context = self._context()
        context.conversation_memory = ScenarioConversationMemory(
            last_horizon_rank_request={
                "direction": "top",
                "limit": 10,
                "metric": "expected_revenue",
                "range": "next_30_days",
            }
        )

        response = handle_scenario_chat("what is highest upside in revenue based on recommended ADR?", context)

        self.assertIn("Top 1 dates by upside versus booked ADR", response.answer)
        self.assertIn("\n1. 2017-09-11: $700.00", response.answer)
        self.assertNotIn("\n2.", response.answer)
        self.assertNotIn("Used the prior ranking clarification", " ".join(response.assumptions))

    def test_top_four_revenue_upside_uses_explicit_word_count_and_newline_list(self):
        response = handle_scenario_chat("show top two highest revenue upside dates", self._context())

        self.assertIn("Top 2 dates by upside versus booked ADR", response.answer)
        self.assertIn("\n1. 2017-09-11: $700.00", response.answer)
        self.assertIn("\n2. 2017-09-19: $500.00", response.answer)
        self.assertNotIn("\n3.", response.answer)

    def test_explicit_least_projected_occupancy_overrides_prior_top_ten_memory(self):
        context = self._context()
        context.conversation_memory = ScenarioConversationMemory(
            last_horizon_rank_request={
                "direction": "top",
                "limit": 10,
                "metric": "expected_revenue",
                "range": "next_30_days",
            }
        )

        response = handle_scenario_chat("what are 5 least projected occupancy dates?", context)

        self.assertIn("Bottom 3 dates by forecast occupancy", response.answer)
        self.assertIn("1. 2017-09-12: 82.0%", response.answer)
        self.assertIn("2. 2017-09-11: 90.4%", response.answer)
        self.assertIn("3. 2017-09-19: 91.0%", response.answer)
        self.assertNotIn("Top 10", response.answer)
        self.assertNotIn("expected revenue", response.answer)

    def test_bottom_five_forecast_occupancy_handles_word_number_and_typo(self):
        response = handle_scenario_chat("what are bottom five dates w.r.t forrested occupancy", self._context())

        self.assertIn("Bottom 3 dates by forecast occupancy", response.answer)
        self.assertIn("1. 2017-09-12: 82.0%", response.answer)
        self.assertNotIn("Top 10", response.answer)

    def test_ranked_date_scenario_runs_on_least_revenue_date(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat(
                "what is the least revenue date and run a scenario for that with demand shock of 20%",
                self._context(),
            )

        self.assertTrue(response.ran_scenario)
        self.assertEqual(captured["target_date"], "2017-09-12")
        self.assertAlmostEqual(captured["manual_demand_shock"], 0.20)
        self.assertEqual(captured["forecasted_occupancy"], 0.82)

    def test_ranked_date_scenario_can_exclude_selected_date_and_reuse_prior_shock(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        context = self._context()
        clarification = ScenarioChatResponse(
            answer="Please confirm which date to use.",
            clarification_question="Please confirm which date to use.",
            intent="run_simulation",
            domain="scenario_lab",
        )
        context.conversation_memory = update_conversation_memory(
            context.conversation_memory,
            "what is the least revenue date and run a scenario for that with demand shock of 20%",
            clarification,
        )

        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat("run scenario for the lowest revenue date not the selected date", context)

        self.assertTrue(response.ran_scenario)
        self.assertEqual(captured["target_date"], "2017-09-19")
        self.assertAlmostEqual(captured["manual_demand_shock"], 0.20)

    def test_pricing_strategy_includes_recommended_adr_for_requested_date(self):
        response = handle_scenario_chat(
            "show me the pricing strategy for 11th september. provide other relevant details too",
            self._context(),
        )

        self.assertIn("For 2017-09-11, recommended ADR is $165.00", response.answer)
        self.assertIn("comp median of $162.16", response.answer)
        self.assertIn("forecast occupancy is 90.4%", response.answer)
        self.assertIn("Scenario Lab pricing recommendation", response.source_labels)
        self.assertEqual(response.referenced_date, "2017-09-11")

    def test_pricing_strategy_tolerates_typo_and_tracks_requested_date(self):
        response = handle_scenario_chat(
            "show me the prcing strategy for 11th september",
            self._context(),
        )

        self.assertIn("For 2017-09-11, recommended ADR is $165.00", response.answer)
        self.assertIn("Scenario Lab pricing recommendation", response.source_labels)
        self.assertEqual(response.referenced_date, "2017-09-11")

    def test_recommended_adr_followup_reuses_last_scenario_lab_date(self):
        context = self._context()
        first_response = handle_scenario_chat(
            "show me the pricing strategy for 11th september. provide other relevant details too",
            context,
        )
        context.conversation_memory = update_conversation_memory(
            context.conversation_memory,
            "show me the pricing strategy for 11th september. provide other relevant details too",
            first_response,
        )

        followup = handle_scenario_chat("what is recommended ADR", context)

        self.assertIn("For 2017-09-11, recommended ADR is $165.00", followup.answer)
        self.assertNotIn("2017-09-12", followup.answer)
        self.assertEqual(followup.referenced_date, "2017-09-11")

    def test_data_question_followup_reuses_last_scenario_lab_date(self):
        context = self._context()
        context.conversation_memory = ScenarioConversationMemory(
            last_domain="scenario_lab",
            last_target_date="2017-09-11",
        )

        response = handle_scenario_chat("how is occupancy looking?", context)

        self.assertIn("For 2017-09-11", response.answer)
        self.assertIn("forecast occupancy is 90.4%", response.answer)
        self.assertEqual(response.referenced_date, "2017-09-11")

    def test_demand_shock_followup_reuses_last_scenario_lab_date(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        context = self._context()
        first_response = handle_scenario_chat("show me the prcing strategy for 11th september", context)
        context.conversation_memory = update_conversation_memory(
            context.conversation_memory,
            "show me the prcing strategy for 11th september",
            first_response,
        )

        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            followup = handle_scenario_chat("what happens if i give a demand shock of +15%", context)

        self.assertTrue(followup.ran_scenario)
        self.assertEqual(captured["target_date"], "2017-09-11")
        self.assertAlmostEqual(captured["manual_demand_shock"], 0.15)
        self.assertEqual(captured["forecasted_occupancy"], 0.904)

    def test_forecast_audit_question_uses_audit_artifacts_not_selected_date(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            champion_path = Path(tmp_dir) / "forecast_champion.json"
            audit_path = Path(tmp_dir) / "backtest_audit_summary.csv"
            champion_path.write_text(
                json.dumps(
                    {
                        "model": "extra_trees_recursive",
                        "metrics": {"MAE_pp": 8.78, "RMSE_pp": 10.54, "Bias_pp": -3.13},
                        "backtest_metadata": {"audit_status": "ok"},
                    }
                ),
                encoding="utf-8",
            )
            audit_path.write_text(
                "\n".join(
                    [
                        "Model,Folds,Observations,MAE_pp,RMSE_pp,Bias_pp,Is_Champion,Selection_Mean_Fold_MAE_pp,Audit_Status",
                        "extra_trees_recursive,8,240,2.32,2.88,-0.93,True,8.78,ok",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("copilot_core.scenario_copilot.FORECAST_CHAMPION_PATH", str(champion_path)), patch(
                "copilot_core.scenario_copilot.BACKTEST_AUDIT_SUMMARY_PATH", str(audit_path)
            ):
                response = handle_scenario_chat(
                    "what is average occupancy miss for current best model during audit?",
                    self._context(),
                )

        self.assertIn("current best model is extra_trees_recursive", response.answer)
        self.assertIn("average occupancy miss is 2.32 percentage points", response.answer)
        self.assertIn("240 forecasted stay-date observations", response.answer)
        self.assertNotIn("2017-09-12", response.answer)
        self.assertNotIn("selected", response.answer.lower())
        self.assertIn("Forecast audit summary", response.source_labels)

    def test_forecast_audit_question_tolerates_typos_and_abbreviations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            champion_path = Path(tmp_dir) / "forecast_champion.json"
            audit_path = Path(tmp_dir) / "backtest_audit_summary.csv"
            champion_path.write_text(
                json.dumps({"model": "extra_trees_recursive", "metrics": {"MAE_pp": 8.78}}),
                encoding="utf-8",
            )
            audit_path.write_text(
                "\n".join(
                    [
                        "Model,Folds,Observations,MAE_pp,RMSE_pp,Bias_pp,Is_Champion,Selection_Mean_Fold_MAE_pp,Audit_Status",
                        "extra_trees_recursive,8,240,2.32,2.88,-0.93,True,8.78,ok",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("copilot_core.scenario_copilot.FORECAST_CHAMPION_PATH", str(champion_path)), patch(
                "copilot_core.scenario_copilot.BACKTEST_AUDIT_SUMMARY_PATH", str(audit_path)
            ):
                response = handle_scenario_chat(
                    "whati is avg occpancy miss for best model for forcasting during audit?",
                    self._context(),
                )

        self.assertIn("current best model is extra_trees_recursive", response.answer)
        self.assertIn("average occupancy miss is 2.32 percentage points", response.answer)
        self.assertNotIn("2017-09-12", response.answer)
        self.assertNotIn("Scenario Lab snapshot", response.source_labels)
        self.assertIn("Forecast audit summary", response.source_labels)

    def test_forecast_audit_question_respects_second_best_rank(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            champion_path = Path(tmp_dir) / "forecast_champion.json"
            audit_path = Path(tmp_dir) / "backtest_audit_summary.csv"
            champion_path.write_text(
                json.dumps({"model": "extra_trees_recursive", "metrics": {"MAE_pp": 8.78}}),
                encoding="utf-8",
            )
            audit_path.write_text(
                "\n".join(
                    [
                        "Model,Folds,Observations,MAE_pp,RMSE_pp,Bias_pp,Is_Champion,Selection_Mean_Fold_MAE_pp,Audit_Status",
                        "extra_trees_recursive,8,240,2.32,2.88,-0.93,True,8.78,ok",
                        "random_forest_recursive,8,240,2.63,3.17,-1.47,False,,",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("copilot_core.scenario_copilot.FORECAST_CHAMPION_PATH", str(champion_path)), patch(
                "copilot_core.scenario_copilot.BACKTEST_AUDIT_SUMMARY_PATH", str(audit_path)
            ):
                response = handle_scenario_chat(
                    "what is avg occupancy miss for 2nd best model in audit for forecasting occupancy?",
                    self._context(),
                )

        self.assertIn("second-best model is random_forest_recursive", response.answer)
        self.assertIn("average occupancy miss is 2.63 percentage points", response.answer)
        self.assertNotIn("current best model is extra_trees_recursive", response.answer)
        self.assertNotIn("selection-backtest average miss was 8.78 pp", response.answer)
        self.assertIn("Forecast audit summary", response.source_labels)

    def test_forecast_backtest_question_uses_leaderboard_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            champion_path = Path(tmp_dir) / "forecast_champion.json"
            comparison_path = Path(tmp_dir) / "model_comparison_metrics.csv"
            champion_path.write_text(
                json.dumps({"model": "extra_trees_recursive", "metrics": {"MAE_pp": 8.78}}),
                encoding="utf-8",
            )
            comparison_path.write_text(
                "\n".join(
                    [
                        "Feature_Profile,Model,Strategy,Folds,Observations,MAE_pp,RMSE_pp,Bias_pp,MAPE,WAPE,Stability",
                        "boruta_selected,extra_trees_recursive,recursive_ml,49,1470,8.78,10.54,-3.13,12.83,12.31,0.928",
                        "boruta_selected,random_forest_recursive,recursive_ml,49,1470,9.18,10.99,-1.42,14.36,13.28,0.926",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("copilot_core.scenario_copilot.FORECAST_CHAMPION_PATH", str(champion_path)), patch(
                "copilot_core.scenario_copilot.MODEL_COMPARISON_PATH", str(comparison_path)
            ):
                response = handle_scenario_chat("show me backtesting KPI results for the best model", self._context())

        self.assertIn("selection backtest champion is extra_trees_recursive", response.answer)
        self.assertIn("49 folds", response.answer)
        self.assertIn("1470 forecasted stay-date observations", response.answer)
        self.assertIn("Avg occupancy miss (MAE) is 8.78 pp", response.answer)
        self.assertIn("WAPE is 12.31%", response.answer)
        self.assertIn("Forecast backtest leaderboard", response.source_labels)

    def test_compare_top_two_models_does_not_match_comp_set_context(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            comparison_path = Path(tmp_dir) / "model_comparison_metrics.csv"
            comparison_path.write_text(
                "\n".join(
                    [
                        "Feature_Profile,Model,Strategy,Folds,Observations,MAE_pp,RMSE_pp,Bias_pp,MAPE,WAPE,Stability",
                        "boruta_selected,extra_trees_recursive,recursive_ml,49,1470,8.78,10.54,-3.13,12.83,12.31,0.928",
                        "boruta_selected,random_forest_recursive,recursive_ml,49,1470,9.18,10.99,-1.42,14.36,13.28,0.926",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("copilot_core.scenario_copilot.MODEL_COMPARISON_PATH", str(comparison_path)):
                response = handle_scenario_chat(
                    "compare the top two best models for forecasting occupancy?",
                    self._context(),
                )

        self.assertIn("top two selection-backtest models", response.answer)
        self.assertIn("extra_trees_recursive", response.answer)
        self.assertIn("random_forest_recursive", response.answer)
        self.assertIn("ahead by 0.40 pp", response.answer)
        self.assertNotIn("comp set", response.answer.lower())
        self.assertNotIn("market regime", response.answer.lower())
        self.assertIn("Forecast backtest leaderboard", response.source_labels)

    def test_followup_audit_performance_uses_previous_compared_models(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            comparison_path = Path(tmp_dir) / "model_comparison_metrics.csv"
            audit_path = Path(tmp_dir) / "backtest_audit_summary.csv"
            comparison_path.write_text(
                "\n".join(
                    [
                        "Feature_Profile,Model,Strategy,Folds,Observations,MAE_pp,RMSE_pp,Bias_pp,MAPE,WAPE,Stability",
                        "boruta_selected,extra_trees_recursive,recursive_ml,49,1470,8.78,10.54,-3.13,12.83,12.31,0.928",
                        "boruta_selected,random_forest_recursive,recursive_ml,49,1470,9.18,10.99,-1.42,14.36,13.28,0.926",
                    ]
                ),
                encoding="utf-8",
            )
            audit_path.write_text(
                "\n".join(
                    [
                        "Model,Folds,Observations,MAE_pp,RMSE_pp,Bias_pp,WAPE,Stability",
                        "extra_trees_recursive,8,240,2.32,2.88,-0.93,2.53,0.974",
                        "random_forest_recursive,8,240,2.63,3.17,-1.47,2.87,0.975",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("copilot_core.scenario_copilot.MODEL_COMPARISON_PATH", str(comparison_path)), patch(
                "copilot_core.scenario_copilot.BACKTEST_AUDIT_SUMMARY_PATH", str(audit_path)
            ):
                context = self._context()
                first_response = handle_scenario_chat("compare the top two best models", context)
                context.conversation_memory = update_conversation_memory(
                    context.conversation_memory,
                    "compare the top two best models",
                    first_response,
                )
                followup = handle_scenario_chat("what about their audit performance?", context)

        self.assertIn("On recent audit performance", followup.answer)
        self.assertIn("extra_trees_recursive has MAE 2.32 pp", followup.answer)
        self.assertIn("random_forest_recursive has MAE 2.63 pp", followup.answer)
        self.assertIn("ahead by 0.31 pp", followup.answer)
        self.assertNotIn("2017-09-12", followup.answer)
        self.assertNotIn("comp set", followup.answer.lower())
        self.assertIn("Forecast audit summary", followup.source_labels)

    def test_local_intel_scenario_requires_confirmation_before_run(self):
        with patch("copilot_core.scenario_copilot.run_agentic_pricing") as fake_run:
            response = handle_scenario_chat(
                "Run scenario for a 150-person conference nearby",
                self._context(),
            )

        fake_run.assert_not_called()
        self.assertIsNotNone(response.draft)
        self.assertIsNotNone(response.confirmation_prompt)
        self.assertTrue(response.draft.confirmation_required)
        self.assertFalse(response.ran_scenario)

    def test_price_question_does_not_trigger_simulation_run(self):
        with patch("copilot_core.scenario_copilot.run_agentic_pricing") as fake_run:
            response = handle_scenario_chat("What price context do we have?", self._context())

        fake_run.assert_not_called()
        self.assertFalse(response.ran_scenario)
        self.assertIn("booked occupancy", response.answer)

    def test_manual_demand_recommended_price_runs_without_confirmation_or_local_intel(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat(
                "On 19th September there is an upside demand by 20%. what is recommended price for rooms?",
                self._context(),
            )

        self.assertTrue(response.ran_scenario)
        self.assertIsNone(response.confirmation_prompt)
        self.assertAlmostEqual(captured["manual_demand_shock"], 0.20)
        self.assertEqual(captured["local_intel_estimate"], {})
        self.assertEqual(captured["manual_event_text"], "")
        self.assertEqual(captured["local_intel_applied_shock"], 0.0)
        self.assertIn("Final ADR", response.answer)
        self.assertIn("no local-intel impact was included", response.answer)

    def test_yes_runs_pending_memory_draft_when_ui_prompt_was_text_only(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        context = self._context()
        context.conversation_memory = ScenarioConversationMemory(
            last_target_date="2017-09-19",
            last_manual_demand_shock=0.20,
            last_draft_pending=True,
            last_intent="scenario_draft",
        )

        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat("yes", context)

        self.assertTrue(response.ran_scenario)
        self.assertAlmostEqual(captured["manual_demand_shock"], 0.20)
        self.assertEqual(captured["local_intel_estimate"], {})

    def test_context_only_confirmation_runs_without_applying_local_intel(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        context = self._context()
        draft_response = handle_scenario_chat(
            "Run scenario for a 150-person conference nearby",
            context,
        )
        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat("run context only", context, draft_response.draft)

        self.assertTrue(response.ran_scenario)
        self.assertEqual(captured["local_intel_applied_shock"], 0.0)
        self.assertEqual(captured["manual_event_text"], "a 150-person conference nearby")
        self.assertIn("context only", response.answer)

    def test_confirmed_local_intel_is_applied_to_priced_demand(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        context = self._context()
        draft_response = handle_scenario_chat(
            "Run scenario for a 150-person conference nearby",
            context,
        )
        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat("confirm and run", context, draft_response.draft)

        self.assertTrue(response.ran_scenario)
        self.assertGreater(captured["local_intel_applied_shock"], 0.0)
        self.assertIn("included in priced demand", response.answer)

    def test_market_override_requires_confirmation_and_runs_with_override(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        context = self._context()
        draft_response = handle_scenario_chat(
            "Run scenario with competitors up 8%",
            context,
        )
        self.assertIsNotNone(draft_response.confirmation_prompt)

        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat("confirm", context, draft_response.draft)

        self.assertTrue(response.ran_scenario)
        self.assertEqual(captured["market_context"]["comp_median"], 151.2)
        self.assertEqual(captured["competitor_price"], 151.2)

    def test_user_prompt_parses_natural_date_and_market_percent_override(self):
        response = handle_scenario_chat(
            "for 19th september what if competitor prices increase by 20%, what will be recommended price and suggestions?",
            self._context(),
        )

        self.assertIsNotNone(response.draft)
        self.assertEqual(response.draft.target_date, "2017-09-19")
        self.assertEqual(response.draft.market_context_override["comp_low"], 168.0)
        self.assertEqual(response.draft.market_context_override["comp_median"], 180.0)
        self.assertEqual(response.draft.market_context_override["comp_high"], 198.0)
        self.assertIn("use the market override", response.confirmation_prompt)
        self.assertIn("2017-09-19", response.answer)
        self.assertNotIn("$20.00", response.answer)

    def test_confirmed_chat_date_uses_matching_forecast_and_live_state(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        context = self._context()
        draft_response = handle_scenario_chat(
            "for 19th september what if competitor prices increase by 20%, what will be recommended price and suggestions?",
            context,
        )
        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat("confirm", context, draft_response.draft)

        self.assertTrue(response.ran_scenario)
        self.assertEqual(captured["target_date"], "2017-09-19")
        self.assertEqual(captured["forecasted_occupancy"], 0.91)
        self.assertAlmostEqual(captured["current_occupancy"], 184 / 237)
        self.assertAlmostEqual(captured["raw_otb_occupancy"], 190 / 237)
        self.assertEqual(captured["market_context"]["comp_median"], 180.0)

    def test_chat_run_uses_same_core_inputs_as_sidebar_scenario(self):
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return self._fake_result(**kwargs)

        with patch("copilot_core.scenario_copilot.run_agentic_pricing", side_effect=fake_run):
            response = handle_scenario_chat("Run scenario with demand up 5%", self._context())

        self.assertTrue(response.ran_scenario)
        self.assertEqual(captured["target_date"], "2017-09-12")
        self.assertAlmostEqual(captured["current_occupancy"], 160 / 237)
        self.assertEqual(captured["forecasted_occupancy"], 0.82)
        self.assertEqual(captured["manual_demand_shock"], 0.05)
        self.assertAlmostEqual(captured["raw_otb_occupancy"], 170 / 237)
        self.assertAlmostEqual(captured["adjusted_otb_occupancy"], 160 / 237)

    def test_manager_facing_response_avoids_internal_pricing_terms(self):
        context = self._context()
        context.latest_result = self._fake_result()

        response = handle_scenario_chat("Why did the ADR change?", context)

        lowered = response.answer.lower()
        self.assertNotIn("raw otb", lowered)
        self.assertNotIn("optimizer", lowered)
        self.assertNotIn("replacement adr", lowered)
        self.assertIn("Final ADR", response.answer)


if __name__ == "__main__":
    unittest.main()
