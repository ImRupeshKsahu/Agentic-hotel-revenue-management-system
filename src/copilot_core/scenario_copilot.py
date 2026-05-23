"""Bounded Scenario Lab chat service.

The service translates manager chat into scenario-lab operations, but pricing
still flows through the deterministic pricing agent.
"""

from __future__ import annotations

import csv
import difflib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import (
    BASE_CAPACITY,
    BACKTEST_AUDIT_SUMMARY_PATH,
    DATA_END_DATE,
    FORECAST_CHAMPION_PATH,
    MODEL_COMPARISON_PATH,
)
from pricing_core.local_intel import estimate_local_intel_impact
from copilot_core.pricing_agent import run_agentic_pricing


@dataclass
class ScenarioConversationMemory:
    rolling_summary: str = ""
    last_user_message: str = ""
    last_intent: str = ""
    last_domain: str = ""
    last_target_date: str = ""
    last_referenced_models: List[str] = field(default_factory=list)
    last_comparison_basis: str = ""
    last_local_intel_text: str = ""
    last_manual_demand_shock: Optional[float] = None
    last_market_context_override: Optional[Dict[str, Any]] = None
    last_horizon_rank_request: Dict[str, Any] = field(default_factory=dict)
    last_draft_pending: bool = False
    last_scenario_result: Dict[str, Any] = field(default_factory=dict)
    previous_scenario_result: Dict[str, Any] = field(default_factory=dict)
    last_sources: List[str] = field(default_factory=list)


@dataclass
class ScenarioChatContext:
    target_date: str
    forecasted_occupancy: float
    current_state: Dict[str, Any]
    manual_demand_shock: float = 0.0
    latest_result: Optional[Dict[str, Any]] = None
    live_market_by_date: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    forecast_occupancy_by_date: Dict[str, float] = field(default_factory=dict)
    clarification_count: int = 0
    conversation_memory: ScenarioConversationMemory = field(default_factory=ScenarioConversationMemory)
    horizon_records: List[Dict[str, Any]] = field(default_factory=list)
    horizon_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioDraft:
    target_date: str
    manual_demand_shock: float = 0.0
    local_intel_text: str = ""
    local_intel_estimate: Dict[str, Any] = field(default_factory=dict)
    market_context_override: Optional[Dict[str, Any]] = None
    apply_local_intel: bool = False
    confirmation_required: bool = False
    confirmed: bool = False
    requested_run: bool = False


@dataclass
class ScenarioChatResponse:
    answer: str
    draft: Optional[ScenarioDraft] = None
    confirmation_prompt: Optional[str] = None
    scenario_result: Optional[Dict[str, Any]] = None
    source_labels: List[str] = field(default_factory=list)
    ran_scenario: bool = False
    intent: str = ""
    domain: str = ""
    referenced_models: List[str] = field(default_factory=list)
    comparison_basis: str = ""
    referenced_date: str = ""
    clarification_question: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)
    grounding_sources: List[str] = field(default_factory=list)
    safety_flags: List[str] = field(default_factory=list)


EVENT_TERMS = {
    "wedding",
    "conference",
    "convention",
    "expo",
    "summit",
    "festival",
    "concert",
    "tournament",
    "match",
    "fifa",
    "world cup",
    "cricket",
    "football",
    "ipl",
    "grand prix",
    "banquet",
    "event",
    "local intel",
    "person",
    "people",
    "guest",
    "rooms",
    "pax",
}
DISRUPTION_TERMS = {
    "traffic",
    "road closure",
    "strike",
    "protest",
    "weather",
    "storm",
    "airport shutdown",
    "cancelled flight",
    "canceled flight",
    "stranded",
}
RUN_TERMS = {"run", "simulate", "rerun"}
CONFIRM_TERMS = {"yes", "confirm", "confirmed", "approve", "approved", "apply and run", "go ahead"}
CONTEXT_ONLY_TERMS = {
    "context only",
    "do not apply",
    "don't apply",
    "without applying",
    "dont apply",
    "keep context",
}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
}
ROUTING_TERM_ALIASES = {
    "avg": "average",
    "avrg": "average",
    "averge": "average",
    "whati": "what",
    "wht": "what",
    "occpancy": "occupancy",
    "ocupancy": "occupancy",
    "occupnacy": "occupancy",
    "occupany": "occupancy",
    "forcasting": "forecasting",
    "forcast": "forecast",
    "forcasted": "forecasted",
    "forecastng": "forecasting",
    "forrested": "forecasted",
    "projected": "forecasted",
    "projection": "forecast",
    "modle": "model",
    "modles": "models",
    "perfromance": "performance",
    "performace": "performance",
    "audt": "audit",
    "baktest": "backtest",
    "backtes": "backtest",
    "compar": "compare",
    "compair": "compare",
    "prcing": "pricing",
    "pricng": "pricing",
    "pricin": "pricing",
    "stratgey": "strategy",
    "startegy": "strategy",
    "recomended": "recommended",
    "recomend": "recommend",
    "recommnded": "recommended",
}
ROUTING_CANONICAL_TERMS = {
    "adr",
    "average",
    "audit",
    "backtest",
    "backtesting",
    "best",
    "champion",
    "compare",
    "comparison",
    "forecast",
    "forecasted",
    "forecasting",
    "leader",
    "leaderboard",
    "least",
    "metric",
    "miss",
    "model",
    "models",
    "occupancy",
    "performance",
    "price",
    "pricing",
    "recommend",
    "recommendation",
    "recommended",
    "result",
    "strategy",
    "validation",
}
MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def handle_scenario_chat(
    message: str,
    context: ScenarioChatContext,
    pending_draft: Optional[ScenarioDraft] = None,
) -> ScenarioChatResponse:
    text = (message or "").strip()
    normalized = text.lower()
    if not normalized:
        return ScenarioChatResponse(
            answer=(
                "Ask about dates, top/bottom horizon rankings, local intel, forecast audit KPIs, "
                "or a scenario you want to run."
            ),
            source_labels=["Scenario Lab"],
        )

    if pending_draft and _is_context_only_confirmation(normalized):
        draft = _copy_draft(pending_draft)
        draft.confirmed = True
        draft.apply_local_intel = False
        draft.confirmation_required = False
        return _run_confirmed_draft(draft, context)

    if pending_draft and _is_confirmation(normalized):
        draft = _copy_draft(pending_draft)
        draft.confirmed = True
        draft.apply_local_intel = _local_intel_can_affect_price(draft)
        draft.confirmation_required = False
        return _run_confirmed_draft(draft, context)

    if not pending_draft and _is_confirmation(normalized):
        memory_draft = _draft_from_pending_memory(context)
        if memory_draft:
            memory_draft.confirmed = True
            memory_draft.apply_local_intel = _local_intel_can_affect_price(memory_draft)
            memory_draft.confirmation_required = False
            return _run_confirmed_draft(memory_draft, context)

    if _is_scenario_request(normalized):
        draft = build_scenario_draft(text, context)
        if draft.requested_run:
            if draft.confirmation_required:
                return ScenarioChatResponse(
                    answer=_draft_summary(draft)
                    + " I need confirmation before those inputs affect priced demand or market values.",
                    draft=draft,
                    confirmation_prompt=_confirmation_prompt(draft),
                    source_labels=["Scenario draft", "Local intel estimate", "Market context"],
                )
            return _run_confirmed_draft(draft, context)

        return ScenarioChatResponse(
            answer=_draft_summary(draft),
            draft=draft,
            confirmation_prompt=_confirmation_prompt(draft) if draft.confirmation_required else None,
            source_labels=["Scenario draft", "Local intel estimate", "Market context"],
        )

    return answer_scenario_question(text, context, pending_draft)


def build_scenario_draft(message: str, context: ScenarioChatContext) -> ScenarioDraft:
    normalized = message.lower()
    target_date = _target_date_for_scenario_action(message, context)
    scenario_state = _state_for_date(context, target_date)
    scenario_forecast = _forecast_for_date(context, target_date)
    manual_shock = _parse_demand_shock(normalized)
    if manual_shock is None:
        manual_shock = float(context.conversation_memory.last_manual_demand_shock or context.manual_demand_shock or 0.0)

    local_intel_text = _extract_local_intel_text(message)
    current_occ = _adjusted_occupancy(scenario_state)
    local_estimate = (
        estimate_local_intel_impact(
            local_intel_text,
            current_occ=current_occ,
            forecast_occ=float(scenario_forecast or 0.0),
            booking_velocity=_safe_float(scenario_state.get("booking_velocity"), 1.0),
            retained_pace_index=_safe_float(
                scenario_state.get("retained_pace_index"),
                _safe_float(scenario_state.get("booking_velocity"), 1.0),
            ),
            pickup_trend_index=_safe_float(
                scenario_state.get("pickup_trend_index"),
                _safe_float(scenario_state.get("booking_velocity"), 1.0),
            ),
            target_date=target_date,
            market_context=_market_context_from_state(scenario_state),
        )
        if local_intel_text
        else {}
    )
    market_override = _parse_market_override(normalized, scenario_state)
    context_only = _is_context_only_confirmation(normalized)
    confirmation_required = (
        not context_only
        and (_local_estimate_can_affect_price(local_estimate) or market_override is not None)
    )

    return ScenarioDraft(
        target_date=target_date,
        manual_demand_shock=manual_shock,
        local_intel_text=local_intel_text,
        local_intel_estimate=local_estimate,
        market_context_override=market_override,
        apply_local_intel=False,
        confirmation_required=confirmation_required,
        confirmed=not confirmation_required,
        requested_run=_is_run_request(normalized),
    )


def answer_scenario_question(
    message: str,
    context: ScenarioChatContext,
    pending_draft: Optional[ScenarioDraft] = None,
) -> ScenarioChatResponse:
    normalized = message.lower()
    target_date = _target_date_for_question(message, context)
    state = _state_for_date(context, target_date)
    forecast = _forecast_for_date(context, target_date)
    latest = context.latest_result

    if _is_horizon_risk_question(normalized):
        return answer_horizon_risk_question(context)

    if _is_forecast_audit_question(normalized):
        return answer_forecast_audit_question(message)

    if _is_forecast_audit_followup_question(normalized, context.conversation_memory):
        return answer_forecast_model_audit_comparison_question(context.conversation_memory.last_referenced_models)

    if _is_forecast_model_comparison_question(normalized):
        return answer_forecast_model_comparison_question()

    if _is_forecast_backtest_question(normalized):
        return answer_forecast_backtest_question()

    if _is_horizon_rank_question(message, context):
        return answer_horizon_rank_question(message, context)

    if _is_pricing_strategy_question(normalized):
        return answer_pricing_strategy_question(message, context)

    if pending_draft and ("pending" in normalized or "draft" in normalized or "local intel" in normalized):
        return ScenarioChatResponse(
            answer=_draft_summary(pending_draft),
            draft=pending_draft,
            confirmation_prompt=_confirmation_prompt(pending_draft)
            if pending_draft.confirmation_required
            else None,
            source_labels=["Scenario draft", "Local intel estimate"],
        )

    if latest and any(
        term in normalized
        for term in ["why", "explain", "result", "adr", "price", "decision", "context", "behind this", "behind it"]
    ):
        return ScenarioChatResponse(
            answer=_result_summary(latest),
            scenario_result=latest,
            source_labels=["Latest scenario result", "Price Path", "Decision Context"],
        )

    if _is_market_context_question(normalized):
        answer = (
            f"For {target_date}, the comp set is "
            f"${_safe_float(state.get('comp_low'), state.get('competitor_price', 0.0)):.2f} low, "
            f"${_safe_float(state.get('comp_median'), state.get('competitor_price', 0.0)):.2f} median, and "
            f"${_safe_float(state.get('comp_high'), state.get('competitor_price', 0.0)):.2f} high. "
            f"Market regime is {str(state.get('market_regime', 'n/a')).replace('_', ' ')}."
        )
        return ScenarioChatResponse(
            answer=answer,
            source_labels=["Live market state"],
            intent="data_question",
            domain="scenario_lab",
            referenced_date=target_date,
        )

    if any(term in normalized for term in ["pace", "pickup", "momentum"]):
        answer = (
            f"For {target_date}, booked pace is {_safe_float(state.get('gross_pace_index'), 1.0):.2f}x, "
            f"likely retained pace is {_safe_float(state.get('retained_pace_index'), 1.0):.2f}x, "
            f"recent pickup is {_safe_float(state.get('pickup_trend_index'), 1.0):.2f}x, and "
            f"pricing pace is {_safe_float(state.get('pricing_pace_index'), 1.0):.2f}x."
        )
        return ScenarioChatResponse(
            answer=answer,
            source_labels=["OTB pace signals"],
            intent="data_question",
            domain="scenario_lab",
            referenced_date=target_date,
        )

    if any(term in normalized for term in ["occupancy", "booked", "retained", "cancellation", "forecast"]):
        answer = (
            f"For {target_date}, booked occupancy is {_format_pct(_raw_occupancy(state))}, "
            f"likely retained occupancy is {_format_pct(_adjusted_occupancy(state))}, "
            f"forecast occupancy is {_format_pct(forecast)}, and expected cancellations are "
            f"{_safe_float(state.get('expected_cancellations'), 0.0):.2f} rooms."
        )
        return ScenarioChatResponse(
            answer=answer,
            source_labels=["OTB snapshot", "Demand forecast"],
            intent="data_question",
            domain="scenario_lab",
            referenced_date=target_date,
        )

    if any(term in normalized for term in ["source", "data", "available"]):
        answer = (
            "I can use the selected date's forecast occupancy, booked rooms, likely retained occupancy, "
            "expected cancellations, comp-set range, market regime, pace signals, local-intel estimate, "
            "guardrails, and the latest Scenario Lab result."
        )
        return ScenarioChatResponse(answer=answer, source_labels=["Scenario Lab data scope"])

    answer = (
        f"For {target_date}, booked occupancy is {_format_pct(_raw_occupancy(state))}, "
        f"forecast occupancy is {_format_pct(forecast)}, comp median is "
        f"${_safe_float(state.get('comp_median'), state.get('competitor_price', 0.0)):.2f}, and pricing pace is "
        f"{_safe_float(state.get('pricing_pace_index'), 1.0):.2f}x. "
        "You can ask me to explain the date, classify local intel, or run a confirmed scenario."
    )
    return ScenarioChatResponse(
        answer=answer,
        source_labels=["Scenario Lab snapshot"],
        intent="data_question",
        domain="scenario_lab",
        referenced_date=target_date,
    )


def answer_horizon_risk_question(context: ScenarioChatContext) -> ScenarioChatResponse:
    risks = _rank_horizon_risks(context.horizon_records, limit=3)
    if not risks:
        summary = context.horizon_summary or {}
        dates_evaluated = int(summary.get("dates_evaluated", len(context.horizon_records) or 0))
        answer = (
            f"I do not see any dates marked for review across the {dates_evaluated}-day Scenario Lab horizon. "
            "I would still watch dates with high booked occupancy, elevated pickup, or a wide gap versus competitors."
        )
        return ScenarioChatResponse(
            answer=answer,
            source_labels=["30-day Scenario Lab risk snapshot"],
            grounding_sources=["30-day Scenario Lab risk snapshot"],
            intent="data_question",
        )

    lead = risks[0]
    lead_reason = _horizon_reason_text(lead)
    other_dates = ", ".join(row["date"] for row in risks[1:])
    answer = (
        f"The most concerning date is {lead['date']}. "
        f"It is marked {lead.get('review_status', 'for review')} with recommended ADR "
        f"${_safe_float(lead.get('recommended_adr'), 0.0):.2f}, booked occupancy "
        f"{_format_pct(lead.get('raw_otb_occupancy'))}, likely retained occupancy "
        f"{_format_pct(lead.get('adjusted_otb_occupancy'))}, forecast occupancy "
        f"{_format_pct(lead.get('forecasted_occupancy'))}, and comp median "
        f"${_safe_float(lead.get('competitor_median'), 0.0):.2f}. "
        f"Main reason: {lead_reason}"
    )
    if other_dates:
        answer += f" Also review {other_dates} next."
    return ScenarioChatResponse(
        answer=answer,
        source_labels=["30-day Scenario Lab risk snapshot", "Pricing guardrails"],
        grounding_sources=["30-day Scenario Lab risk snapshot", "Pricing guardrails"],
        intent="data_question",
    )


def answer_horizon_rank_question(message: str, context: ScenarioChatContext) -> ScenarioChatResponse:
    request = _parse_horizon_rank_request(message, context)
    if not request:
        return ScenarioChatResponse(
            answer=(
                "I can rank the Scenario Lab horizon by expected revenue at recommended ADR, revenue upside, "
                "recommended ADR, booked occupancy, likely retained occupancy, forecast occupancy, or comp median."
            ),
            source_labels=["30-day Scenario Lab ranking"],
            grounding_sources=["30-day Scenario Lab ranking"],
            intent="data_question",
            domain="scenario_lab",
        )

    records = list(context.horizon_records or [])
    metric = request["metric"]
    limit = int(request["limit"])
    direction = request["direction"]
    metric_label = _horizon_rank_metric_label(metric)
    ranked = sorted(
        [record for record in records if _optional_float(record.get(metric)) is not None],
        key=lambda record: _safe_float(record.get(metric), 0.0),
        reverse=direction == "top",
    )[:limit]

    if not ranked:
        return ScenarioChatResponse(
            answer=f"I do not have enough Scenario Lab horizon data to rank dates by {metric_label}.",
            source_labels=["30-day Scenario Lab ranking"],
            grounding_sources=["30-day Scenario Lab ranking"],
            intent="data_question",
            domain="scenario_lab",
        )

    rows = []
    for index, record in enumerate(ranked, start=1):
        rows.append(
            f"{index}. {record.get('date')}: {_format_horizon_rank_value(metric, record.get(metric))} "
            f"(ADR ${_safe_float(record.get('recommended_adr'), 0.0):.2f}, "
            f"forecast {_format_pct(record.get('forecasted_occupancy'))}, "
            f"booked {_format_pct(record.get('raw_otb_occupancy'))})"
        )
    range_text = "the next 30 days" if request.get("range") == "next_30_days" else "the Scenario Lab horizon"
    answer = f"{_rank_direction_label(direction)} {len(ranked)} dates by {metric_label} across {range_text}:\n" + "\n".join(rows)
    assumptions = []
    if request.get("metric_from_memory") or request.get("limit_from_memory"):
        assumptions.append("Used the prior ranking clarification from conversation memory.")
    if request.get("metric_defaulted"):
        assumptions.append("Ranked revenue as expected revenue at recommended ADR.")

    return ScenarioChatResponse(
        answer=answer,
        source_labels=["30-day Scenario Lab ranking"],
        grounding_sources=["30-day Scenario Lab ranking"],
        intent="data_question",
        domain="scenario_lab",
        assumptions=assumptions,
    )


def answer_pricing_strategy_question(message: str, context: ScenarioChatContext) -> ScenarioChatResponse:
    target_date = _target_date_for_question(message, context)
    state = _state_for_date(context, target_date)
    forecast = _forecast_for_date(context, target_date)
    record = _horizon_record_for_date(context, target_date)
    recommended_adr = _optional_float(record.get("recommended_adr")) if record else None
    comp_median = _safe_float(
        record.get("competitor_median") if record else None,
        _safe_float(state.get("comp_median"), state.get("competitor_price", 0.0)),
    )
    review_status = str(record.get("review_status") or "n/a") if record else "n/a"

    if recommended_adr is None:
        answer = (
            f"For {target_date}, I can see booked occupancy {_format_pct(_raw_occupancy(state))}, "
            f"likely retained occupancy {_format_pct(_adjusted_occupancy(state))}, forecast occupancy "
            f"{_format_pct(forecast)}, and comp median ${comp_median:.2f}, but I do not have a saved "
            "recommended ADR in the Scenario Lab horizon snapshot for that date."
        )
        return ScenarioChatResponse(
            answer=answer,
            source_labels=["Scenario Lab snapshot", "Demand forecast", "OTB snapshot", "Live market state"],
            grounding_sources=["Scenario Lab snapshot", "Demand forecast", "OTB snapshot", "Live market state"],
            intent="data_question",
            domain="scenario_lab",
            referenced_date=target_date,
        )

    gap_pct = ((recommended_adr - comp_median) / comp_median * 100.0) if comp_median else 0.0
    answer = (
        f"For {target_date}, recommended ADR is ${recommended_adr:.2f}. "
        f"It is {gap_pct:+.2f}% versus the comp median of ${comp_median:.2f}. "
        f"Booked occupancy is {_format_pct(_raw_occupancy(state))}, likely retained occupancy is "
        f"{_format_pct(_adjusted_occupancy(state))}, forecast occupancy is {_format_pct(forecast)}, "
        f"and expected cancellations are {_safe_float(state.get('expected_cancellations'), 0.0):.2f} rooms. "
        f"Market regime is {str(state.get('market_regime', 'n/a')).replace('_', ' ')}, "
        f"pricing pace is {_safe_float(state.get('pricing_pace_index'), 1.0):.2f}x, "
        f"and review status is {review_status}."
    )
    reasons = [str(item).rstrip(".") for item in (record.get("top_reasons") or []) if str(item).strip()] if record else []
    flags = [str(item).rstrip(".") for item in (record.get("review_flags") or []) if str(item).strip()] if record else []
    if reasons:
        answer += f" Main reason: {reasons[0]}."
    elif flags:
        answer += f" Review note: {flags[0]}."

    return ScenarioChatResponse(
        answer=answer,
        source_labels=[
            "Scenario Lab pricing recommendation",
            "30-day Scenario Lab risk snapshot",
            "Demand forecast",
            "OTB snapshot",
            "Live market state",
        ],
        grounding_sources=[
            "Scenario Lab pricing recommendation",
            "30-day Scenario Lab risk snapshot",
            "Demand forecast",
            "OTB snapshot",
            "Live market state",
        ],
        intent="data_question",
        domain="scenario_lab",
        referenced_date=target_date,
    )


def answer_forecast_audit_question(message: str = "") -> ScenarioChatResponse:
    champion_payload = _load_json_file(FORECAST_CHAMPION_PATH)
    audit_rows = _load_csv_rows(BACKTEST_AUDIT_SUMMARY_PATH)
    champion_model = str(champion_payload.get("model") or "").strip()
    requested_rank = _extract_model_rank(message)
    ranked_audit_rows = _rank_backtest_rows(audit_rows)

    if not champion_model and not ranked_audit_rows:
        return ScenarioChatResponse(
            answer=(
                "I do not have a saved current best model to answer that audit question. "
                "Run forecast backtesting first so the champion model and audit summary artifacts are available."
            ),
            source_labels=["Forecast audit scope"],
            grounding_sources=["Forecast audit scope"],
            intent="forecast_audit",
        )

    if requested_rank is not None:
        audit_row = _row_for_rank(ranked_audit_rows, requested_rank)
        champion_model = str(audit_row.get("Model") or f"rank {requested_rank}") if audit_row else champion_model
    else:
        audit_row = _find_champion_audit_row(audit_rows, champion_model)
    mae_pp = _first_available_metric(audit_row, champion_payload, "MAE_pp", "MAE")
    rmse_pp = _first_available_metric(audit_row, champion_payload, "RMSE_pp", "RMSE")
    bias_pp = _first_available_metric(audit_row, champion_payload, "Bias_pp", "Bias")
    selection_mae_pp = _optional_float(audit_row.get("Selection_Mean_Fold_MAE_pp"))
    if selection_mae_pp is None and champion_model == str(champion_payload.get("model") or "").strip():
        selection_mae_pp = _first_available_metric({}, champion_payload, "MAE_pp", "MAE")
    folds = int(_safe_float(audit_row.get("Folds"), 0.0)) if audit_row else 0
    observations = int(_safe_float(audit_row.get("Observations"), 0.0)) if audit_row else 0
    audit_status = str(
        audit_row.get("Audit_Status")
        or (champion_payload.get("backtest_metadata") or {}).get("audit_status")
        or "n/a"
    )

    if mae_pp is None:
        return ScenarioChatResponse(
            answer=(
                f"The requested model is {champion_model}, but I do not see an audit-period MAE/average "
                "occupancy miss in the saved audit summary. Run or refresh forecast backtesting to populate it."
            ),
            source_labels=["Forecast champion", "Forecast audit summary"],
            grounding_sources=["Forecast champion", "Forecast audit summary"],
            intent="forecast_audit",
        )

    rank_label = _rank_label(requested_rank) if requested_rank is not None else "current best"
    answer = (
        f"The {rank_label} model is {champion_model}. During the recent audit window, its average occupancy miss "
        f"is {mae_pp:.2f} percentage points"
    )
    if observations:
        answer += f" across {observations} forecasted stay-date observations"
    if folds:
        answer += f" from {folds} audit folds"
    answer += "."
    if selection_mae_pp is not None:
        answer += f" Its selection-backtest average miss was {selection_mae_pp:.2f} pp."
    if rmse_pp is not None:
        answer += f" Large-miss guardrail (RMSE) is {rmse_pp:.2f} pp."
    if bias_pp is not None:
        answer += f" Bias is {bias_pp:+.2f} pp."
    answer += f" Audit status: {audit_status.replace('_', ' ').title()}."

    return ScenarioChatResponse(
        answer=answer,
        source_labels=["Forecast champion", "Forecast audit summary"],
        grounding_sources=["Forecast champion", "Forecast audit summary"],
        intent="forecast_audit",
        domain="forecast_modeling",
        referenced_models=[champion_model],
        comparison_basis="audit",
    )


def answer_forecast_backtest_question() -> ScenarioChatResponse:
    champion_payload = _load_json_file(FORECAST_CHAMPION_PATH)
    comparison_rows = _load_csv_rows(MODEL_COMPARISON_PATH)
    champion_model = str(champion_payload.get("model") or "").strip()

    if not comparison_rows:
        return ScenarioChatResponse(
            answer=(
                "I do not have saved model-comparison backtest results available. "
                "Run forecast backtesting first so the leaderboard artifact is populated."
            ),
            source_labels=["Forecast backtest scope"],
            grounding_sources=["Forecast backtest scope"],
            intent="forecast_backtest",
        )

    ranked_rows = _rank_backtest_rows(comparison_rows)
    champion_row = _find_model_row(ranked_rows, champion_model) if champion_model else ranked_rows[0]
    if not champion_model:
        champion_model = str(champion_row.get("Model") or "n/a")

    model = str(champion_row.get("Model") or champion_model)
    strategy = str(champion_row.get("Strategy") or "n/a")
    profile = str(champion_row.get("Feature_Profile") or "n/a")
    folds = int(_safe_float(champion_row.get("Folds"), 0.0))
    observations = int(_safe_float(champion_row.get("Observations"), 0.0))
    mae_pp = _optional_float(champion_row.get("MAE_pp"))
    rmse_pp = _optional_float(champion_row.get("RMSE_pp"))
    bias_pp = _optional_float(champion_row.get("Bias_pp"))
    wape = _optional_float(champion_row.get("WAPE"))
    mape = _optional_float(champion_row.get("MAPE"))
    stability = _optional_float(champion_row.get("Stability"))

    answer = (
        f"The selection backtest champion is {model} ({strategy}, {profile}). "
        f"It was evaluated over {folds} folds and {observations} forecasted stay-date observations."
    )
    if mae_pp is not None:
        answer += f" Avg occupancy miss (MAE) is {mae_pp:.2f} pp."
    if rmse_pp is not None:
        answer += f" Large-miss guardrail (RMSE) is {rmse_pp:.2f} pp."
    if bias_pp is not None:
        answer += f" Bias is {bias_pp:+.2f} pp."
    if wape is not None:
        answer += f" WAPE is {wape:.2f}%."
    if mape is not None:
        answer += f" MAPE is {mape:.2f}%."
    if stability is not None:
        answer += f" Stability is {stability:.3f}."

    challengers = [row for row in ranked_rows if str(row.get("Model") or "") != model][:2]
    if challengers:
        challenger_text = "; ".join(
            f"{row.get('Model')} MAE {_safe_float(row.get('MAE_pp'), 0.0):.2f} pp"
            for row in challengers
        )
        answer += f" Next closest challengers: {challenger_text}."

    return ScenarioChatResponse(
        answer=answer,
        source_labels=["Forecast champion", "Forecast backtest leaderboard"],
        grounding_sources=["Forecast champion", "Forecast backtest leaderboard"],
        intent="forecast_backtest",
        domain="forecast_modeling",
        referenced_models=[model],
        comparison_basis="selection_backtest",
    )


def answer_forecast_ranked_audit_question(message: str) -> ScenarioChatResponse:
    return answer_forecast_audit_question(message)


def answer_forecast_model_comparison_question(limit: int = 2) -> ScenarioChatResponse:
    comparison_rows = _load_csv_rows(MODEL_COMPARISON_PATH)
    if not comparison_rows:
        return ScenarioChatResponse(
            answer=(
                "I do not have saved model-comparison backtest results available. "
                "Run forecast backtesting first so the leaderboard artifact is populated."
            ),
            source_labels=["Forecast backtest scope"],
            grounding_sources=["Forecast backtest scope"],
            intent="forecast_backtest",
        )

    ranked_rows = _rank_backtest_rows(comparison_rows)[: max(limit, 2)]
    if len(ranked_rows) < 2:
        return answer_forecast_backtest_question()

    first, second = ranked_rows[0], ranked_rows[1]
    first_model = str(first.get("Model") or "n/a")
    second_model = str(second.get("Model") or "n/a")
    first_mae = _safe_float(first.get("MAE_pp"), 0.0)
    second_mae = _safe_float(second.get("MAE_pp"), 0.0)
    mae_gap = second_mae - first_mae

    answer = (
        f"The top two selection-backtest models are {first_model} and {second_model}. "
        f"{first_model} ranks first with MAE {first_mae:.2f} pp, RMSE {_safe_float(first.get('RMSE_pp'), 0.0):.2f} pp, "
        f"WAPE {_safe_float(first.get('WAPE'), 0.0):.2f}%, bias {_safe_float(first.get('Bias_pp'), 0.0):+.2f} pp, "
        f"and stability {_safe_float(first.get('Stability'), 0.0):.3f}. "
        f"{second_model} ranks second with MAE {second_mae:.2f} pp, RMSE {_safe_float(second.get('RMSE_pp'), 0.0):.2f} pp, "
        f"WAPE {_safe_float(second.get('WAPE'), 0.0):.2f}%, bias {_safe_float(second.get('Bias_pp'), 0.0):+.2f} pp, "
        f"and stability {_safe_float(second.get('Stability'), 0.0):.3f}. "
        f"The leader is ahead by {mae_gap:.2f} pp on average occupancy miss, across "
        f"{int(_safe_float(first.get('Folds'), 0.0))} folds and "
        f"{int(_safe_float(first.get('Observations'), 0.0))} forecasted stay-date observations."
    )
    return ScenarioChatResponse(
        answer=answer,
        source_labels=["Forecast backtest leaderboard"],
        grounding_sources=["Forecast backtest leaderboard"],
        intent="forecast_backtest",
        domain="forecast_modeling",
        referenced_models=[first_model, second_model],
        comparison_basis="selection_backtest",
    )


def answer_forecast_model_audit_comparison_question(models: Optional[List[str]] = None) -> ScenarioChatResponse:
    audit_rows = _load_csv_rows(BACKTEST_AUDIT_SUMMARY_PATH)
    if not audit_rows:
        return ScenarioChatResponse(
            answer=(
                "I do not have saved audit summary results available. "
                "Run forecast backtesting first so the audit summary artifact is populated."
            ),
            source_labels=["Forecast audit scope"],
            grounding_sources=["Forecast audit scope"],
            intent="forecast_audit",
            domain="forecast_modeling",
            comparison_basis="audit",
        )

    requested_models = [model for model in (models or []) if model]
    ranked_rows = _rank_backtest_rows(audit_rows)
    selected_rows = [_find_model_row(audit_rows, model) for model in requested_models]
    selected_rows = [row for row in selected_rows if row]
    if len(selected_rows) < 2:
        selected_rows = ranked_rows[:2]
    if len(selected_rows) < 2:
        return answer_forecast_audit_question()

    first, second = selected_rows[0], selected_rows[1]
    first_model = str(first.get("Model") or "n/a")
    second_model = str(second.get("Model") or "n/a")
    first_mae = _safe_float(first.get("MAE_pp"), 0.0)
    second_mae = _safe_float(second.get("MAE_pp"), 0.0)
    mae_gap = second_mae - first_mae

    answer = (
        f"On recent audit performance, {first_model} has MAE {first_mae:.2f} pp, "
        f"RMSE {_safe_float(first.get('RMSE_pp'), 0.0):.2f} pp, WAPE {_safe_float(first.get('WAPE'), 0.0):.2f}%, "
        f"bias {_safe_float(first.get('Bias_pp'), 0.0):+.2f} pp, and stability {_safe_float(first.get('Stability'), 0.0):.3f}. "
        f"{second_model} has MAE {second_mae:.2f} pp, RMSE {_safe_float(second.get('RMSE_pp'), 0.0):.2f} pp, "
        f"WAPE {_safe_float(second.get('WAPE'), 0.0):.2f}%, bias {_safe_float(second.get('Bias_pp'), 0.0):+.2f} pp, "
        f"and stability {_safe_float(second.get('Stability'), 0.0):.3f}. "
    )
    if mae_gap >= 0:
        answer += f"{first_model} is ahead by {mae_gap:.2f} pp on audit average occupancy miss."
    else:
        answer += f"{second_model} is ahead by {abs(mae_gap):.2f} pp on audit average occupancy miss."
    observations = int(_safe_float(first.get("Observations"), 0.0))
    folds = int(_safe_float(first.get("Folds"), 0.0))
    if observations or folds:
        answer += f" The audit comparison uses {observations} observations across {folds} audit folds."

    return ScenarioChatResponse(
        answer=answer,
        source_labels=["Forecast audit summary"],
        grounding_sources=["Forecast audit summary"],
        intent="forecast_audit",
        domain="forecast_modeling",
        referenced_models=[first_model, second_model],
        comparison_basis="audit",
    )


def _run_confirmed_draft(draft: ScenarioDraft, context: ScenarioChatContext) -> ScenarioChatResponse:
    state = _state_for_date(context, draft.target_date)
    forecasted_occupancy = _forecast_for_date(context, draft.target_date)
    market_context = draft.market_context_override or _market_context_from_state(state)
    local_applied_shock = (
        _safe_float(draft.local_intel_estimate.get("suggested_shock"), 0.0)
        if draft.apply_local_intel
        else 0.0
    )
    local_applied_adr_headroom = (
        _safe_float(draft.local_intel_estimate.get("adr_headroom"), 0.0)
        if draft.apply_local_intel
        else 0.0
    )
    result = run_agentic_pricing(
        target_date=draft.target_date,
        current_occupancy=_adjusted_occupancy(state),
        forecasted_occupancy=float(forecasted_occupancy or 0.0),
        shock=draft.manual_demand_shock,
        manual_event_text=draft.local_intel_text,
        competitor_price=_safe_float(market_context.get("comp_median"), state.get("competitor_price", 120.0)),
        market_context=market_context,
        booking_velocity=_safe_float(state.get("booking_velocity"), 1.0),
        gross_pace_index=_safe_float(state.get("gross_pace_index"), state.get("booking_velocity", 1.0)),
        retained_pace_index=_safe_float(state.get("retained_pace_index"), state.get("booking_velocity", 1.0)),
        pickup_trend_index=_safe_float(state.get("pickup_trend_index"), state.get("booking_velocity", 1.0)),
        pricing_pace_index=_safe_float(state.get("pricing_pace_index"), state.get("booking_velocity", 1.0)),
        historical_avg_otb=int(_safe_float(state.get("historical_avg_otb"), 1)),
        market_state=state,
        manual_demand_shock=draft.manual_demand_shock,
        local_intel_estimate=draft.local_intel_estimate,
        local_intel_applied_shock=local_applied_shock,
        local_intel_applied_adr_headroom=local_applied_adr_headroom,
        raw_otb_occupancy=_raw_occupancy(state),
        adjusted_otb_occupancy=_adjusted_occupancy(state),
        expected_cancellations=_safe_float(state.get("expected_cancellations"), 0.0),
    )
    return ScenarioChatResponse(
        answer=_result_summary(result),
        draft=draft,
        scenario_result=result,
        source_labels=["Scenario simulation", "Local intel estimate", "Pricing guardrails"],
        ran_scenario=True,
    )


def _draft_summary(draft: ScenarioDraft) -> str:
    parts = [
        f"I prepared a scenario for {draft.target_date}.",
        f"Manual demand adjustment: {draft.manual_demand_shock * 100:+.1f}%.",
    ]
    if draft.local_intel_text:
        estimate = draft.local_intel_estimate or {}
        parts.append(
            f"Local intel classified as {estimate.get('classification', 'n/a')} with "
            f"{_safe_float(estimate.get('suggested_shock_pct'), 0.0):+.1f}% suggested demand impact "
            f"and {_safe_float(estimate.get('adr_headroom_pct'), 0.0):+.1f}% ADR headroom "
            f"at {estimate.get('confidence', 'n/a')} confidence."
        )
        if not estimate.get("apply_allowed"):
            parts.append("That local intel is context only under the current guardrails.")
    else:
        parts.append("No local-intel demand impact was supplied.")
    if draft.market_context_override:
        parts.append(
            "Market override prepared at "
            f"${_safe_float(draft.market_context_override.get('comp_low'), 0.0):.2f} / "
            f"${_safe_float(draft.market_context_override.get('comp_median'), 0.0):.2f} / "
            f"${_safe_float(draft.market_context_override.get('comp_high'), 0.0):.2f}."
        )
    return " ".join(parts)


def _result_summary(result: Dict[str, Any]) -> str:
    local_applied = _safe_float(result.get("local_intel_applied_shock"), 0.0)
    local_estimate = result.get("local_intel_estimate") or {}
    manual_event_text = str(result.get("manual_event_text") or "").strip()
    local_headroom = _safe_float(result.get("local_intel_applied_adr_headroom"), 0.0)
    has_meaningful_local_intel = bool(
        manual_event_text
        or (
            local_estimate
            and str(local_estimate.get("classification") or "").lower() not in {"irrelevant", "ambiguous"}
            and (
                _safe_float(local_estimate.get("suggested_shock"), 0.0) != 0
                or _safe_float(local_estimate.get("adr_headroom"), 0.0) != 0
                or local_estimate.get("evidence")
                or local_estimate.get("calendar_events")
            )
        )
    )
    if local_applied or local_headroom:
        intel_text = "local intel was included in priced demand as a scenario overlay"
    elif has_meaningful_local_intel:
        intel_text = "local intel was kept as context only"
    else:
        intel_text = "no extra demand overlay was included"
    action = result.get("ai_recommended_action") or result.get("strategy_applied", "Review Before Publishing")
    risk = result.get("ai_risk_level", "Medium")
    return (
        f"Scenario run complete. Final ADR is ${_safe_float(result.get('final_adr'), 0.0):.2f}; "
        f"ADR vs Reference is {_safe_float(result.get('pct_delta_from_reference'), 0.0):+.2f}% "
        f"({_money_delta(result.get('absolute_delta'))}). "
        f"Market gap is {_safe_float(result.get('competitor_gap_pct'), 0.0):+.2f}%, "
        f"pricing pace is {_safe_float(result.get('pricing_pace_index'), 1.0):.2f}x, "
        f"and {intel_text}. Recommended action: {action}. Risk: {risk}."
    )


def _confirmation_prompt(draft: ScenarioDraft) -> Optional[str]:
    if not draft.confirmation_required:
        return None
    changes = []
    if _local_intel_can_affect_price(draft):
        changes.append("apply the local-intel scenario overlay")
    if draft.market_context_override:
        changes.append("use the market override")
    if not changes:
        return None
    return "Confirm to " + " and ".join(changes) + " before I run this scenario."


def _is_horizon_risk_question(normalized: str) -> bool:
    normalized = _normalize_for_routing(normalized)
    horizon_terms = ["which date", "what date", "most concerning", "most concern", "highest risk", "riskiest", "need review", "needs review", "all dates", "next 30"]
    risk_terms = ["concern", "risk", "review", "watch", "worried", "problem"]
    return any(term in normalized for term in horizon_terms) and any(term in normalized for term in risk_terms)


def _is_forecast_audit_question(normalized: str) -> bool:
    normalized = _normalize_for_routing(normalized)
    audit_terms = ["audit", "backtest", "validation", "model performance", "current best model", "champion model"]
    miss_terms = ["occupancy miss", "average miss", "avg miss", "mae", "forecast error", "forecast miss", "performance"]
    model_terms = ["model", "forecast", "current best", "champion"]
    return (
        any(term in normalized for term in audit_terms)
        and any(term in normalized for term in miss_terms)
        and any(term in normalized for term in model_terms)
    )


def _is_forecast_audit_followup_question(normalized: str, memory: ScenarioConversationMemory) -> bool:
    normalized = _normalize_for_routing(normalized)
    if memory.last_domain != "forecast_modeling" or not memory.last_referenced_models:
        return False
    audit_terms = ["audit", "validation", "recent performance", "performance"]
    followup_terms = ["their", "these", "those", "same", "them", "what about", "how about"]
    return any(term in normalized for term in audit_terms) and any(term in normalized for term in followup_terms)


def _is_forecast_backtest_question(normalized: str) -> bool:
    normalized = _normalize_for_routing(normalized)
    backtest_terms = ["backtest", "backtesting", "selection backtest", "leaderboard", "model comparison"]
    kpi_terms = ["result", "metric", "kpi", "mae", "rmse", "wape", "mape", "bias", "accuracy", "champion", "best model"]
    return any(term in normalized for term in backtest_terms) and any(term in normalized for term in kpi_terms)


def _is_forecast_model_comparison_question(normalized: str) -> bool:
    normalized = _normalize_for_routing(normalized)
    comparison_terms = ["compare", "comparison", "top two", "top 2", "best two", "best 2"]
    model_terms = ["model", "models", "forecast", "forecasting", "occupancy"]
    ranking_terms = ["top", "best", "leader", "leaderboard", "champion"]
    return (
        any(term in normalized for term in comparison_terms)
        and any(term in normalized for term in model_terms)
        and any(term in normalized for term in ranking_terms)
    )


def _is_market_context_question(normalized: str) -> bool:
    normalized = _normalize_for_routing(normalized)
    if _has_competitor_context(normalized):
        return True
    return bool(re.search(r"\bmarket\b", normalized)) and any(
        term in normalized for term in ["rate", "rates", "position", "regime", "context"]
    )


def _has_competitor_context(normalized: str) -> bool:
    normalized = _normalize_for_routing(normalized)
    return bool(re.search(r"\b(comp(?:[-\s]?set)?|competitor(?:s)?|market rate(?:s)?)\b", normalized))


def _is_pricing_strategy_question(normalized: str) -> bool:
    normalized = _normalize_for_routing(normalized)
    pricing_terms = [
        "recommended adr",
        "recommend adr",
        "recommended price",
        "recommend price",
        "pricing strategy",
        "price strategy",
        "pricing recommendation",
        "adr recommendation",
    ]
    if any(term in normalized for term in pricing_terms):
        return True
    return bool(re.search(r"\badr\b", normalized)) and any(
        term in normalized for term in ["recommended", "recommendation", "strategy", "price"]
    )


def _target_date_for_question(message: str, context: ScenarioChatContext) -> str:
    parsed_date = _parse_date(message)
    if parsed_date:
        return parsed_date
    if context.conversation_memory.last_domain == "scenario_lab" and context.conversation_memory.last_target_date:
        return context.conversation_memory.last_target_date
    return context.target_date


def _target_date_for_scenario_action(message: str, context: ScenarioChatContext) -> str:
    parsed_date = _parse_date(message)
    if parsed_date:
        return parsed_date
    ranked_date = _resolve_ranked_date_reference(message, context)
    if ranked_date:
        return str(ranked_date["date"])
    memory = context.conversation_memory
    if memory.last_domain == "scenario_lab" and memory.last_target_date:
        return memory.last_target_date
    return context.target_date


def _horizon_record_for_date(context: ScenarioChatContext, target_date: str) -> Dict[str, Any]:
    for record in context.horizon_records:
        if str(record.get("date") or "") == target_date:
            return record
    return {}


def _is_horizon_rank_question(message: str, context: ScenarioChatContext) -> bool:
    return _parse_horizon_rank_request(message, context) is not None


def _is_ranked_scenario_action(message: str, context: ScenarioChatContext) -> bool:
    normalized = (message or "").lower()
    return (
        _resolve_ranked_date_reference(message, context) is not None
        and (
            _is_run_request(normalized)
            or _parse_demand_shock(normalized) is not None
            or "scenario" in normalized
            or "what if" in normalized
        )
    )


def _resolve_ranked_date_reference(message: str, context: ScenarioChatContext) -> Optional[Dict[str, Any]]:
    normalized = _normalize_for_routing(message)
    request = _parse_horizon_rank_request(message, context)
    if not request:
        return None
    rank_terms_present = any(
        term in normalized
        for term in ["top", "bottom", "highest", "lowest", "least", "best", "worst", "largest", "smallest"]
    )
    if not rank_terms_present and not any(term in normalized for term in ["that", "those", "same", "previous"]):
        return None

    records = list(context.horizon_records or [])
    if _excludes_selected_date(normalized):
        records = [record for record in records if str(record.get("date") or "") != context.target_date]
    metric = request["metric"]
    ranked = sorted(
        [record for record in records if _optional_float(record.get(metric)) is not None],
        key=lambda record: _safe_float(record.get(metric), 0.0),
        reverse=request["direction"] == "top",
    )
    if not ranked:
        return None
    selected = ranked[0]
    return {
        "date": str(selected.get("date") or ""),
        "metric": metric,
        "direction": request["direction"],
        "value": selected.get(metric),
        "excluded_selected_date": _excludes_selected_date(normalized),
    }


def _excludes_selected_date(normalized: str) -> bool:
    return any(
        term in normalized
        for term in [
            "not selected",
            "not the selected",
            "excluding selected",
            "exclude selected",
            "other than selected",
            "not current selected",
        ]
    )


def _parse_horizon_rank_request(message: str, context: ScenarioChatContext) -> Optional[Dict[str, Any]]:
    normalized = _normalize_for_routing(message)
    memory_request = context.conversation_memory.last_horizon_rank_request or {}
    has_explicit_rank_language = _has_explicit_rank_language(normalized)
    has_rank_language = any(
        term in normalized
        for term in ["top", "bottom", "highest", "lowest", "least", "best", "worst", "largest", "smallest", "rank", "ranking"]
    )
    has_horizon_language = any(term in normalized for term in ["next 30", "30 days", "horizon", "all dates", "days"])
    metric = _parse_horizon_rank_metric(normalized)
    is_rank_followup = bool(memory_request) and (
        bool(metric)
        or has_horizon_language
        or any(term in normalized for term in ["recommended", "adr", "revenue", "occupancy", "consider"])
    )

    if not has_rank_language and not is_rank_followup:
        return None
    if not metric and not memory_request and "revenue" not in normalized:
        return None

    direction = _parse_rank_direction(normalized) or (memory_request.get("direction") if not has_explicit_rank_language else None) or "top"
    limit = _parse_rank_limit(normalized)
    limit_from_memory = False
    if limit is None:
        if not has_explicit_rank_language and memory_request.get("limit"):
            limit = int(memory_request.get("limit") or 10)
            limit_from_memory = True
        else:
            limit = _default_rank_limit(normalized)
    limit = max(1, min(int(limit), 30))

    metric_from_memory = False
    metric_defaulted = False
    if not metric:
        metric = str(memory_request.get("metric") or "")
        metric_from_memory = bool(metric)
    if not metric and "revenue" in normalized:
        metric = "expected_revenue"
        metric_defaulted = True
    if not metric:
        return None

    return {
        "direction": direction,
        "limit": limit,
        "metric": metric,
        "range": "next_30_days" if has_horizon_language or memory_request.get("range") == "next_30_days" else "horizon",
        "metric_from_memory": metric_from_memory,
        "limit_from_memory": limit_from_memory,
        "metric_defaulted": metric_defaulted,
    }


def _has_explicit_rank_language(normalized: str) -> bool:
    return any(
        term in normalized
        for term in ["top", "bottom", "highest", "lowest", "least", "best", "worst", "largest", "smallest"]
    )


def _default_rank_limit(normalized: str) -> int:
    if any(term in normalized for term in ["highest", "lowest", "least", "best", "worst", "largest", "smallest"]):
        return 1
    return 10


def _parse_rank_direction(normalized: str) -> Optional[str]:
    if any(term in normalized for term in ["bottom", "lowest", "least", "worst", "smallest"]):
        return "bottom"
    if any(term in normalized for term in ["top", "highest", "best", "largest"]):
        return "top"
    return None


def _parse_rank_limit(normalized: str) -> Optional[int]:
    rank_terms = r"top|bottom|highest|lowest|least|best|worst"
    count_word_pattern = "|".join(NUMBER_WORDS)
    match = re.search(rf"\b(?:{rank_terms})\s+(\d{{1,2}})\b", normalized)
    if match:
        return int(match.group(1))
    match = re.search(rf"\b(\d{{1,2}})\s+(?:{rank_terms}|days|dates)\b", normalized)
    if match:
        return int(match.group(1))
    match = re.search(rf"\b(?:{rank_terms})\s+({count_word_pattern})\b", normalized)
    if match:
        return NUMBER_WORDS[match.group(1)]
    match = re.search(rf"\b({count_word_pattern})\s+(?:{rank_terms}|days|dates)\b", normalized)
    if match:
        return NUMBER_WORDS[match.group(1)]
    return None


def _parse_horizon_rank_metric(normalized: str) -> str:
    if "revenue upside" in normalized or "upside" in normalized:
        return "revenue_upside"
    if "revenue" in normalized and ("recommended" in normalized or "adr" in normalized or "forecast" in normalized):
        return "expected_revenue"
    if "recommended adr" in normalized or "adr" in normalized or "price" in normalized:
        return "recommended_adr"
    if "booked" in normalized and "occupancy" in normalized:
        return "raw_otb_occupancy"
    if "likely retained" in normalized or "retained occupancy" in normalized:
        return "adjusted_otb_occupancy"
    if "forecast" in normalized or "forecasted" in normalized or "occupancy" in normalized:
        return "forecasted_occupancy"
    if "comp median" in normalized or "competitor" in normalized:
        return "competitor_median"
    if "revenue" in normalized:
        return "expected_revenue"
    return ""


def _horizon_rank_metric_label(metric: str) -> str:
    return {
        "expected_revenue": "expected revenue at recommended ADR",
        "revenue_upside": "upside versus booked ADR",
        "recommended_adr": "recommended ADR",
        "raw_otb_occupancy": "booked occupancy",
        "adjusted_otb_occupancy": "likely retained occupancy",
        "forecasted_occupancy": "forecast occupancy",
        "competitor_median": "comp median",
    }.get(metric, metric.replace("_", " "))


def _format_horizon_rank_value(metric: str, value: Any) -> str:
    number = _safe_float(value, 0.0)
    if metric in {"expected_revenue", "revenue_upside", "recommended_adr", "competitor_median"}:
        return f"${number:,.2f}"
    if metric in {"raw_otb_occupancy", "adjusted_otb_occupancy", "forecasted_occupancy"}:
        return _format_pct(number)
    return f"{number:.2f}"


def _rank_direction_label(direction: str) -> str:
    return "Bottom" if direction == "bottom" else "Top"


def _normalize_for_routing(text: str) -> str:
    raw = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", raw)
    normalized_tokens = []
    for token in tokens:
        if token in ROUTING_TERM_ALIASES:
            normalized_tokens.append(ROUTING_TERM_ALIASES[token])
            continue
        if token in ROUTING_CANONICAL_TERMS or len(token) < 4:
            normalized_tokens.append(token)
            continue
        match = difflib.get_close_matches(token, ROUTING_CANONICAL_TERMS, n=1, cutoff=0.84)
        normalized_tokens.append(match[0] if match else token)
    return " ".join(normalized_tokens)


def _rank_horizon_risks(records: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    risk_records = [record for record in records if record.get("review_status") == "Review needed"]
    return sorted(
        risk_records,
        key=lambda item: (
            bool(item.get("manual_approval_required")),
            bool(item.get("sold_out")) and bool(item.get("material_retention_gap")),
            len(item.get("review_flags") or []),
            _safe_float(item.get("revenue_upside"), 0.0),
            str(item.get("date", "")),
        ),
        reverse=True,
    )[:limit]


def _horizon_reason_text(record: Dict[str, Any]) -> str:
    flags = [str(flag).rstrip(".") for flag in record.get("review_flags", []) if str(flag).strip()]
    if flags:
        return flags[0] + "."
    reasons = [str(reason).rstrip(".") for reason in record.get("top_reasons", []) if str(reason).strip()]
    if reasons:
        return reasons[0] + "."
    if record.get("manual_approval_required"):
        return "Manual review is required before publishing."
    return "The date has elevated review signals."


def _is_scenario_request(normalized: str) -> bool:
    return (
        _is_run_request(normalized)
        or _parse_demand_shock(normalized) is not None
        or _asks_for_price_recommendation(normalized)
        or "what if" in normalized
        or "scenario" in normalized
        or any(term in normalized for term in EVENT_TERMS)
        or any(term in normalized for term in DISRUPTION_TERMS)
        or _parse_market_override(normalized, {}) is not None
    )


def _is_run_request(normalized: str) -> bool:
    return (
        any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in RUN_TERMS)
        or _asks_for_price_recommendation(normalized)
        or (_parse_demand_shock(normalized) is not None and any(term in normalized for term in ["what happens if", "what if"]))
    )


def _asks_for_price_recommendation(normalized: str) -> bool:
    recommendation_terms = [
        "recommended price",
        "recommend price",
        "recommended adr",
        "recommend adr",
        "what price",
        "what is the price",
        "price for rooms",
        "price for room",
    ]
    has_recommendation_term = any(term in normalized for term in recommendation_terms)
    if not has_recommendation_term:
        return False
    has_scenario_modifier = (
        _parse_demand_shock(normalized) is not None
        or _parse_market_override(normalized, {}) is not None
        or "what if" in normalized
        or any(term in normalized for term in EVENT_TERMS)
        or any(term in normalized for term in DISRUPTION_TERMS)
    )
    return has_scenario_modifier


def _is_confirmation(normalized: str) -> bool:
    return any(term in normalized for term in CONFIRM_TERMS)


def _is_context_only_confirmation(normalized: str) -> bool:
    return any(term in normalized for term in CONTEXT_ONLY_TERMS)


def _parse_date(message: str) -> Optional[str]:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
    if match:
        return match.group(1)

    normalized = message.lower()
    natural_match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        normalized,
    )
    if not natural_match:
        natural_match = re.search(
            r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
            r"\s+(\d{1,2})(?:st|nd|rd|th)?\b",
            normalized,
        )
        if not natural_match:
            return None
        month_text, day_text = natural_match.group(1), natural_match.group(2)
    else:
        day_text, month_text = natural_match.group(1), natural_match.group(2)

    month = MONTH_LOOKUP.get(month_text)
    if month is None:
        return None
    year = int(getattr(DATA_END_DATE, "year", datetime.now().year))
    try:
        return datetime(year, month, int(day_text)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_demand_shock(normalized: str) -> Optional[float]:
    match = re.search(r"(?:demand|occupancy|shock)(?:\s+shock)?\s*(?:up|increase|plus|\+|by|of)?\s*(-?\d+(?:\.\d+)?)\s*%", normalized)
    if match:
        return _clamp(float(match.group(1)) / 100.0, -0.30, 0.30)
    match = re.search(r"(?:up|increase|plus|\+)\s*(\d+(?:\.\d+)?)\s*%\s*(?:demand|occupancy)?", normalized)
    if match and not _has_competitor_context(normalized):
        return _clamp(float(match.group(1)) / 100.0, -0.30, 0.30)
    match = re.search(r"(?:down|decrease|minus|-)\s*(\d+(?:\.\d+)?)\s*%\s*(?:demand|occupancy)?", normalized)
    if match and not _has_competitor_context(normalized):
        return _clamp(-float(match.group(1)) / 100.0, -0.30, 0.30)
    return None


def _parse_market_override(normalized: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _has_competitor_context(normalized):
        return None
    base = _market_context_from_state(state)
    pct_match = re.search(
        r"(?:up|increase|increases|increased|rise|rises|higher|plus|\+|by)\s*(\d+(?:\.\d+)?)\s*%",
        normalized,
    )
    direction = 1
    if not pct_match:
        pct_match = re.search(
            r"(?:down|decrease|decreases|decreased|drop|drops|lower|minus|-|by)\s*(\d+(?:\.\d+)?)\s*%",
            normalized,
        )
        direction = -1
    if not pct_match:
        pct_match = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:up|increase|increases|increased|rise|rises|higher)",
            normalized,
        )
        direction = 1
    if not pct_match:
        pct_match = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:down|decrease|decreases|decreased|drop|drops|lower)",
            normalized,
        )
        direction = -1
    if pct_match:
        factor = 1 + direction * (float(pct_match.group(1)) / 100.0)
        return {
            **base,
            "comp_low": round(_safe_float(base.get("comp_low"), 0.0) * factor, 2),
            "comp_median": round(_safe_float(base.get("comp_median"), 0.0) * factor, 2),
            "comp_high": round(_safe_float(base.get("comp_high"), 0.0) * factor, 2),
            "source_quality": "chat_market_override",
            "market_regime": "chat_market_override",
        }
    median_match = re.search(
        r"(?:median|competitor|comp)\s*(?:rate|price|median)?\s*(?:is|to|at|=)\s*\$?\s*(\d{2,4}(?:\.\d+)?)\b",
        normalized,
    )
    if median_match:
        median = float(median_match.group(1))
        return {
            **base,
            "comp_median": round(median, 2),
            "source_quality": "chat_market_override",
            "market_regime": "chat_market_override",
        }
    return None


def _extract_local_intel_text(message: str) -> str:
    normalized = message.lower()
    local_intel_event_terms = EVENT_TERMS - {"person", "people", "guest", "rooms", "pax"}
    has_group_size = re.search(r"\b\d{2,4}\s*(-|\s)?(person|people|guest|room|rooms|pax)\b", normalized)
    if not (
        any(term in normalized for term in local_intel_event_terms)
        or any(term in normalized for term in DISRUPTION_TERMS)
        or has_group_size
    ):
        return ""
    cleaned = re.sub(
        r"\b(run|simulate|scenario|what if|please|can you|for \d{4}-\d{2}-\d{2}|for)\b",
        " ",
        message,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or message.strip()


def _market_context_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    competitor = _safe_float(state.get("competitor_price"), 120.0)
    return {
        "comp_low": _safe_float(state.get("comp_low"), competitor),
        "comp_median": _safe_float(state.get("comp_median"), competitor),
        "comp_high": _safe_float(state.get("comp_high"), competitor),
        "sample_size": int(_safe_float(state.get("sample_size"), 1)),
        "source_quality": state.get("source_quality"),
        "market_regime": state.get("market_regime"),
        "market_as_of_timestamp": state.get("market_as_of_timestamp"),
    }


def _local_intel_can_affect_price(draft: ScenarioDraft) -> bool:
    return _local_estimate_can_affect_price(draft.local_intel_estimate)


def _local_estimate_can_affect_price(estimate: Dict[str, Any]) -> bool:
    return bool(
        estimate
        and estimate.get("apply_allowed")
        and (
            _safe_float(estimate.get("suggested_shock"), 0.0) != 0
            or _safe_float(estimate.get("adr_headroom"), 0.0) != 0
        )
    )


def _copy_draft(draft: ScenarioDraft) -> ScenarioDraft:
    return ScenarioDraft(
        target_date=draft.target_date,
        manual_demand_shock=draft.manual_demand_shock,
        local_intel_text=draft.local_intel_text,
        local_intel_estimate=dict(draft.local_intel_estimate),
        market_context_override=dict(draft.market_context_override) if draft.market_context_override else None,
        apply_local_intel=draft.apply_local_intel,
        confirmation_required=draft.confirmation_required,
        confirmed=draft.confirmed,
        requested_run=draft.requested_run,
    )


def _draft_from_pending_memory(context: ScenarioChatContext) -> Optional[ScenarioDraft]:
    memory = context.conversation_memory
    if not getattr(memory, "last_draft_pending", False):
        return None
    has_price_changing_input = (
        memory.last_manual_demand_shock not in (None, 0, 0.0)
        or bool(memory.last_market_context_override)
        or bool(memory.last_local_intel_text)
    )
    if not has_price_changing_input:
        return None

    target_date = memory.last_target_date or context.target_date
    local_estimate: Dict[str, Any] = {}
    if memory.last_local_intel_text:
        scenario_state = _state_for_date(context, target_date)
        local_estimate = estimate_local_intel_impact(
            memory.last_local_intel_text,
            current_occ=_adjusted_occupancy(scenario_state),
            forecast_occ=_forecast_for_date(context, target_date),
            booking_velocity=_safe_float(scenario_state.get("booking_velocity"), 1.0),
            retained_pace_index=_safe_float(
                scenario_state.get("retained_pace_index"),
                _safe_float(scenario_state.get("booking_velocity"), 1.0),
            ),
            pickup_trend_index=_safe_float(
                scenario_state.get("pickup_trend_index"),
                _safe_float(scenario_state.get("booking_velocity"), 1.0),
            ),
            target_date=target_date,
            market_context=_market_context_from_state(scenario_state),
        )

    return ScenarioDraft(
        target_date=target_date,
        manual_demand_shock=float(memory.last_manual_demand_shock or 0.0),
        local_intel_text=memory.last_local_intel_text,
        local_intel_estimate=local_estimate,
        market_context_override=dict(memory.last_market_context_override) if memory.last_market_context_override else None,
        apply_local_intel=False,
        confirmation_required=(
            _local_estimate_can_affect_price(local_estimate) or bool(memory.last_market_context_override)
        ),
        confirmed=False,
        requested_run=True,
    )


def update_conversation_memory(
    memory: ScenarioConversationMemory,
    user_message: str,
    response: ScenarioChatResponse,
) -> ScenarioConversationMemory:
    updated = ScenarioConversationMemory(
        rolling_summary=memory.rolling_summary,
        last_user_message=user_message,
        last_intent=response.intent or memory.last_intent,
        last_domain=response.domain or memory.last_domain,
        last_target_date=memory.last_target_date,
        last_referenced_models=list(memory.last_referenced_models),
        last_comparison_basis=memory.last_comparison_basis,
        last_local_intel_text=memory.last_local_intel_text,
        last_manual_demand_shock=memory.last_manual_demand_shock,
        last_market_context_override=(
            dict(memory.last_market_context_override)
            if memory.last_market_context_override
            else None
        ),
        last_horizon_rank_request=dict(memory.last_horizon_rank_request),
        last_draft_pending=getattr(memory, "last_draft_pending", False),
        last_scenario_result=dict(memory.last_scenario_result),
        previous_scenario_result=dict(memory.previous_scenario_result),
        last_sources=list(response.grounding_sources or response.source_labels or memory.last_sources),
    )

    if response.draft:
        updated.last_target_date = response.draft.target_date
        updated.last_local_intel_text = response.draft.local_intel_text
        updated.last_manual_demand_shock = response.draft.manual_demand_shock
        updated.last_market_context_override = (
            dict(response.draft.market_context_override)
            if response.draft.market_context_override
            else None
        )
        updated.last_draft_pending = not response.ran_scenario

    if response.referenced_models:
        updated.last_referenced_models = list(response.referenced_models)
    if response.comparison_basis:
        updated.last_comparison_basis = response.comparison_basis
    if response.domain:
        updated.last_domain = response.domain
    if response.referenced_date:
        updated.last_target_date = response.referenced_date
    parsed_user_shock = _parse_demand_shock((user_message or "").lower())
    if parsed_user_shock is not None:
        updated.last_manual_demand_shock = parsed_user_shock
    rank_request = _parse_horizon_rank_request(user_message, ScenarioChatContext(
        target_date=updated.last_target_date or "",
        forecasted_occupancy=0.0,
        current_state={},
        conversation_memory=memory,
    ))
    if rank_request:
        updated.last_horizon_rank_request = {
            "direction": rank_request["direction"],
            "limit": rank_request["limit"],
            "metric": rank_request["metric"],
            "range": rank_request["range"],
        }

    if response.scenario_result:
        updated.last_draft_pending = False
        if updated.last_scenario_result:
            updated.previous_scenario_result = dict(updated.last_scenario_result)
        updated.last_scenario_result = _scenario_result_memory(response.scenario_result)
        if updated.last_scenario_result.get("target_date"):
            updated.last_target_date = str(updated.last_scenario_result["target_date"])

    updated.rolling_summary = _build_memory_summary(updated)
    return updated


def _scenario_result_memory(result: Dict[str, Any]) -> Dict[str, Any]:
    if not result:
        return {}
    return {
        "target_date": result.get("target_date"),
        "final_adr": result.get("final_adr"),
        "adr_vs_reference_pct": result.get("pct_delta_from_reference"),
        "adr_vs_reference_amount": result.get("absolute_delta"),
        "market_gap_pct": result.get("competitor_gap_pct"),
        "pricing_pace_index": result.get("pricing_pace_index"),
        "local_intel_applied_shock": result.get("local_intel_applied_shock"),
        "recommended_action": result.get("ai_recommended_action") or result.get("strategy_applied"),
        "risk_level": result.get("ai_risk_level"),
    }


def _build_memory_summary(memory: ScenarioConversationMemory) -> str:
    parts = []
    if memory.last_target_date:
        parts.append(f"Last scenario date: {memory.last_target_date}.")
    if memory.last_market_context_override:
        comp_median = memory.last_market_context_override.get("comp_median")
        if comp_median is not None:
            parts.append(f"Last market override used comp median ${float(comp_median):.2f}.")
    if memory.last_local_intel_text:
        parts.append(f"Last local intel: {memory.last_local_intel_text}.")
    if memory.last_scenario_result.get("final_adr") is not None:
        parts.append(f"Last scenario ADR: ${float(memory.last_scenario_result['final_adr']):.2f}.")
    if memory.last_referenced_models:
        parts.append(f"Last forecast models discussed: {', '.join(memory.last_referenced_models)}.")
    if memory.last_comparison_basis:
        parts.append(f"Last forecast comparison basis: {memory.last_comparison_basis}.")
    if memory.last_horizon_rank_request:
        rank_request = memory.last_horizon_rank_request
        parts.append(
            "Last Scenario Lab ranking request: "
            f"{rank_request.get('direction', 'top')} {rank_request.get('limit', 10)} by "
            f"{_horizon_rank_metric_label(str(rank_request.get('metric', 'expected_revenue')))}."
        )
    return " ".join(parts)


def _load_json_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _find_champion_audit_row(rows: List[Dict[str, str]], champion_model: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("Model") or "").strip() == champion_model:
            return row
    for row in rows:
        if str(row.get("Is_Champion") or "").strip().lower() in {"true", "1", "yes"}:
            return row
    return {}


def _find_model_row(rows: List[Dict[str, str]], model: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("Model") or "").strip() == model:
            return row
    return rows[0] if rows else {}


def _rank_backtest_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _safe_float(row.get("MAE_pp"), float("inf")),
            _safe_float(row.get("RMSE_pp"), float("inf")),
            _safe_float(row.get("WAPE"), float("inf")),
            str(row.get("Model") or ""),
        ),
    )


def _row_for_rank(rows: List[Dict[str, str]], rank: int) -> Dict[str, str]:
    if rank < 1 or rank > len(rows):
        return {}
    return rows[rank - 1]


def _extract_model_rank(message: str) -> Optional[int]:
    normalized = _normalize_for_routing(message)
    rank_patterns = [
        (r"\b(?:2nd|second)\s+(?:best\s+)?model\b", 2),
        (r"\b(?:3rd|third)\s+(?:best\s+)?model\b", 3),
        (r"\b(?:4th|fourth)\s+(?:best\s+)?model\b", 4),
        (r"\b(?:5th|fifth)\s+(?:best\s+)?model\b", 5),
        (r"\b(?:1st|first|top|best|champion)\s+(?:best\s+)?model\b", 1),
    ]
    for pattern, rank in rank_patterns:
        if re.search(pattern, normalized):
            return rank
    numeric_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:best\s+)?model\b", normalized)
    if numeric_match:
        rank = int(numeric_match.group(1))
        return rank if rank > 0 else None
    return None


def _rank_label(rank: Optional[int]) -> str:
    labels = {
        1: "current best",
        2: "second-best",
        3: "third-best",
        4: "fourth-best",
        5: "fifth-best",
    }
    if rank is None:
        return "current best"
    return labels.get(rank, f"rank-{rank}")


def _first_available_metric(
    audit_row: Dict[str, Any],
    champion_payload: Dict[str, Any],
    pp_name: str,
    raw_name: str,
) -> Optional[float]:
    audit_pp = _optional_float(audit_row.get(pp_name))
    if audit_pp is not None:
        return audit_pp
    audit_raw = _optional_float(audit_row.get(raw_name))
    if audit_raw is not None:
        return audit_raw * 100

    metrics = champion_payload.get("metrics") or {}
    champion_pp = _optional_float(metrics.get(pp_name))
    if champion_pp is not None:
        return champion_pp
    champion_raw = _optional_float(metrics.get(raw_name))
    if champion_raw is not None:
        return champion_raw * 100
    return None


def _raw_occupancy(state: Dict[str, Any]) -> float:
    if state.get("raw_otb_occupancy") is not None:
        return _safe_float(state.get("raw_otb_occupancy"), 0.0)
    total_rooms = max(_safe_float(state.get("total_rooms"), BASE_CAPACITY), 1.0)
    return _safe_float(state.get("current_otb"), 0.0) / total_rooms


def _adjusted_occupancy(state: Dict[str, Any]) -> float:
    if state.get("adjusted_otb_occupancy") is not None:
        return _safe_float(state.get("adjusted_otb_occupancy"), 0.0)
    total_rooms = max(_safe_float(state.get("total_rooms"), BASE_CAPACITY), 1.0)
    return _safe_float(state.get("adjusted_otb", state.get("current_otb", 0.0)), 0.0) / total_rooms


def _state_for_date(context: ScenarioChatContext, target_date: str) -> Dict[str, Any]:
    return context.live_market_by_date.get(target_date) or context.current_state


def _forecast_for_date(context: ScenarioChatContext, target_date: str) -> float:
    return _safe_float(
        context.forecast_occupancy_by_date.get(target_date),
        _safe_float(context.forecasted_occupancy, 0.0),
    )


def _format_pct(value: Any) -> str:
    return f"{_safe_float(value, 0.0) * 100:.1f}%"


def _money_delta(value: Any) -> str:
    amount = _safe_float(value, 0.0)
    return f"${amount:+.2f}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
