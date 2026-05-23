"""Grounded DeepSeek/LangGraph Scenario Lab copilot.

DeepSeek handles language understanding and final wording. Deterministic
Scenario Lab tools still own scenario execution, ADR, and confirmation gates.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from config import BASE_DIR, CHAT_MODEL
from copilot_core.pricing_agent import _get_client, _resolve_api_key
from pricing_core.local_intel import estimate_local_intel_impact
from copilot_core.scenario_copilot import (
    ScenarioChatContext,
    ScenarioChatResponse,
    ScenarioConversationMemory,
    ScenarioDraft,
    answer_forecast_model_audit_comparison_question,
    answer_forecast_backtest_question,
    answer_forecast_model_comparison_question,
    answer_scenario_question,
    answer_forecast_audit_question,
    answer_horizon_risk_question,
    answer_horizon_rank_question,
    build_scenario_draft,
    handle_scenario_chat,
    _adjusted_occupancy,
    _forecast_for_date,
    _format_pct,
    _is_confirmation,
    _is_context_only_confirmation,
    _is_forecast_audit_question,
    _is_forecast_audit_followup_question,
    _is_forecast_backtest_question,
    _is_forecast_model_comparison_question,
    _is_horizon_rank_question,
    _is_pricing_strategy_question,
    _is_ranked_scenario_action,
    _market_context_from_state,
    _parse_date,
    _raw_occupancy,
    _safe_float,
    _horizon_record_for_date,
    _state_for_date,
)


SCENARIO_COPILOT_PROMPT_PATH = os.path.join(BASE_DIR, "src", "prompts", "scenario_copilot.txt")
ALLOWED_INTENTS = {
    "data_question",
    "scenario_draft",
    "run_simulation",
    "explain_result",
    "local_intel",
    "forecast_audit",
    "forecast_backtest",
    "unsupported",
}
PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"reveal (the )?(system|developer|hidden) prompt",
    r"show (the )?(system|developer|hidden) prompt",
    r"bypass (approval|confirmation|guardrail|rules)",
    r"override (approval|confirmation|guardrail|rules)",
    r"forget (the )?(rules|instructions)",
    r"act as (?:an? )?(unrestricted|jailbroken)",
    r"invent (data|adr|price|occupancy)",
]
TECHNICAL_LEAK_TERMS = [
    "raw otb",
    "candidate optimizer",
    "optimizer internals",
    "json",
    "system prompt",
    "developer prompt",
]
PRICE_ACTION_LEAK_TERMS = [
    "replacement adr",
    "set adr",
    "set the adr",
    "change adr to",
    "change the adr to",
    "publish adr",
    "publish the adr",
    "override confirmation",
    "skip confirmation",
]
PROTECTED_BUSINESS_TERMS = {
    "final adr": ("final adr", "final_adr"),
    "recommended adr": ("recommended adr", "recommended_adr"),
    "adr vs reference": ("adr vs reference", "adr_vs_reference"),
    "comp median": ("comp median", "comp_median", "competitor_median"),
    "comp set": ("comp set", "comp_low", "comp_median", "comp_high"),
    "booked occupancy": ("booked occupancy", "booked_occupancy", "raw_otb_occupancy"),
    "likely retained occupancy": ("likely retained occupancy", "likely_retained_occupancy", "adjusted_otb_occupancy"),
    "forecast occupancy": ("forecast occupancy", "forecast_occupancy", "forecasted_occupancy"),
    "cancellations": ("cancellations", "expected_cancellations"),
    "pricing pace": ("pricing pace", "pricing_pace", "pricing_pace_index"),
    "market regime": ("market regime", "market_regime"),
    "revenue upside": ("revenue upside", "revenue_upside"),
    "review needed": ("review needed", "review_status"),
    "manual approval": ("manual approval", "manual_approval_required"),
    "mae": ("mae", "mae_pp"),
    "rmse": ("rmse", "rmse_pp"),
    "wape": ("wape",),
    "mape": ("mape",),
    "bias": ("bias", "bias_pp"),
    "accuracy": ("accuracy",),
    "stability": ("stability",),
    "folds": ("folds",),
    "observations": ("observations",),
}
MONEY_FIELD_HINTS = ("adr", "price", "comp", "revenue", "delta", "upside", "amount")
PERCENT_FIELD_HINTS = ("pct", "percent", "occupancy", "shock", "gap")
RATIO_FIELD_HINTS = ("pace", "ratio", "velocity", "index")
KPI_FIELD_HINTS = (
    "mae",
    "rmse",
    "wape",
    "mape",
    "bias",
    "accuracy",
    "stability",
    "volatility",
    "folds",
    "observations",
)


class ScenarioLLMState(TypedDict, total=False):
    message: str
    context: ScenarioChatContext
    pending_draft: Optional[ScenarioDraft]
    safety_flags: List[str]
    blocked: bool
    classification: Dict[str, Any]
    grounded_context: Dict[str, Any]
    tool_response: ScenarioChatResponse
    final_response: ScenarioChatResponse


def handle_grounded_scenario_chat(
    message: str,
    context: ScenarioChatContext,
    pending_draft: Optional[ScenarioDraft] = None,
) -> ScenarioChatResponse:
    """Run the grounded LLM copilot, falling back to deterministic chat."""
    normalized = (message or "").strip().lower()
    if pending_draft and (_is_confirmation(normalized) or _is_context_only_confirmation(normalized)):
        response = handle_scenario_chat(message, context, pending_draft)
        response.intent = response.intent or "run_simulation"
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response

    if not _resolve_api_key():
        fallback = handle_scenario_chat(message, context, pending_draft)
        fallback.safety_flags.append("DeepSeek unavailable; deterministic Scenario Copilot fallback used.")
        fallback.intent = "deterministic_fallback"
        return fallback

    try:
        graph = _create_scenario_llm_graph()
        state = graph.invoke(
            {
                "message": message,
                "context": context,
                "pending_draft": pending_draft,
                "safety_flags": [],
                "blocked": False,
            }
        )
        response = state.get("final_response")
        if response:
            return response
    except Exception:
        fallback = handle_scenario_chat(message, context, pending_draft)
        fallback.safety_flags.append("LLM copilot unavailable; deterministic fallback used.")
        fallback.intent = "deterministic_fallback"
        return fallback

    fallback = handle_scenario_chat(message, context, pending_draft)
    fallback.safety_flags.append("LLM copilot returned no usable response; deterministic fallback used.")
    fallback.intent = "deterministic_fallback"
    return fallback


@lru_cache(maxsize=1)
def _create_scenario_llm_graph():
    builder = StateGraph(ScenarioLLMState)
    builder.add_node("sanitize_input", _sanitize_input_node)
    builder.add_node("classify_intent", _classify_intent_node)
    builder.add_node("route_tool", _route_tool_node)
    builder.add_node("generate_response", _generate_response_node)
    builder.set_entry_point("sanitize_input")
    builder.add_edge("sanitize_input", "classify_intent")
    builder.add_edge("classify_intent", "route_tool")
    builder.add_edge("route_tool", "generate_response")
    builder.add_edge("generate_response", END)
    return builder.compile()


def _sanitize_input_node(state: ScenarioLLMState) -> ScenarioLLMState:
    message = str(state.get("message") or "")
    safety_flags = list(state.get("safety_flags") or [])
    detected = detect_prompt_injection(message)
    if detected:
        safety_flags.extend(detected)
        response = ScenarioChatResponse(
            answer=(
                "I cannot follow instructions that bypass Scenario Lab guardrails or ask for hidden prompts. "
                "I can still answer grounded questions about the selected date, local intel, market position, "
                "pace, or a confirmed scenario."
            ),
            source_labels=["Scenario Lab guardrails"],
            grounding_sources=["Scenario Lab guardrails"],
            safety_flags=safety_flags,
            intent="unsupported",
        )
        return {**state, "safety_flags": safety_flags, "blocked": True, "final_response": response}
    return {**state, "safety_flags": safety_flags, "blocked": False}


def _classify_intent_node(state: ScenarioLLMState) -> ScenarioLLMState:
    if state.get("blocked"):
        return state
    context = state["context"]
    message = str(state.get("message") or "")
    pending_draft = state.get("pending_draft")
    grounded_context = build_grounded_context(message, context, pending_draft)
    classification = classify_scenario_intent(message, grounded_context)
    return {**state, "classification": classification, "grounded_context": grounded_context}


def _route_tool_node(state: ScenarioLLMState) -> ScenarioLLMState:
    if state.get("blocked"):
        return state
    context = state["context"]
    message = str(state.get("message") or "")
    pending_draft = state.get("pending_draft")
    classification = state.get("classification") or {}
    tool_response = route_scenario_tool(message, context, pending_draft, classification)
    return {**state, "tool_response": tool_response}


def _generate_response_node(state: ScenarioLLMState) -> ScenarioLLMState:
    if state.get("blocked"):
        return state
    tool_response = state.get("tool_response")
    if not tool_response:
        return state
    context = state["context"]
    grounded_context = state.get("grounded_context") or build_grounded_context(
        str(state.get("message") or ""),
        context,
        state.get("pending_draft"),
    )
    final_response = generate_grounded_response(
        str(state.get("message") or ""),
        grounded_context,
        tool_response,
        state.get("classification") or {},
        state.get("safety_flags") or [],
    )
    return {**state, "final_response": final_response}


def classify_scenario_intent(message: str, grounded_context: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _load_copilot_prompt()
    user_payload = {
        "task": "classify_intent",
        "user_message": message,
        "grounded_context": grounded_context,
        "required_json_schema": {
            "intent": "one allowed intent",
            "target_date": "YYYY-MM-DD or empty",
            "needs_clarification": "boolean",
            "clarification_question": "string or empty",
            "assumptions": ["short assumption strings"],
            "tool": "one allowed tool",
        },
    }
    response = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return validate_intent_payload(json.loads(response.choices[0].message.content), grounded_context)


def validate_intent_payload(payload: Dict[str, Any], grounded_context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Intent payload must be a JSON object.")
    intent = str(payload.get("intent") or "unsupported").strip().lower()
    if intent not in ALLOWED_INTENTS:
        intent = "unsupported"
    target_date = str(payload.get("target_date") or "").strip()
    available_dates = set(grounded_context.get("available_dates") or [])
    selected_date = grounded_context.get("selected_date")
    requested_date = str(grounded_context.get("requested_date") or "").strip()
    date_anchor_source = str(grounded_context.get("date_anchor_source") or "").strip()
    if target_date and target_date not in available_dates and target_date != selected_date:
        target_date = ""
    if (
        requested_date
        and requested_date != selected_date
        and date_anchor_source in {"conversation_memory", "pending_draft", "latest_result"}
        and (not target_date or target_date == selected_date)
    ):
        target_date = requested_date
    assumptions = payload.get("assumptions") or []
    if not isinstance(assumptions, list):
        assumptions = [str(assumptions)]
    assumptions = _sanitize_date_assumptions(
        [str(item) for item in assumptions if str(item).strip()],
        selected_date=str(selected_date or ""),
        target_date=target_date,
        date_anchor_source=date_anchor_source,
    )
    clarification = str(payload.get("clarification_question") or "").strip()
    return {
        "intent": intent,
        "target_date": target_date,
        "needs_clarification": bool(payload.get("needs_clarification")) and bool(clarification),
        "clarification_question": clarification,
        "assumptions": assumptions[:3],
        "tool": _validated_tool(payload.get("tool"), intent),
    }


def route_scenario_tool(
    message: str,
    context: ScenarioChatContext,
    pending_draft: Optional[ScenarioDraft],
    classification: Dict[str, Any],
) -> ScenarioChatResponse:
    intent = classification.get("intent", "unsupported")
    assumptions = list(classification.get("assumptions") or [])
    if _is_ranked_scenario_action(message, context):
        response = handle_scenario_chat(message, context, pending_draft)
        response.intent = response.intent or "run_simulation"
        response.domain = response.domain or "scenario_lab"
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response
    if (
        classification.get("needs_clarification")
        and context.conversation_memory.last_horizon_rank_request
        and _is_horizon_rank_question(message, context)
    ):
        response = answer_horizon_rank_question(message, context)
        response.intent = intent
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response
    if classification.get("needs_clarification") and context.clarification_count < 1:
        question = classification.get("clarification_question") or "Which stay date should I use?"
        return ScenarioChatResponse(
            answer=question,
            clarification_question=question,
            assumptions=assumptions,
            source_labels=["Scenario Copilot clarification"],
            grounding_sources=["Scenario Copilot clarification"],
            intent=intent,
        )
    if classification.get("needs_clarification") and context.clarification_count >= 1:
        assumptions.append(f"Used selected Scenario Lab date {context.target_date}.")

    if _is_forecast_audit_question(message.lower()):
        response = answer_forecast_audit_question(message)
        response.intent = "forecast_audit"
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response
    if _is_forecast_audit_followup_question(message.lower(), context.conversation_memory):
        response = answer_forecast_model_audit_comparison_question(context.conversation_memory.last_referenced_models)
        response.intent = "forecast_audit"
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response
    if _is_forecast_model_comparison_question(message.lower()):
        response = answer_forecast_model_comparison_question()
        response.intent = "forecast_backtest"
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response
    if _is_forecast_backtest_question(message.lower()):
        response = answer_forecast_backtest_question()
        response.intent = "forecast_backtest"
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response

    if _is_horizon_rank_question(message, context):
        response = answer_horizon_rank_question(message, context)
        response.intent = intent
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response

    if intent == "unsupported":
        return ScenarioChatResponse(
            answer="I can help with Scenario Lab data, local intel, market position, pace, and confirmed simulations.",
            assumptions=assumptions,
            source_labels=["Scenario Lab data scope"],
            grounding_sources=["Scenario Lab data scope"],
            intent=intent,
        )

    enriched_message = _enrich_message_for_deterministic_tools(message, classification)
    if _is_horizon_rank_question(enriched_message, context):
        response = answer_horizon_rank_question(enriched_message, context)
        response.intent = intent
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response
    if _is_horizon_question(message, classification):
        response = answer_horizon_risk_question(context)
        response.intent = intent
        response.assumptions.extend(assumptions)
        response.grounding_sources.extend(_grounding_sources_for_response(response))
        return response
    if intent in {"data_question", "explain_result"}:
        response = answer_scenario_question(enriched_message, context, pending_draft)
    else:
        response = handle_scenario_chat(enriched_message, context, pending_draft)
        response = _apply_memory_to_followup_response(message, context, response)
    response.intent = intent
    response.domain = response.domain or "scenario_lab"
    if not response.referenced_date and classification.get("target_date"):
        response.referenced_date = str(classification["target_date"])
    response.assumptions.extend(assumptions)
    response.grounding_sources.extend(_grounding_sources_for_response(response))
    return response


def generate_grounded_response(
    message: str,
    grounded_context: Dict[str, Any],
    tool_response: ScenarioChatResponse,
    classification: Dict[str, Any],
    safety_flags: List[str],
) -> ScenarioChatResponse:
    if tool_response.clarification_question or tool_response.confirmation_prompt or tool_response.ran_scenario:
        return _finalize_response(tool_response, safety_flags)
    if "Scenario Lab pricing recommendation" in set(tool_response.source_labels or []) | set(tool_response.grounding_sources or []):
        return _finalize_response(tool_response, safety_flags)
    if "30-day Scenario Lab ranking" in set(tool_response.source_labels or []) | set(tool_response.grounding_sources or []):
        return _finalize_response(tool_response, safety_flags)
    deterministic_forecast_sources = {
        "Forecast audit summary",
        "Forecast backtest leaderboard",
    }
    if deterministic_forecast_sources & (set(tool_response.source_labels or []) | set(tool_response.grounding_sources or [])):
        return _finalize_response(tool_response, safety_flags)

    prompt = _load_copilot_prompt()
    user_payload = {
        "task": "generate_grounded_response",
        "user_message": message,
        "intent": classification,
        "grounded_context": grounded_context,
        "tool_result": _response_to_grounded_payload(tool_response),
        "required_json_schema": {
            "answer": "manager-facing answer grounded only in tool_result and grounded_context",
            "sources": ["source labels used"],
            "assumptions": ["assumptions stated"],
        },
    }
    try:
        response = _get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        payload = json.loads(response.choices[0].message.content)
        answer = str(payload.get("answer") or "").strip()
        sources = payload.get("sources") or []
        assumptions = payload.get("assumptions") or []
        if not answer:
            return _finalize_response(tool_response, safety_flags)
        if _answer_needs_fallback(answer, tool_response, grounded_context):
            tool_response.safety_flags.append("LLM answer failed grounding checks; deterministic answer used.")
            return _finalize_response(tool_response, safety_flags)
        tool_response.answer = answer
        tool_response.source_labels = _merge_unique(tool_response.source_labels, [str(item) for item in sources])
        tool_response.assumptions = _merge_unique(tool_response.assumptions, [str(item) for item in assumptions])
        return _finalize_response(tool_response, safety_flags)
    except Exception:
        tool_response.safety_flags.append("LLM response generation failed; deterministic answer used.")
        return _finalize_response(tool_response, safety_flags)


def build_grounded_context(
    message: str,
    context: ScenarioChatContext,
    pending_draft: Optional[ScenarioDraft] = None,
) -> Dict[str, Any]:
    requested_date, date_anchor_source = _resolve_requested_date(message, context, pending_draft)
    state = _state_for_date(context, requested_date)
    forecast = _forecast_for_date(context, requested_date)
    market = _market_context_from_state(state)
    latest = context.latest_result or {}
    horizon_record = _horizon_record_for_date(context, requested_date)
    available_dates = set(context.live_market_by_date.keys()) | {context.target_date}
    available_dates.update(str(record.get("date")) for record in context.horizon_records if record.get("date"))
    return {
        "selected_date": context.target_date,
        "requested_date": requested_date,
        "date_anchor_source": date_anchor_source,
        "available_dates": sorted(available_dates),
        "forecast_occupancy_pct": round(_safe_float(forecast) * 100, 1),
        "booked_occupancy_pct": round(_raw_occupancy(state) * 100, 1),
        "likely_retained_occupancy_pct": round(_adjusted_occupancy(state) * 100, 1),
        "expected_cancellations": round(_safe_float(state.get("expected_cancellations"), 0.0), 2),
        "current_booked_rooms": _safe_float(state.get("current_otb"), 0.0),
        "market_context": {
            "comp_low": market.get("comp_low"),
            "comp_median": market.get("comp_median"),
            "comp_high": market.get("comp_high"),
            "market_regime": market.get("market_regime"),
            "source_quality": market.get("source_quality"),
        },
        "pace": {
            "booked_pace": _safe_float(state.get("gross_pace_index"), 1.0),
            "likely_retained_pace": _safe_float(state.get("retained_pace_index"), 1.0),
            "recent_pickup": _safe_float(state.get("pickup_trend_index"), 1.0),
            "pricing_pace": _safe_float(state.get("pricing_pace_index"), 1.0),
        },
        "pending_draft": _draft_payload(pending_draft),
        "latest_result": _latest_result_payload(latest),
        "requested_date_pricing_recommendation": _compact_horizon_record(horizon_record) if horizon_record else {},
        "conversation_memory": _memory_payload(context.conversation_memory),
        "horizon_summary": context.horizon_summary,
        "top_risk_dates": _horizon_risk_payload(context.horizon_records, limit=5),
        "top_opportunity_dates": _horizon_opportunity_payload(context.horizon_records, limit=5),
        "source_labels": [
            "Demand forecast",
            "OTB snapshot",
            "Live market state",
            "Scenario Lab state",
            "30-day Scenario Lab risk snapshot",
        ],
    }


def _resolve_requested_date(
    message: str,
    context: ScenarioChatContext,
    pending_draft: Optional[ScenarioDraft] = None,
) -> tuple[str, str]:
    if _mentions_selected_date(message):
        return context.target_date, "selected_date"
    parsed_date = _parse_date(message)
    if parsed_date:
        return parsed_date, "explicit_message"
    if pending_draft:
        return pending_draft.target_date, "pending_draft"
    memory = context.conversation_memory
    if memory.last_domain == "scenario_lab" and memory.last_target_date:
        return memory.last_target_date, "conversation_memory"
    latest_target = str((context.latest_result or {}).get("target_date") or "").strip()
    if latest_target:
        return latest_target, "latest_result"
    return context.target_date, "selected_date"


def _mentions_selected_date(message: str) -> bool:
    normalized = (message or "").lower()
    return any(term in normalized for term in ["selected date", "sidebar date", "current selected date"])


def _sanitize_date_assumptions(
    assumptions: List[str],
    *,
    selected_date: str,
    target_date: str,
    date_anchor_source: str,
) -> List[str]:
    if not selected_date or not target_date or target_date == selected_date:
        return assumptions
    if date_anchor_source not in {"conversation_memory", "pending_draft", "latest_result"}:
        return assumptions

    cleaned: List[str] = []
    dropped_selected_date_assumption = False
    for assumption in assumptions:
        lowered = assumption.lower()
        if selected_date in assumption and "selected date" in lowered:
            dropped_selected_date_assumption = True
            continue
        cleaned.append(assumption)

    if dropped_selected_date_assumption:
        label = {
            "conversation_memory": "the prior Scenario Lab date",
            "pending_draft": "the pending scenario draft date",
            "latest_result": "the latest scenario result date",
        }.get(date_anchor_source, "the active Scenario Lab date")
        cleaned.append(f"Used {label} {target_date}.")
    return cleaned


def detect_prompt_injection(message: str) -> List[str]:
    normalized = (message or "").lower()
    flags = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            flags.append("Prompt-injection attempt detected and blocked.")
            break
    return flags


def _load_copilot_prompt() -> str:
    with open(SCENARIO_COPILOT_PROMPT_PATH, "r", encoding="utf-8") as prompt_file:
        return prompt_file.read()


def _validated_tool(tool: Any, intent: str) -> str:
    allowed = {
        "deterministic_scenario_chat",
        "deterministic_answer_only",
        "deterministic_run_confirmed",
        "none",
    }
    value = str(tool or "").strip()
    if value in allowed:
        return value
    if intent in {"data_question", "explain_result"}:
        return "deterministic_answer_only"
    if intent in {"scenario_draft", "run_simulation", "local_intel"}:
        return "deterministic_scenario_chat"
    return "none"


def _enrich_message_for_deterministic_tools(message: str, classification: Dict[str, Any]) -> str:
    enriched = message
    target_date = classification.get("target_date")
    if target_date and target_date not in enriched:
        enriched = f"{enriched} for {target_date}"
    intent = classification.get("intent")
    if intent == "run_simulation" and not re.search(r"\b(run|simulate|rerun)\b", enriched.lower()):
        enriched = f"Run scenario: {enriched}"
    elif intent in {"scenario_draft", "local_intel"} and "scenario" not in enriched.lower():
        enriched = f"Scenario: {enriched}"
    return enriched


def _apply_memory_to_followup_response(
    message: str,
    context: ScenarioChatContext,
    response: ScenarioChatResponse,
) -> ScenarioChatResponse:
    if not response.draft or not _is_followup_request(message):
        return response

    memory = context.conversation_memory
    reused = []
    if not response.draft.market_context_override and memory.last_market_context_override:
        response.draft.market_context_override = dict(memory.last_market_context_override)
        response.draft.confirmation_required = True
        response.confirmation_prompt = _confirmation_prompt_from_memory(response.confirmation_prompt, "use the previous market override")
        reused.append("previous market override")

    if not response.draft.local_intel_text and memory.last_local_intel_text and _asks_for_same_event(message):
        response.draft.local_intel_text = memory.last_local_intel_text
        scenario_state = _state_for_date(context, response.draft.target_date)
        response.draft.local_intel_estimate = estimate_local_intel_impact(
            response.draft.local_intel_text,
            current_occ=_adjusted_occupancy(scenario_state),
            forecast_occ=_forecast_for_date(context, response.draft.target_date),
            booking_velocity=_safe_float(scenario_state.get("booking_velocity"), 1.0),
            retained_pace_index=_safe_float(
                scenario_state.get("retained_pace_index"),
                _safe_float(scenario_state.get("booking_velocity"), 1.0),
            ),
            pickup_trend_index=_safe_float(
                scenario_state.get("pickup_trend_index"),
                _safe_float(scenario_state.get("booking_velocity"), 1.0),
            ),
            target_date=response.draft.target_date,
            market_context=_market_context_from_state(scenario_state),
        )
        response.draft.confirmation_required = True
        response.confirmation_prompt = _confirmation_prompt_from_memory(response.confirmation_prompt, "reuse the previous local intel")
        reused.append("previous local intel")

    if response.draft.manual_demand_shock == 0 and memory.last_manual_demand_shock:
        response.draft.manual_demand_shock = memory.last_manual_demand_shock
        reused.append("previous manual demand adjustment")

    if reused:
        response.assumptions.extend([f"Reused {item} from conversation memory." for item in reused])
        response.grounding_sources.append("Conversation memory")
        response.answer = f"{response.answer} I also reused {', '.join(reused)} from the previous scenario."
    return response


def _is_horizon_question(message: str, classification: Dict[str, Any]) -> bool:
    normalized = (message or "").lower()
    return (
        classification.get("intent") == "data_question"
        and any(term in normalized for term in ["which date", "what date", "most concerning", "highest risk", "riskiest", "need review", "needs review", "all dates", "next 30"])
        and any(term in normalized for term in ["concern", "risk", "review", "watch", "worried", "problem"])
    )


def _is_followup_request(message: str) -> bool:
    normalized = (message or "").lower()
    return any(
        term in normalized
        for term in [
            "same",
            "again",
            "previous",
            "last one",
            "that scenario",
            "that event",
            "next day",
            "following day",
            "compare",
        ]
    )


def _asks_for_same_event(message: str) -> bool:
    normalized = (message or "").lower()
    return any(term in normalized for term in ["same event", "same local", "that event", "same thing"])


def _confirmation_prompt_from_memory(existing: Optional[str], addition: str) -> str:
    if existing:
        return existing
    return f"Confirm to {addition} before I run this scenario."


def _answer_needs_fallback(
    answer: str,
    tool_response: ScenarioChatResponse,
    grounded_context: Optional[Dict[str, Any]] = None,
) -> bool:
    lowered = answer.lower()
    if any(term in lowered for term in TECHNICAL_LEAK_TERMS):
        return True
    if any(term in lowered for term in PRICE_ACTION_LEAK_TERMS):
        return True
    grounding_text = _grounding_text(tool_response, grounded_context)
    if _introduces_unsupported_business_terms(lowered, grounding_text):
        return True
    if _has_unsupported_dates(answer, grounding_text):
        return True
    if _has_signed_metric_polarity_conflict(lowered, tool_response, grounded_context):
        return True
    money_values = _money_values(answer)
    if _has_unsupported_numeric_values(money_values, _allowed_money_values(tool_response, grounded_context), tolerance=0.02):
        return True
    percent_values = _percent_values(answer)
    if _has_unsupported_numeric_values(
        percent_values,
        _allowed_percent_values(tool_response, grounded_context),
        tolerance=0.15,
    ):
        return True
    kpi_values = _kpi_point_values(answer)
    if _has_unsupported_numeric_values(
        kpi_values,
        _allowed_kpi_values(tool_response, grounded_context),
        tolerance=0.02,
    ):
        return True
    ratio_values = _ratio_values(answer)
    return _has_unsupported_numeric_values(
        ratio_values,
        _allowed_ratio_values(tool_response, grounded_context),
        tolerance=0.02,
    )


def _introduces_unsupported_business_terms(answer: str, grounding_text: str) -> bool:
    for term, grounding_aliases in PROTECTED_BUSINESS_TERMS.items():
        if term in answer and not any(alias in grounding_text for alias in grounding_aliases):
            return True
    return False


def _has_unsupported_dates(answer: str, grounding_text: str) -> bool:
    for date_text in re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", answer):
        if date_text not in grounding_text:
            return True
    return False


def _has_signed_metric_polarity_conflict(
    answer: str,
    tool_response: ScenarioChatResponse,
    grounded_context: Optional[Dict[str, Any]] = None,
) -> bool:
    market_gap = _grounded_signed_value(
        tool_response,
        grounded_context,
        ["market_gap_pct", "competitor_gap_pct"],
    )
    if market_gap is not None:
        if market_gap > 0.05 and _mentions_direction(answer, "below", ["comp", "competitor", "market"]):
            return True
        if market_gap < -0.05 and _mentions_direction(answer, "above", ["comp", "competitor", "market"]):
            return True

    reference_gap = _grounded_signed_value(
        tool_response,
        grounded_context,
        ["adr_vs_reference_pct", "pct_delta_from_reference"],
    )
    if reference_gap is not None:
        if reference_gap > 0.05 and _mentions_direction(answer, "below", ["reference"]):
            return True
        if reference_gap < -0.05 and _mentions_direction(answer, "above", ["reference"]):
            return True
    return False


def _mentions_direction(answer: str, direction: str, anchors: List[str]) -> bool:
    anchor_pattern = "|".join(re.escape(anchor) for anchor in anchors)
    return bool(
        re.search(rf"\b{direction}\b[^.?!;]{{0,90}}\b({anchor_pattern})\b", answer)
        or re.search(rf"\b({anchor_pattern})\b[^.?!;]{{0,90}}\b{direction}\b", answer)
    )


def _grounded_signed_value(
    tool_response: ScenarioChatResponse,
    grounded_context: Optional[Dict[str, Any]],
    keys: List[str],
) -> Optional[float]:
    payloads = [
        tool_response.scenario_result or {},
        _response_to_grounded_payload(tool_response),
        grounded_context or {},
    ]
    for payload in payloads:
        value = _first_numeric_value_for_keys(payload, keys)
        if value is not None:
            return value
    return None


def _first_numeric_value_for_keys(payload: Any, keys: List[str]) -> Optional[float]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys:
                numeric = _optional_float(value)
                if numeric is not None:
                    return numeric
            nested = _first_numeric_value_for_keys(value, keys)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _first_numeric_value_for_keys(item, keys)
            if nested is not None:
                return nested
    return None


def _has_unsupported_numeric_values(values: List[float], allowed_values: List[float], tolerance: float) -> bool:
    if not values:
        return False
    if not allowed_values:
        return True
    return any(all(abs(value - allowed) > tolerance for allowed in allowed_values) for value in values)


def _allowed_money_values(
    tool_response: ScenarioChatResponse,
    grounded_context: Optional[Dict[str, Any]] = None,
) -> List[float]:
    result = tool_response.scenario_result or {}
    draft = tool_response.draft
    values = _money_values(tool_response.answer)
    values.extend(
        [
            _optional_float(result.get("final_adr")),
            _optional_float(result.get("absolute_delta")),
        ]
    )
    if draft and draft.market_context_override:
        values.extend(
            [
                _optional_float(draft.market_context_override.get("comp_low")),
                _optional_float(draft.market_context_override.get("comp_median")),
                _optional_float(draft.market_context_override.get("comp_high")),
            ]
        )
    values.extend(_numeric_values_by_key_hint(_response_to_grounded_payload(tool_response), MONEY_FIELD_HINTS))
    values.extend(_numeric_values_by_key_hint(grounded_context or {}, MONEY_FIELD_HINTS))
    return _unique_numbers([value for value in values if value is not None])


def _allowed_percent_values(
    tool_response: ScenarioChatResponse,
    grounded_context: Optional[Dict[str, Any]] = None,
) -> List[float]:
    values = _percent_values(tool_response.answer)
    values.extend(_numeric_values_by_key_hint(_response_to_grounded_payload(tool_response), PERCENT_FIELD_HINTS))
    values.extend(_numeric_values_by_key_hint(grounded_context or {}, PERCENT_FIELD_HINTS))
    return _unique_numbers(values)


def _allowed_ratio_values(
    tool_response: ScenarioChatResponse,
    grounded_context: Optional[Dict[str, Any]] = None,
) -> List[float]:
    values = _ratio_values(tool_response.answer)
    values.extend(_numeric_values_by_key_hint(_response_to_grounded_payload(tool_response), RATIO_FIELD_HINTS))
    values.extend(_numeric_values_by_key_hint(grounded_context or {}, RATIO_FIELD_HINTS))
    return _unique_numbers(values)


def _allowed_kpi_values(
    tool_response: ScenarioChatResponse,
    grounded_context: Optional[Dict[str, Any]] = None,
) -> List[float]:
    values = _kpi_point_values(tool_response.answer)
    values.extend(_numeric_values_by_key_hint(_response_to_grounded_payload(tool_response), KPI_FIELD_HINTS))
    values.extend(_numeric_values_by_key_hint(grounded_context or {}, KPI_FIELD_HINTS))
    return _unique_numbers(values)


def _money_values(text: str) -> List[float]:
    values = []
    for match in re.finditer(r"\$\s*([0-9]{1,4}(?:,[0-9]{3})*(?:\.\d+)?)", text):
        try:
            values.append(float(match.group(1).replace(",", "")))
        except ValueError:
            pass
    return values


def _percent_values(text: str) -> List[float]:
    values = []
    for match in re.finditer(r"(-?\d{1,4}(?:\.\d+)?)\s*%", text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            pass
    return values


def _ratio_values(text: str) -> List[float]:
    values = []
    for match in re.finditer(r"\b(-?\d{1,3}(?:\.\d+)?)\s*x\b", text.lower()):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            pass
    return values


def _kpi_point_values(text: str) -> List[float]:
    values = []
    normalized = text.lower()
    for match in re.finditer(
        r"\b(mae|rmse|bias|avg occupancy miss|average occupancy miss|large-miss guardrail|wape|mape|accuracy|stability|volatility)"
        r"\b[^.?!;]{0,80}?(-?\d{1,4}(?:\.\d+)?)\s*(?:pp|percentage points|%)?",
        normalized,
    ):
        try:
            values.append(float(match.group(2)))
        except ValueError:
            pass
    return values


def _grounding_text(
    tool_response: ScenarioChatResponse,
    grounded_context: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "tool_result": _response_to_grounded_payload(tool_response),
        "grounded_context": grounded_context or {},
        "grounding_sources": tool_response.grounding_sources,
        "source_labels": tool_response.source_labels,
    }
    return json.dumps(payload, sort_keys=True, default=str).lower()


def _numeric_values_by_key_hint(payload: Any, key_hints: tuple[str, ...], key_path: str = "") -> List[float]:
    values: List[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_path = f"{key_path}.{key}" if key_path else str(key)
            values.extend(_numeric_values_by_key_hint(value, key_hints, next_path))
        return values
    if isinstance(payload, list):
        for item in payload:
            values.extend(_numeric_values_by_key_hint(item, key_hints, key_path))
        return values
    if any(hint in key_path.lower() for hint in key_hints):
        value = _optional_float(payload)
        if value is not None:
            values.append(value)
    return values


def _unique_numbers(values: List[float]) -> List[float]:
    unique: List[float] = []
    for value in values:
        if not any(abs(value - existing) <= 0.001 for existing in unique):
            unique.append(value)
    return unique


def _finalize_response(response: ScenarioChatResponse, safety_flags: List[str]) -> ScenarioChatResponse:
    response.safety_flags = _merge_unique(response.safety_flags, safety_flags)
    response.grounding_sources = _merge_unique(response.grounding_sources, _grounding_sources_for_response(response))
    return response


def _grounding_sources_for_response(response: ScenarioChatResponse) -> List[str]:
    sources = list(response.source_labels or [])
    if response.scenario_result:
        sources.extend(["Scenario simulation", "Pricing guardrails"])
    if response.draft:
        sources.extend(["Scenario draft", "Local intel estimate", "Market context"])
    return _merge_unique([], sources)


def _response_to_grounded_payload(response: ScenarioChatResponse) -> Dict[str, Any]:
    return {
        "answer": response.answer,
        "confirmation_prompt": response.confirmation_prompt,
        "ran_scenario": response.ran_scenario,
        "sources": response.source_labels,
        "domain": response.domain,
        "referenced_models": response.referenced_models,
        "comparison_basis": response.comparison_basis,
        "referenced_date": response.referenced_date,
        "draft": _draft_payload(response.draft),
        "scenario_result": _latest_result_payload(response.scenario_result or {}),
        "assumptions": response.assumptions,
        "safety_flags": response.safety_flags,
    }


def _memory_payload(memory: ScenarioConversationMemory) -> Dict[str, Any]:
    return {
        "rolling_summary": memory.rolling_summary,
        "last_intent": memory.last_intent,
        "last_domain": memory.last_domain,
        "last_target_date": memory.last_target_date,
        "last_referenced_models": memory.last_referenced_models,
        "last_comparison_basis": memory.last_comparison_basis,
        "last_local_intel_text": memory.last_local_intel_text,
        "last_manual_demand_shock": memory.last_manual_demand_shock,
        "last_market_context_override": memory.last_market_context_override,
        "last_horizon_rank_request": memory.last_horizon_rank_request,
        "last_draft_pending": getattr(memory, "last_draft_pending", False),
        "last_scenario_result": memory.last_scenario_result,
        "previous_scenario_result": memory.previous_scenario_result,
        "last_sources": memory.last_sources,
    }


def _horizon_risk_payload(records: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    risks = [record for record in records if record.get("review_status") == "Review needed"]
    ranked = sorted(
        risks,
        key=lambda item: (
            bool(item.get("manual_approval_required")),
            bool(item.get("sold_out")) and bool(item.get("material_retention_gap")),
            len(item.get("review_flags") or []),
            _safe_float(item.get("revenue_upside"), 0.0),
            str(item.get("date", "")),
        ),
        reverse=True,
    )[:limit]
    return [_compact_horizon_record(record) for record in ranked]


def _horizon_opportunity_payload(records: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (
            _safe_float(item.get("revenue_upside"), 0.0),
            _safe_float(item.get("expected_revenue"), 0.0),
            str(item.get("date", "")),
        ),
        reverse=True,
    )[:limit]
    return [_compact_horizon_record(record) for record in ranked]


def _compact_horizon_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "date": record.get("date"),
        "recommended_adr": record.get("recommended_adr"),
        "booked_occupancy_pct": round(_safe_float(record.get("raw_otb_occupancy"), 0.0) * 100, 1),
        "likely_retained_occupancy_pct": round(_safe_float(record.get("adjusted_otb_occupancy"), 0.0) * 100, 1),
        "forecast_occupancy_pct": round(_safe_float(record.get("forecasted_occupancy"), 0.0) * 100, 1),
        "comp_median": record.get("competitor_median"),
        "revenue_upside": record.get("revenue_upside"),
        "review_status": record.get("review_status"),
        "manual_approval_required": record.get("manual_approval_required"),
        "sold_out": record.get("sold_out"),
        "review_flags": list(record.get("review_flags") or [])[:2],
        "top_reasons": list(record.get("top_reasons") or [])[:2],
    }


def _draft_payload(draft: Optional[ScenarioDraft]) -> Dict[str, Any]:
    if not draft:
        return {}
    return {
        "target_date": draft.target_date,
        "manual_demand_shock_pct": round(draft.manual_demand_shock * 100, 1),
        "local_intel_text": draft.local_intel_text,
        "local_intel_estimate": draft.local_intel_estimate,
        "market_context_override": draft.market_context_override,
        "apply_local_intel": draft.apply_local_intel,
        "confirmation_required": draft.confirmation_required,
        "confirmed": draft.confirmed,
    }


def _latest_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    if not result:
        return {}
    return {
        "target_date": result.get("target_date"),
        "final_adr": result.get("final_adr"),
        "adr_vs_reference_pct": result.get("pct_delta_from_reference"),
        "adr_vs_reference_amount": result.get("absolute_delta"),
        "market_gap_pct": result.get("competitor_gap_pct"),
        "pricing_pace_index": result.get("pricing_pace_index"),
        "local_intel_applied_shock_pct": round(_safe_float(result.get("local_intel_applied_shock"), 0.0) * 100, 1),
        "recommended_action": result.get("ai_recommended_action") or result.get("strategy_applied"),
        "risk_level": result.get("ai_risk_level"),
    }


def _merge_unique(left: List[str], right: List[str]) -> List[str]:
    merged = []
    for item in list(left or []) + list(right or []):
        text = str(item).strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
