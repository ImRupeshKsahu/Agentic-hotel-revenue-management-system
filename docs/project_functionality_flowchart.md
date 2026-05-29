# Hotel RMS Project Functionality Flowchart

This Mermaid flowchart maps the current Hotel RMS PoC end to end: source data, PMS simulation, demand forecasting, backtesting, market feed, pricing optimization, Streamlit surfaces, Scenario Lab, copilot layers, artifacts, and tests.

```mermaid
flowchart TD
    User["Hotel manager / analyst"] --> UI["Streamlit app<br/>src/app.py"]

    subgraph DataLayer["Data Inputs And Configuration"]
        RawBookings["Raw booking data<br/>data/hotel_bookings.csv"]
        DailyData["Daily hotel data<br/>data/daily_hotel_data.csv"]
        LocalCalendar["Local intel calendar<br/>data/local_intel_calendar.csv"]
        Config["Project config<br/>src/config.py"]
        Prompts["Prompt contracts<br/>src/prompts/strategist.txt<br/>src/prompts/scenario_copilot.txt"]
        Env["DeepSeek env settings<br/>.env / .env.example"]
    end

    subgraph PMSLayer["PMS And On-The-Books Layer"]
        NormalizeBookings["Normalize bookings<br/>pms_core.data_pipeline.normalize_bookings"]
        RoomNights["Unroll room nights<br/>pms_core.data_pipeline.unroll_room_nights"]
        DailyBuilder["Build PMS-derived daily table<br/>refresh_daily_hotel_data"]
        LiveLedger["Seed / simulate live PMS ledger<br/>pms_core.live_ledger"]
        CancellationRisk["Estimate cancellation risk<br/>pricing_core.cancellation"]
        OTBSnapshot["Calculate OTB snapshot<br/>gross OTB, adjusted OTB, OTB ADR"]
        PaceSignals["Pace signals<br/>gross pace, retained pace, pickup trend, pricing pace"]
        LiveState["Export live market state<br/>data/live_market_state.json"]
    end

    subgraph MarketLayer["Competitor Market Layer"]
        MarketBaseline["Build competitor baseline<br/>market_core.feed"]
        MarketInit["Initialize comp-set market<br/>data/live_competitor_market.csv"]
        MarketEvent["Simulate competitor market event<br/>simulate_competitor_market.py"]
        MarketContext["Market context<br/>comp low, median, high, source quality, regime"]
    end

    subgraph ForecastLayer["Demand Forecasting And Model Governance"]
        ForecastWorkflow["Forecast workflow CLI<br/>workflows/demand_forecast.py<br/>modes: auto, backtest, forecast"]
        FeatureEng["Feature engineering<br/>lags, rolling stats, trend, calendar, operational signals"]
        Boruta["Boruta feature selection<br/>stable force-kept feature spine"]
        Optuna["One-time hyperparameter tuning<br/>model_hyperparameters.json<br/>hyperparameter_tuning_report.csv"]
        ModelRegistry["Model registry / algorithms<br/>statistical, recursive ML, chain ML"]
        Backtest["Weekly model competition backtest<br/>scenario lags and audit folds"]
        Champion["Champion selection<br/>forecast_champion.json"]
        ForecastRun["Daily champion forecast<br/>data/demand_forecast_output.csv"]
        ForecastArtifacts["Forecast artifacts<br/>metrics, comparison, lag metrics, predictions, plots"]
    end

    subgraph PricingLayer["Explainable Pricing Engine"]
        ScenarioInputs["Scenario inputs<br/>target date, forecast occupancy, live OTB, market context"]
        LocalIntel["Local intel scorer<br/>calendar event or manager event"]
        LocalApproval["Manager approval gate<br/>context-only unless approved"]
        DemandShock["Demand adjustments<br/>manual shock + approved local-intel shock"]
        Policy["Dynamic price policy<br/>recent ADR anchor, floor, ceiling, comp-set constraints"]
        CandidateOpt["Candidate ADR optimizer<br/>expected occupancy and revenue"]
        Guardrails["Guardrails and review flags<br/>sold-out protection, parity, bounds, confidence"]
        FinalADR["Final ADR recommendation<br/>price path + decision context + ADR vs reference"]
        DecisionLog["Pricing decision log<br/>data/pricing_decision_log.jsonl"]
    end

    subgraph CopilotLayer["AI And Copilot Layer"]
        PricingAgent["Agentic pricing workflow<br/>data ingestion, optimizer, pace analyst, strategist, validation"]
        LLMStrategist["DeepSeek strategy/explainer<br/>manager-facing advisory wording"]
        ManagerCopilot["Manager copilot summaries<br/>opportunities, risks, executive briefing, market outlook"]
        ScenarioCopilot["Scenario Lab copilot<br/>grounded intent routing, date parsing, ranked questions"]
        ScenarioLLM["Grounded LLM Scenario Copilot<br/>language understanding with deterministic tools"]
        Safety["Safety and grounding checks<br/>prompt injection, unsupported values, confirmation gates"]
    end

    subgraph AppLayer["Streamlit User Views"]
        Morning["Morning Briefing<br/>executive summary, KPIs, opportunities, risks, 30-day snapshot, date detail"]
        Outlook["Market Outlook<br/>market KPIs, pricing-vs-market chart, strategy feed, champion audit"]
        ScenarioLab["Scenario Lab<br/>manual scenarios, market override, local intel, run scenario, technical trace"]
        Chat["Scenario Copilot Chat<br/>Q&A, draft scenarios, confirmation, latest simulation result"]
        Visuals["Charts and tables<br/>booking quality, pricing vs market, audit tables, traces"]
    end

    subgraph Outputs["Outputs And Reusable Artifacts"]
        Docs["Project docs<br/>pricing, cancellation, local intel, scenario copilot"]
        Tests["Unit tests<br/>pricing, cancellation, market feed, backtest, copilot"]
        GeneratedData["Generated data artifacts<br/>forecast, OTB, market state, model comparison, audit outputs"]
        Plots["Plots<br/>forecast plots and backtest timeline"]
    end

    RawBookings --> NormalizeBookings
    NormalizeBookings --> RoomNights
    RoomNights --> DailyBuilder
    DailyBuilder --> DailyData
    RawBookings --> LiveLedger
    LiveLedger --> CancellationRisk
    CancellationRisk --> OTBSnapshot
    OTBSnapshot --> PaceSignals
    PaceSignals --> LiveState

    DailyData --> ForecastWorkflow
    Config --> ForecastWorkflow
    ForecastWorkflow --> FeatureEng
    FeatureEng --> Boruta
    Boruta --> ModelRegistry
    ModelRegistry --> Optuna
    Optuna --> Backtest
    Backtest --> Champion
    Champion --> ForecastRun
    ForecastRun --> ForecastArtifacts
    Backtest --> ForecastArtifacts
    ForecastArtifacts --> GeneratedData
    ForecastArtifacts --> Plots

    ForecastRun --> MarketBaseline
    OTBSnapshot --> MarketInit
    MarketBaseline --> MarketInit
    MarketInit --> MarketEvent
    MarketEvent --> MarketContext
    MarketContext --> LiveState

    ForecastRun --> ScenarioInputs
    LiveState --> ScenarioInputs
    MarketContext --> ScenarioInputs
    LocalCalendar --> LocalIntel
    User --> LocalIntel
    LocalIntel --> LocalApproval
    LocalApproval --> DemandShock
    ScenarioInputs --> DemandShock
    DemandShock --> Policy
    MarketContext --> Policy
    DailyData --> Policy
    Policy --> CandidateOpt
    CandidateOpt --> Guardrails
    Guardrails --> FinalADR
    FinalADR --> DecisionLog

    ScenarioInputs --> PricingAgent
    FinalADR --> PricingAgent
    Prompts --> LLMStrategist
    Env --> LLMStrategist
    PricingAgent --> LLMStrategist
    LLMStrategist --> Safety
    Safety --> FinalADR

    ForecastArtifacts --> ManagerCopilot
    FinalADR --> ManagerCopilot
    LiveState --> ManagerCopilot
    ManagerCopilot --> Morning
    ManagerCopilot --> Outlook

    UI --> Morning
    UI --> Outlook
    UI --> ScenarioLab
    UI --> Chat
    Morning --> Visuals
    Outlook --> Visuals
    ScenarioLab --> Visuals

    ScenarioLab --> ScenarioInputs
    ScenarioLab --> LocalIntel
    ScenarioLab --> PricingAgent
    Chat --> ScenarioLLM
    ScenarioLLM --> ScenarioCopilot
    ScenarioCopilot --> Safety
    ScenarioCopilot --> ScenarioInputs
    ScenarioCopilot --> ForecastArtifacts
    ScenarioCopilot --> MarketContext
    ScenarioCopilot --> LocalIntel
    ScenarioCopilot --> PricingAgent

    FinalADR --> Morning
    FinalADR --> Outlook
    FinalADR --> ScenarioLab
    ForecastRun --> Morning
    ForecastRun --> Outlook
    ForecastRun --> ScenarioLab
    LiveState --> Morning
    LiveState --> Outlook
    LiveState --> ScenarioLab

    Docs --> User
    Tests --> User
    GeneratedData --> UI
    Plots --> UI
    ForecastArtifacts --> Tests
    FinalADR --> Tests
    ScenarioCopilot --> Tests
```

## Functional Coverage

- **Forecast pipeline:** raw bookings become daily demand history, engineered forecast features, Boruta-selected ML features, tuned model parameters, backtest artifacts, champion metadata, and daily demand forecasts.
- **Live hotel state:** live PMS ledger and OTB snapshot produce booked rooms, cancellation-adjusted retained rooms, booked ADR, pace signals, and a JSON market state consumed by the UI and pricing engine.
- **Competitor market:** simulated comp-set feed provides low / median / high rates, market regime, source quality, and override-able Scenario Lab market context.
- **Pricing engine:** forecast occupancy, OTB, pace, market context, local-intel overlays, dynamic price policy, candidate revenue optimization, and guardrails produce an explainable final ADR.
- **Local intelligence:** seeded calendar events and manager-entered events are scored into suggested demand pressure and ADR headroom, but affect pricing only after explicit approval.
- **AI layer:** DeepSeek/LangGraph components explain and route, while deterministic code owns pricing, scenario execution, data lookups, and confirmation gates.
- **Streamlit app:** Morning Briefing, Market Outlook, Scenario Lab, charts, traces, audit tables, and Scenario Copilot chat expose the system to a manager.
- **Governance:** tests, docs, audit metrics, model comparison artifacts, price-path traces, guardrail rows, and decision logs keep the PoC inspectable.
