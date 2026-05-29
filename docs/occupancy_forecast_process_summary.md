# Occupancy Forecast Process Summary

## 1. Executive Summary

The Hotel RMS PoC uses a dedicated occupancy forecasting layer to estimate future stay-date demand before pricing decisions are made.

The current forecasting layer is best described as:

> rolling-origin model competition for a 30-day hotel occupancy forecast

or, more plainly:

> a weekly backtested forecasting pipeline that selects one champion model, saves its feature schema and tuning metadata, and then reuses that champion for daily forecast generation.

The forecast process is intentionally separate from pricing:

- forecasting predicts expected occupancy for each future stay date,
- backtesting decides which model should be trusted for the current data state,
- the saved champion forecast becomes an input to the deterministic pricing optimizer,
- pricing remains rule-based and explainable rather than controlled by the forecast model.

The current saved champion is:

```text
Model: extra_trees_recursive
Strategy: recursive_ml
Feature profile: boruta_selected
Forecast horizon: 30 days
Selection objective: mae_pp_with_rmse_guardrail
Backtest cadence: 7 days
```

Current artifact snapshot:

```text
Selected at: 2026-05-20T16:40:47.292646
Training data end date: 2017-08-31
Selection folds: 49
Audit folds: 8
Selection MAE_pp: 8.7854
Selection RMSE_pp: 10.5446
Selection WAPE: 12.3077
Recent audit MAE_pp: 2.3207
Audit status: ok
```

---

## 2. Where the Forecasting Pipeline Lives

The top-level compatibility command is:

```powershell
python src\demand_forecast.py backtest
python src\demand_forecast.py forecast
python src\demand_forecast.py auto
```

The real CLI implementation is in:

```text
src/workflows/demand_forecast.py
```

The forecasting internals are organized under:

```text
src/forecasting_core/
```

Important files:

| File | Role |
| --- | --- |
| `src/workflows/demand_forecast.py` | CLI workflow for backtest, forecast, and auto modes |
| `src/forecasting_core/engine.py` | High-level orchestration facade |
| `src/forecasting_core/legacy.py` | Core forecast, backtest, model-selection, metrics, artifact logic |
| `src/forecasting_core/feature_engineering.py` | Feature generation contract |
| `src/forecasting_core/hyperparameter_tuning.py` | Optuna tuning for ML models |
| `src/forecasting_core/model_registry.py` | Model lists, strategy lookup, availability checks |
| `src/forecasting_core/backtesting.py` | Backtest facade over rolling-origin helpers |
| `src/forecasting_core/artifacts.py` | Saved champion and artifact helpers |
| `src/project_core/config.py` | Forecast horizon, artifact paths, feature lists, tuning settings |

The old import path `src/forecasting.py` remains a compatibility wrapper for code and tests that still import the earlier module name.

---

## 3. Business Question Answered

For each future stay date, the forecast answers:

> Given the latest historical occupancy, booking pace, cancellation signals, calendar effects, and event flags, what occupancy rate should we expect over the next 30 days?

The output is a daily forecast table with:

```text
Date
Forecasted_Occupancy
Min_Occupancy
Max_Occupancy
Competitor_Rate
Selected_Model
Feature_Profile
```

The current forecast output starts at `2017-09-01` because the configured data end date is `2017-08-31`.

Example from `data/demand_forecast_output.csv`:

| Date | Forecasted occupancy | Min occupancy | Max occupancy | Model |
| --- | ---: | ---: | ---: | --- |
| 2017-09-01 | 92.33% | 81.82% | 100.00% | extra_trees_recursive |
| 2017-09-02 | 91.56% | 82.49% | 100.00% | extra_trees_recursive |
| 2017-09-03 | 89.43% | 79.72% | 100.00% | extra_trees_recursive |
| 2017-09-04 | 90.22% | 78.22% | 100.00% | extra_trees_recursive |
| 2017-09-05 | 91.63% | 75.93% | 100.00% | extra_trees_recursive |

---

## 4. End-to-End Workflow

```mermaid
flowchart LR
    A["Raw hotel bookings"] --> B["Refresh daily hotel data"]
    B --> C["Build historical occupancy series"]
    C --> D["Run rolling-origin backtest"]
    D --> E["Evaluate model leaderboard"]
    E --> F["Select champion model"]
    F --> G["Calibrate forecast intervals"]
    F --> H["Save champion metadata"]
    H --> I["Daily forecast mode reuses champion"]
    I --> J["30-day occupancy forecast"]
    J --> K["Seed PMS / OTB / market state"]
    K --> L["Pricing and Streamlit UI"]
```

The same workflow file supports three modes.

| Mode | What it does |
| --- | --- |
| `backtest` | Refreshes daily data, tunes ML models, runs model competition, selects and saves a champion, writes forecast and backtest artifacts |
| `forecast` | Uses the saved champion and saved tuning artifact to produce the current 30-day forecast |
| `auto` | Runs backtest when the champion is missing or older than the configured cadence; otherwise runs forecast |

The backtest cadence is currently:

```text
BACKTEST_CADENCE_DAYS = 7
```

So `auto` treats the champion as stale when it is older than 7 days.

---

## 5. Data Inputs and Historical Target

The pipeline refreshes daily data through:

```text
pms_core.data_pipeline.refresh_daily_hotel_data(...)
```

The forecast target is:

```text
Occupancy_Rate
```

The forecast history is built by `_actuals(...)` in `src/forecasting_core/legacy.py`.

That function:

- parses `Date`,
- keeps rows where `Occupancy_Rate` is present,
- sorts by date,
- ensures expected columns exist,
- forward-fills competitor rate if needed,
- fills missing booking pace with zero.

The key historical covariates expected by the forecast layer are:

```text
Date
Occupancy_Rate
Is_Weekend
Local_Event
Competitor_Rate
Booking_Pace
Cancellations
Bookings_Created
```

Important note:

> The current implementation no longer treats competitor rate as a strong forecasting feature for ML feature selection. The forecast layer mainly depends on occupancy history, calendar, local-event flags, booking pace, booking creation, and cancellation signals.

---

## 6. Feature Engineering

The feature contract is defined in `src/project_core/config.py` and implemented in `src/forecasting_core/feature_engineering.py`.

There are two practical feature profiles:

| Profile | Used by | Meaning |
| --- | --- | --- |
| `statistical` | statistical models | No Boruta-selected ML feature schema |
| `boruta_selected` | ML models | Baseline forecasting spine plus Boruta-selected enhanced features |

Older names like `baseline` and `enhanced_v1` are normalized for compatibility, but the current ML path uses one profile:

```text
boruta_selected
```

### 6.1 Baseline features

The stable baseline spine includes:

- recent occupancy lags: `1, 2, 3, 7, 14, 21, 28, 56`,
- rolling means and standard deviations over `7, 14, 28, 56` days,
- short trend features such as `trend_7` and `trend_14`,
- recent min and max occupancy over 28 days,
- recent booking pace,
- recent cancellations,
- origin-date calendar features.

These baseline features are force-kept for ML models so Boruta cannot remove the proven core signal.

### 6.2 Enhanced candidate features

Boruta can choose from additional engineered features, including:

- extra occupancy lags such as `4, 5, 6, 35, 42, 84, 112, 364`,
- rolling min, max, and slope features,
- trend projection features,
- week-over-week and year-over-year level differences,
- day-of-week and month seasonal indexes,
- booking pace, bookings-created, and cancellation rolling stats.

### 6.3 Future-known features

For each forecast horizon row, the model can use date features that are known in advance:

```text
hN_dow_sin
hN_dow_cos
hN_doy_sin
hN_doy_cos
hN_month_sin
hN_month_cos
hN_is_weekend
hN_local_event
```

For the recursive champion, the saved schema currently includes `h1_*` features because the recursive model predicts one day ahead repeatedly.

---

## 7. Boruta Feature Selection

For ML models, feature selection is performed once from all available history before model competition.

The current settings are:

```text
BORUTA_MAX_ITER = 10
BORUTA_TREE_COUNT = "auto"
BORUTA_PERC = 80
```

The selection behavior differs by strategy:

| Strategy | Boruta target |
| --- | --- |
| `recursive_ml` | one-day-ahead recursive training target |
| `regressor_chain` | selected horizon anchors, currently days 1, 14, and 30 |

The chain settings are:

```text
CHAIN_BORUTA_ANCHORS = [1, 14, 30]
CHAIN_BORUTA_MIN_ANCHORS = 2
```

This means a chain feature is considered stable when it is supported across enough horizon anchors.

The saved artifacts show:

| Artifact | What it explains |
| --- | --- |
| `data/feature_manifest.csv` | candidate features, roles, and whether each feature was selected in the champion |
| `data/boruta_selection_report.csv` | Boruta support/rank outcomes by strategy and anchor |
| `data/forecast_champion.json` | the final champion schema and selected historical features |

Current champion feature summary:

```text
Champion selected historical features: 47
Champion mandatory future-known features: 8
Champion total schema features: 55
```

Current `data/feature_manifest.csv` summary:

| Feature role | Selected in champion | Count |
| --- | ---: | ---: |
| force_kept_historical | true | 29 |
| historical_candidate | true | 18 |
| historical_candidate | false | 38 |
| mandatory_future_known | true | 8 |
| mandatory_future_known | false | 232 |

---

## 8. Models Included in Backtesting

The current default backtest slate is:

| Model | Strategy | Feature profile | Notes |
| --- | --- | --- | --- |
| `seasonal_naive_7` | statistical | statistical | repeats the latest weekly seasonal pattern |
| `ewma_14` | statistical | statistical | exponentially weighted moving average baseline |
| `sarimax` | statistical | statistical | statsmodels SARIMAX with small AIC candidate search |
| `random_forest_recursive` | recursive_ml | boruta_selected | one-step model rolled forward recursively |
| `extra_trees_recursive` | recursive_ml | boruta_selected | one-step Extra Trees model rolled forward recursively |
| `xgboost_recursive` | recursive_ml | boruta_selected | one-step XGBoost model rolled forward recursively |
| `random_forest_chain` | regressor_chain | boruta_selected | direct multi-output chain for the full horizon |
| `extra_trees_chain` | regressor_chain | boruta_selected | Extra Trees regressor chain |
| `xgboost_chain` | regressor_chain | boruta_selected | XGBoost regressor chain |

The broader supported model list also contains experimental or non-default models:

```text
naive
rolling_mean_7
rolling_mean_14
rolling_mean_28
ets
ridge_chain
elasticnet_chain
```

These are supported by code paths but are not in the current default weekly model competition slate.

### 8.1 Statistical model behavior

Statistical models do not use Boruta.

Examples:

- `seasonal_naive_7` repeats the last 7 observed occupancy values over the forecast horizon.
- `ewma_14` repeats the latest 14-span exponentially weighted mean.
- `sarimax` uses statsmodels SARIMAX when available and enough history exists.

SARIMAX candidates currently include three small seasonal specifications:

```text
(1,0,1) x (1,0,1,7), trend c
(1,1,1) x (0,1,1,7), trend n
(2,0,1) x (1,0,1,7), trend c
```

If SARIMAX is unavailable at import time, model competition skips it rather than scoring another model under the SARIMAX name.

### 8.2 Recursive ML behavior

Recursive ML models train a one-step-ahead predictor:

```text
history through cutoff -> predict next day occupancy
```

To forecast 30 days, the model:

1. predicts day 1,
2. appends that prediction as if it were the next known occupancy,
3. recomputes the one-step feature vector,
4. predicts day 2,
5. repeats until day 30.

This strategy is compact and realistic for sequential operations, but errors can compound across the horizon.

### 8.3 Regressor-chain behavior

Regressor-chain models train a full 30-day target vector:

```text
history through cutoff -> predict occupancy days 1..30
```

The chain can learn horizon-specific patterns directly, but it is more complex and depends on enough historical examples for the full 30-day target.

---

## 9. Hyperparameter Tuning

Before the weekly backtest, the pipeline tunes tunable ML models once and saves the result.

The tuning code lives in:

```text
src/forecasting_core/hyperparameter_tuning.py
```

The current tuning contract is:

```text
Optuna trials per ML model: 5
Recent folds used for tuning: 5
Objective: recent_cv_mae_pp_rmse_guardrail
Feature profile: boruta_selected
```

Tunable model families:

```text
random_forest_*
extra_trees_*
xgboost_*
```

Saved tuning artifacts:

```text
data/model_hyperparameters.json
data/hyperparameter_tuning_report.csv
```

The saved payload is considered current only when its metadata matches:

- objective,
- number of trials,
- recent-fold count,
- horizon,
- model list,
- data end date,
- feature profile,
- status.

If Optuna is unavailable, the tuning payload records a skipped status and the ML models use default parameters.

Current tuning artifact status:

```text
Status: ok
Tuned at: 2026-05-20T15:05:13.627646
Data end date: 2017-08-31
Models tuned: 6
Recent folds used: 5
```

Current champion tuning result:

```text
Champion model: extra_trees_recursive
Best params: max_features=sqrt, min_samples_leaf=8, n_estimators=17
Best tuning MAE_pp: 2.1102
Best tuning RMSE_pp: 2.7578
Best tuning WAPE: 2.2880
```

---

## 10. Backtesting Design

Backtesting uses rolling-origin evaluation.

For each fold:

1. choose a cutoff date,
2. train only on dates up to the cutoff,
3. predict the next 30 days,
4. compare predicted occupancy to actual occupancy for those 30 days,
5. repeat across weekly cutoffs.

The fold generator is `_generate_weekly_folds(...)` in `src/forecasting_core/legacy.py`.

Current defaults:

```text
Forecast horizon: 30 days
Minimum training history: 365 days
Backtest step: 7 days
Audit folds: 8
```

With the current data, the saved run has:

```text
Total folds: 57
Selection folds: 49
Audit folds: 8
Models per fold: 9
Prediction rows: 15,390
```

Why 15,390 prediction rows?

```text
57 folds * 9 models * 30 forecast days = 15,390 predictions
```

### 10.1 Selection vs audit split

The backtest deliberately separates folds into:

| Split | Purpose |
| --- | --- |
| `selection` | used to rank and select the champion |
| `audit` | recent holdout-style folds used to check whether the selected champion still performs well |

The current saved run uses:

```text
49 selection folds * 9 models = 441 fold metric rows
8 audit folds * 9 models = 72 fold metric rows
```

### 10.2 Scenario folds

The code also supports scenario-lag folds through `_scenario_folds(...)`.

Those folds are built from explicit lag values such as:

```text
10, 14, 21, 30, 45, 60
```

The default weekly model competition path uses rolling weekly folds, not scenario folds, unless `scenario_lags` is passed explicitly.

---

## 11. Metrics Used in Backtesting

Metrics are calculated by `calculate_forecast_metrics(...)`.

Let:

```text
error = predicted - actual
```

The main metrics are:

| Metric | Formula / meaning |
| --- | --- |
| `MAE` | mean(abs(error)) on 0-1 occupancy scale |
| `MAE_pp` | `MAE * 100`, average miss in occupancy percentage points |
| `RMSE` | sqrt(mean(error^2)) on 0-1 occupancy scale |
| `RMSE_pp` | `RMSE * 100`, penalizes larger misses |
| `MAPE` | mean(abs(error / actual)) * 100 |
| `sMAPE` | symmetric percentage error |
| `WAPE` | sum(abs(error)) / sum(abs(actual)) * 100 |
| `Bias` | mean(error) on 0-1 occupancy scale |
| `Bias_pp` | `Bias * 100`; positive means over-forecasting, negative means under-forecasting |
| `Abs_Bias_pp` | abs(`Bias_pp`) |
| `Accuracy` | max(0, 100 - WAPE) |
| `Volatility` | standard deviation of predictions |
| `Stability` | `1 / (1 + std(error))` |

For manager and model-selection readability, exported summaries focus on percent-point metrics:

```text
MAE_pp
RMSE_pp
Bias_pp
Abs_Bias_pp
MAPE
WAPE
Volatility
Stability
Complexity
```

---

## 12. Champion Selection

Champion selection is controlled by:

```text
MODEL_SELECTION_OBJECTIVE = "mae_pp_with_rmse_guardrail"
MODEL_SELECTION_MAE_TIE_THRESHOLD_PP = 0.50
```

The ranking rule is:

1. find the best `MAE_pp`,
2. treat models within `0.50` percentage points of that MAE as eligible,
3. within the eligible set, sort by:
   - lowest `RMSE_pp`,
   - lowest `MAE_pp`,
   - lowest `Abs_Bias_pp`,
   - lowest `Complexity`,
   - model name.

This is why the selection story should be described as:

> primary MAE_pp selection with an RMSE_pp guardrail for large-miss risk.

The saved champion metadata records:

```text
selection_objective
mae_tie_threshold_pp
metrics
feature_schema
feature_selection_metadata
interval_quantiles
backtest_metadata
hyperparameter_tuning_metadata
```

---

## 13. Current Model Leaderboard

Current `data/model_comparison_metrics.csv` selection leaderboard:

| Rank | Feature profile | Model | Strategy | MAE_pp | RMSE_pp | Bias_pp | WAPE |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | boruta_selected | extra_trees_recursive | recursive_ml | 8.7854 | 10.5446 | -3.1363 | 12.3077 |
| 2 | boruta_selected | random_forest_recursive | recursive_ml | 9.1817 | 10.9917 | -1.4247 | 13.2784 |
| 3 | statistical | ewma_14 | statistical | 9.3120 | 11.2431 | -0.3022 | 13.4441 |
| 4 | boruta_selected | xgboost_recursive | recursive_ml | 9.5564 | 11.3295 | -1.3981 | 13.4540 |
| 5 | statistical | sarimax | statistical | 10.3891 | 12.4133 | 4.6370 | 14.6515 |
| 6 | statistical | seasonal_naive_7 | statistical | 10.5457 | 12.8537 | 0.0247 | 15.3045 |
| 7 | boruta_selected | xgboost_chain | regressor_chain | 10.8657 | 12.6559 | -5.5399 | 14.9818 |
| 8 | boruta_selected | random_forest_chain | regressor_chain | 12.3602 | 14.2824 | -10.9466 | 16.8438 |
| 9 | boruta_selected | extra_trees_chain | regressor_chain | 12.9602 | 14.9152 | -12.1453 | 17.4564 |

Interpretation:

- `extra_trees_recursive` is the current champion.
- Recursive ML models currently outperform chain ML models in the selection split.
- `ewma_14` is a strong simple baseline and remains useful as a sanity check.
- SARIMAX participates as a statistical benchmark, not as the champion.

---

## 14. Audit and Drift Check

After champion selection, the last 8 folds are evaluated as a recent audit split.

The audit compares recent performance to selection performance.

Current drift rule:

```text
Audit_Drift_Ratio = audit_mean_mae_pp / selection_mean_mae_pp
Status = recent_degradation_flagged if Audit_Drift_Ratio > 1.25
```

Current champion audit:

```text
Champion: extra_trees_recursive
Selection MAE_pp: 8.7854
Audit MAE_pp: 2.3207
Audit_Drift_Ratio: 0.2642
Audit_Status: ok
Interval_Coverage: 1.0
```

Because recent audit MAE is lower than selection MAE, the current champion is not flagged for degradation.

---

## 15. Forecast Intervals

The forecast output includes:

```text
Min_Occupancy
Max_Occupancy
```

When backtest interval quantiles exist, intervals are calibrated from selection residuals by lag:

```text
residual = actual - predicted
lower residual quantile = alpha / 2
upper residual quantile = 1 - alpha / 2
```

With the current interval level:

```text
DEFAULT_INTERVAL_LEVEL = 0.90
```

The final bounds are:

```text
Min_Occupancy = clip(prediction + lower_residual_quantile_for_lag, 0, 1)
Max_Occupancy = clip(prediction + upper_residual_quantile_for_lag, 0, 1)
```

If interval quantiles are unavailable, forecast mode falls back to a simple recent residual standard-deviation band.

---

## 16. Saved Artifacts

Backtest mode writes the main artifacts used by the app, audits, and stakeholder review.

| Artifact | Purpose |
| --- | --- |
| `data/daily_hotel_data.csv` | refreshed daily historical occupancy and operational data |
| `data/demand_forecast_output.csv` | current 30-day occupancy forecast |
| `data/model_comparison_metrics.csv` | selection leaderboard |
| `data/model_validation_metrics.csv` | top model row for compact validation display |
| `data/forecast_champion.json` | champion model, metrics, schema, interval, tuning, and audit metadata |
| `data/backtest_predictions.csv` | row-level predictions for every model/fold/day |
| `data/backtest_fold_metrics.csv` | per-fold metrics by model |
| `data/backtest_lag_metrics.csv` | metrics by forecast lag |
| `data/backtest_scenario_metrics.csv` | fold-level scenario metrics |
| `data/backtest_audit_predictions.csv` | audit split predictions |
| `data/backtest_audit_fold_metrics.csv` | audit split fold metrics |
| `data/backtest_audit_summary.csv` | recent audit summary and drift fields |
| `data/backtest_audit_lag_metrics.csv` | audit metrics by lag |
| `data/backtest_audit_interval_coverage.csv` | interval coverage by lag |
| `data/feature_manifest.csv` | feature roles and champion selection flags |
| `data/boruta_selection_report.csv` | Boruta support/rank outcomes |
| `data/model_hyperparameters.json` | saved tuning payload |
| `data/hyperparameter_tuning_report.csv` | trial-level tuning report |
| `docs/backtest_timeline_explainer.png` | visual timeline of backtest folds |
| `data/plots/champion_actuals_forecast.png` | actuals plus champion forecast |
| `data/plots/backtest_models_latest_scenario.png` | latest backtest scenario plot |

After forecast generation, the workflow also seeds:

```text
data/live_hotel_bookings.csv
data/otb_snapshot.csv
data/live_market_state.json
data/live_competitor_market.csv
```

That makes the forecast available to the Streamlit app and the pricing layer.

---

## 17. Forecast Mode vs Backtest Mode

### Backtest mode

Use when:

- refreshing model competition,
- selecting a new champion,
- regenerating forecast/backtest artifacts,
- tuning ML models,
- validating model drift.

Command:

```powershell
python src\demand_forecast.py backtest
```

Backtest mode:

1. refreshes daily data,
2. tunes ML hyperparameters,
3. runs rolling-origin model competition,
4. selects champion,
5. calibrates intervals,
6. saves all artifacts,
7. writes the 30-day forecast,
8. seeds live PMS, OTB, and market state.

### Forecast mode

Use when:

- a current champion already exists,
- tuning artifact is current,
- the goal is only to refresh the 30-day forecast.

Command:

```powershell
python src\demand_forecast.py forecast
```

Forecast mode:

1. refreshes daily data,
2. loads `data/forecast_champion.json`,
3. loads `data/model_hyperparameters.json`,
4. reuses the saved champion feature schema,
5. writes `data/demand_forecast_output.csv`,
6. seeds live PMS, OTB, and market state.

If the champion or tuning artifact is missing/stale, forecast mode automatically falls back to backtest mode.

---

## 18. How Forecasting Connects to Pricing

The forecast output is not the final pricing decision.

Instead:

```text
Forecasted_Occupancy -> demand context for pricing
```

The pricing engine also considers:

- current booked occupancy,
- likely retained occupancy after cancellation risk,
- booking pace,
- competitor market context,
- local-event intelligence,
- guardrails and allowed ADR range.

Important manager-facing distinction:

| Concept | Meaning |
| --- | --- |
| current booked occupancy | rooms already booked now |
| likely retained occupancy | booked rooms expected to remain after cancellation risk |
| forecast occupancy | model-predicted stay-date occupancy |

The forecast is therefore one demand signal, not an autonomous price setter.

---

## 19. Current Limitations

### 19.1 PoC data size and date range

The data covers a single historical demo property. Model rankings may change with richer production data, more properties, or a longer recent operating period.

### 19.2 Recursive model compounding

The current champion is recursive, so day-2 through day-30 forecasts depend partly on earlier predictions. This can compound errors, even though the current leaderboard favors the recursive strategy.

### 19.3 Chain models are currently weaker

Regressor-chain models are included and useful for comparison, but the latest saved leaderboard shows them underperforming recursive ML and the strongest statistical baselines.

### 19.4 SARIMAX has engineering fallback behavior

SARIMAX is skipped if statsmodels is unavailable at import time. If fitting fails inside a valid SARIMAX path, the model code can fall back to `seasonal_naive_7` for that prediction. That makes availability checks important before interpreting SARIMAX metrics.

### 19.5 Feature selection is stable but still shallow

Boruta runs with a small MVP-friendly setting:

```text
max_iter = 10
perc = 80
```

This is practical for the PoC, but production-grade feature selection would likely use stronger validation, repeated seeds, and richer drift monitoring.

### 19.6 Forecast intervals are empirical

Intervals are calibrated from backtest residual quantiles by lag. They are useful for communication and pricing guardrails, but they are not a full probabilistic demand model.

---

## 20. Recommended Next Steps

### Phase 1 - Forecast governance

- Keep `model_comparison_metrics.csv`, `forecast_champion.json`, and audit summaries visible in the UI.
- Add a compact "why this model won" panel for the champion.
- Surface selection folds, audit folds, and audit status in manager-safe language.

### Phase 2 - Stronger model validation

- Add repeated backtests over different cutoffs.
- Compare weekly rolling-origin results to scenario-lag runs.
- Track champion changes over time.
- Add explicit monitoring for recursive forecast flattening or runaway drift.

### Phase 3 - Feature and data improvements

- Add real demand drivers such as events, search interest, channel mix, and rate-shop history once available.
- Keep fake/self-derived external signals out of forecasting.
- Expand cancellation and booking-pace features after validating their live data quality.

### Phase 4 - Production forecast science

- Calibrate prediction intervals more formally.
- Add segment or room-type forecasts.
- Evaluate hierarchical or pooled learning across properties if the system becomes multi-property.
- Introduce champion-challenger governance before automating model replacement.

---

## 21. Suggested Slide Storyline

### Slide 1 - Why forecasting exists

> The RMS needs an expected stay-date occupancy forecast before it can make a pricing recommendation.

### Slide 2 - The system design

> Forecasting is a separate model-governance layer: refresh data, backtest models, select champion, save artifacts, then reuse champion for daily forecasts.

### Slide 3 - Backtesting method

> Rolling-origin weekly backtests simulate repeated historical forecast decisions: train through a cutoff, forecast 30 days, compare to actuals.

### Slide 4 - Model competition

> The current slate includes statistical baselines, SARIMAX, recursive tree/boosting models, and regressor-chain tree/boosting models.

### Slide 5 - Champion result

> `extra_trees_recursive` currently wins with `MAE_pp = 8.7854` and `RMSE_pp = 10.5446` across 49 selection folds.

### Slide 6 - Governance

> The champion is audited on recent folds, interval coverage is calibrated from residuals, and all decisions are saved in CSV/JSON artifacts.

### Slide 7 - What not to oversell

> The forecast layer is data-driven, but it is still a PoC on demo data. Pricing is deterministic and uses the forecast as one input, not as a black-box command.

---

## 22. Likely Questions and Prepared Answers

### Q: What models are part of the backtest?

**A:** The default backtest includes `seasonal_naive_7`, `ewma_14`, `sarimax`, `random_forest_recursive`, `extra_trees_recursive`, `xgboost_recursive`, `random_forest_chain`, `extra_trees_chain`, and `xgboost_chain`.

### Q: What model currently wins?

**A:** The current saved champion is `extra_trees_recursive` using the `boruta_selected` feature profile.

### Q: How is the champion chosen?

**A:** Models are ranked mainly by `MAE_pp`. Any model within 0.50 percentage points of the best MAE is eligible for the RMSE guardrail; the eligible set is sorted by `RMSE_pp`, then `MAE_pp`, `Abs_Bias_pp`, complexity, and model name.

### Q: Why keep simple statistical models if ML models win?

**A:** They are sanity checks. A simple model like `ewma_14` performing strongly tells us whether the ML layer is adding real value or just adding complexity.

### Q: Does Boruta run for every fold?

**A:** No. In the current production-style path, Boruta selects schemas once from the full available history before model competition. Those schemas are reused during fold scoring.

### Q: Is hyperparameter tuning repeated every forecast?

**A:** No. Tuning is saved to `data/model_hyperparameters.json`. Forecast mode reuses the saved artifact when it is current; if missing or stale, the workflow runs backtest mode first.

### Q: What does `MAE_pp` mean?

**A:** It is the average occupancy miss in percentage points. `MAE_pp = 8.7854` means the model missed actual occupancy by about 8.79 occupancy points on average across the selection folds.

### Q: What does a negative `Bias_pp` mean?

**A:** Negative bias means the model under-forecasted occupancy on average. The current champion has `Bias_pp = -3.1363` in the selection split.

### Q: What is the audit split?

**A:** The audit split is the most recent 8 folds, kept separate from model selection to check whether the selected champion degrades on recent data.

### Q: Does the forecast set the price?

**A:** No. The forecast estimates future occupancy. The deterministic pricing optimizer uses it alongside live booked demand, retained demand, market rates, and guardrails.

