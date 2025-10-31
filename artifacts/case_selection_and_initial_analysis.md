## Case selection: Optimizing Race Strategy and Pit-stop Decisions (Motorsports / Transportation)

### Chosen sector
Motorsports (professional racing) — a specialized transportation / sports-technology sector where data-driven strategy yields direct competitive and commercial value.

### Problem statement
Racing teams must decide in real time when to pit (and which tyres/fuel strategy to use) and how to adjust race tactics. Small differences in pit timing and tyre choice frequently determine podium finishes and championship points. The problem: build a machine learning-driven decision-support system to optimize pit-stop timing and race strategy to maximize the probability of achieving the team's objectives (win/podium/points) while controlling cost and risk.

### Market and business drivers
- Competitive performance: teams measure success in points, podiums and wins; small gains translate to large strategic value across a season.
- Sponsor ROI: higher results increase sponsor exposure and value; predictable improvement attracts more sponsorship dollars.
- Cost control: optimizing tyre/fuel usage and reducing avoidable strategy errors reduces operating cost and penalties.
- Fan and broadcast engagement: better strategy and predictive insights enrich broadcast graphics and fan analytics products.
- Betting & media: sportsbooks and broadcasters can license or build products based on predictive models, creating monetization channels.
- Technology transfer: automotive partners (manufacturers/suppliers) benefit from data-driven vehicle and tyre insights.

### Background & urgency
Motorsport is increasingly data-driven. Telemetry, lap times, pit-stop logs and weather forecasts are available in high volume and velocity. Teams that can convert this data into high-confidence in-race decisions gain measurable competitive advantage. Urgency comes from:

- Tight competitive margins: championships often decided by a few points or seconds.
- Rapid season cadence: limited opportunities to iterate between races; early-season improvements compound.
- Regulatory/tyre changes: new regulations or tyre compounds increase uncertainty; teams that adapt quickly benefit.

The project is time-sensitive around race weekends: an operational prototype for in-race use must be reliable, low-latency, and interpretable.

### Stakeholders
- Race strategy team / chief strategist — primary users of decisions and recommendations.
- Race engineers and performance engineers — feed telemetry and validate model outputs.
- Drivers — receive recommendations; need clarity and trust in guidance.
- Team management and sporting directors — evaluate ROI, risk appetite.
- Pit crew / operations — execution depends on chosen strategy.
- Sponsors and commercial partners — interested in performance uplift and visibility metrics.
- Broadcasters / analytics partners — secondary consumers for visualizations and fan products.
- Data engineering and ML teams — build, deploy, and maintain models and pipelines.
- Regulatory bodies / series organizers — may impose constraints (safety/fairness) that affect strategy options.

### Available data sources (within this project)
This repository already contains rich historical race data that can feed analysis and modeling:

- `data/lap_times.csv` — lap-level performance for drivers.
- `data/pit_stops.csv` — pit-stop timings and lap numbers.
- `data/qualifying.csv`, `data/results.csv`, `data/races.csv` — race context and outcomes.
- `data/driver_standings.csv`, `data/constructor_standings.csv` — season-level performance.
- `data/f1_results_joined.csv` — pre-joined results that can speed feature engineering.

Project code components relevant to implementation:
- `etl/` — extract/load/transform scripts to build feature tables.
- `ml/` and `pipelines/` — training scripts and CI for model experiments.

### Project success indicators (technical & business)

Technical KPIs:
- Model predictive performance: AUC/ROC or log-loss for probabilistic outcome models (e.g., predicting position changes after a pit) and RMSE/MAPE for regression targets (time delta predictions).
- Calibration: predicted probability of finishing position matches observed frequencies (important for decision-making under uncertainty).
- Latency: inference time < acceptable in-race window (e.g., sub-second to a few seconds for a given decision request).
- Robustness: model maintains performance across circuits, weather, and tyre compounds.

Business KPIs:
- Race outcome uplift: measurable improvement in expected points per race (e.g., increase in expected points or probability of podium relative to historical baseline).
- Strategy error reduction: fewer strategy decisions that lead to negative outcomes (e.g., lost positions because of bad timing).
- Sponsor exposure / ROI uplift: increased podiums or consistent top finishes leading to higher valuation.
- Operational reliability: % of race weekends where model provided usable recommendations.

Suggested success thresholds (example targets; refine with stakeholders):
- Model: ROC-AUC > 0.80 for discrete outcome predictions; calibration error (Brier score) reduced vs baseline.
- Business: 5–10% relative lift in expected points per race in pilot (short-term target), or measurable increase in podium probability.

### Minimal contract (inputs / outputs / constraints)
- Inputs: current session telemetry (lap times, tyre stint age), live pit stop data, circuit characteristics, weather forecast, competitor states (gaps), tyre compound models, and historical feature store.
- Outputs: ranked pit/tyre strategy options with expected outcome distributions (probability of positions, expected time gain/loss), confidence intervals, and rationale/feature contributions.
- Constraints: safety/regulatory constraints (minimum pit windows, parc fermé rules), maximum recommended risk thresholds, and human-in-the-loop override.

### Key assumptions
- Historical data is representative of near-future races; major rule or tyre compound changes are accounted for or flagged.
- Real-time telemetry and pit-stop logs are available and can be ingested with low latency.
- The strategy team will accept probabilistic recommendations with explainability.

### Edge cases and failure modes
- Sudden safety cars / red flags: models must detect and degrade gracefully or hand over to rule-based heuristics.
- Missing or delayed telemetry: fallback to coarser models or historical averages.
- New circuits or format changes: require rapid retraining or domain adaptation.

### Risks and mitigations
- Overfitting to historical idiosyncrasies — mitigate with cross-season validation and holdout races.
- Trust & adoption — include interpretable outputs and a clear human-in-the-loop workflow with conservative default recommendations.
- Latency/availability — lightweight model variants and edge deployment for low-latency inference.

### Next steps (short-term pilot)
1. Finalize objectives with stakeholders (target KPI definitions and acceptable risk thresholds).
2. Inventory and preprocess data using `etl/transform_data.py` and create a feature table (driver stint features, tyre age, gaps).
3. Prototype offline models in `ml/` to predict short-horizon outcomes (position change after pit, time delta distributions).
4. Validate model with backtesting on historical races and measure uplift vs simple heuristics.
5. Build a low-latency inference wrapper (REST or socket) and integrate with strategy UI or dashboard for a race-simulation pilot.

### How this repository helps
This project already contains the necessary historical data and ETL/ML scaffolding (`etl/`, `ml/`, `pipelines/`) to implement the pilot quickly. Use the `f1_results_joined.csv` and `pit_stops.csv` as the initial feature source and extend the `etl/` scripts to produce per-stint features required by the strategy model.

---
Document created as an initial analysis for a motorsport team use-case: optimizing pit-stop and race strategy using ML. Adjust thresholds and scope after stakeholder interviews.

Last updated: 2025-10-31
