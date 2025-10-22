# Agentic Driver-Based Planning — v14
Adds **Agents + Scenarios** to the simplified v13.3.1 app.

## New
- **Scenarios** tab: run agents (Profiler → Driver Inference → Forecaster → Evaluator), save scenario JSON + Parquet results.
- **Compare** tab: pick two scenarios, see KPIs & deltas.
- Persists to `data/` (created at runtime).

## Keep
- Robust CSV loader, per-combo drivers, months-across hierarchical P&L, actuals cutoff lock, manual edit apportioning.
