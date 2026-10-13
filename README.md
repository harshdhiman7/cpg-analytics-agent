# 🛒 CPG Analytics Agent

A local, tool-using AI agent for **CPG / Retail analytics**. Ask questions in plain English and the agent routes them to the right analytical tool, then explains the results in business terms.

Built with **LangGraph** (agent orchestration), **MCP** (tool server), and **Ollama** (local LLM) — no data leaves your machine.

## ✨ Capabilities

| Capability | Tool | Method |
|---|---|---|
| **Price elasticity** (per UPC) | `calculate_price_elasticity` | Multivariate log-log regression controlling for FEATURE, DISPLAY and TPR_ONLY promotions |
| **Demand forecasting** (per category) | `_forecast_demand` | XGBoost regressor on weekly category-level aggregates with holdout evaluation (RMSE, MAPE) |

Example questions:

- *"What is the price elasticity for UPC 3000006610"*
- *"Forecast 8 weeks of demand for BAG SNACKS."*
- *"Calculate elasticity for UPC 3000006610 and forecast COLD CEREAL for 12 weeks."*

## 🏗️ Architecture

```
          ┌──────────────┐
 User ──▶ │  LangGraph   │ ──▶ Ollama (qwen2.5:3b)
          │    agent     │ ◀──  decides which tool to call
          └──────┬───────┘
                 │ tool calls (MCP, stdio)
                 ▼
     ┌─────────────────────────┐
     │  FastMCP server         │
     │  • calculate_price_...  │ ──▶ pandas / sklearn / XGBoost
     │  • _forecast_demand     │ ◀── data/your_retail_data.csv
     └─────────────────────────┘
```

The agent loop is `agent → tools → agent` and ends when the model replies without a tool call.

## 📁 Project structure

```
cpg-analytics-agent/
├── agent.py                          # LangGraph agent + system prompt + CLI
├── mcp_servers/
│   ├── __init__.py
│   └── pricing.py    # FastMCP server exposing the analytics tools
├── config.yaml                       # Data path and forecasting settings
├── data/                             # Put your dataset here (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting started

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the model pulled:

```bash
ollama pull qwen2.5:3b
```

### 2. Install

```bash
git clone harshdhiman7/cpg-analytics-agent
cd cpg-analytics-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your data

Place your retail dataset in `data/` and point `config.yaml` at it:

```yaml
data:
  path: data/retail_data.csv
```

CSV and Excel (`.xlsx`) are both supported.

### 4. Run

Run from the **repo root** — both the agent and the server resolve paths (`config.yaml`, `mcp_servers/...`) relative to the working directory.

```bash
python agent.py
```

You'll see the discovered MCP tools, then be prompted for a question. The agent prints each tool call, its raw result, and the final business summary.

## 📊 Expected data schema

| Column | Used by | Notes |
|---|---|---|
| `UPC` | Elasticity | Product identifier |
| `UNITS` | Both | Units sold (target) |
| `PRICE` | Both | Shelf price |
| `BASE_PRICE` | Forecast | Regular price |
| `FEATURE`, `DISPLAY`, `TPR_ONLY` | Both | 0/1 promotion flags |
| `CATEGORY` | Forecast | Matched case-insensitively |
| `WEEK_END_DATE` | Forecast | Parseable date |
| `DESCRIPTION` | Elasticity | Optional, used as product name |

This matches the layout of the public *Dunnhumby "Carbo-Loading"* retail dataset.

## 🔧 Configuration

`config.yaml` controls the dataset and forecasting behavior:

| Key | Description |
|---|---|
| `data.path` | Dataset location (`.csv` or `.xlsx`) |
| `forecast.horizon` | Default horizon (weeks) when none is supplied |
| `forecast.target_variable` | Column to forecast (default `UNITS`) |
| `forecast.required_data` | Columns that must exist after aggregation |
| `forecast.data_preprocessing.missing_value_strategy` | Per-column fill: `forward_fill` or a constant |

To change the LLM, edit the `ChatOllama(...)` block in `agent.py`.

## ⚠️ Notes & limitations

- **Forecasting is a holdout evaluation.** The model trains on all but the last *N* weeks and predicts those *N* weeks using their known price/promo features. It measures how well the model fits recent history rather than predicting genuinely unseen future weeks.
- **Elasticity is correlational.** Positive or unusually large coefficients should not be read as causal; promotions and price are often confounded.
- **Small models need guardrails.** `qwen2.5:3b` can misroute tool calls; the system prompt is deliberately strict. A larger model (e.g. `qwen2.5:7b`) improves reliability.
- The forecast tool is named `_forecast_demand` (leading underscore). If it doesn't appear under *DISCOVERED MCP TOOLS*, check the server registration.
- Only `category`-level forecasting is currently supported.

## 🗺️ Roadmap

- True out-of-sample (future) forecasting
- UPC- and brand-level forecasting
- Additional MCP servers (promotion lift, assortment)
- Tests and CI

## 📄 License

MIT License

Copyright (c) 2026 Harsh Dhiman

## 👨🏻‍💻 Author 
Harsh Dhiman