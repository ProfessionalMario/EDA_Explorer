1️⃣ MetadataAgent ✔ (done)
2️⃣ DataFrameAgent ✔ (done)
3️⃣ VisualizationAgent ← NEXT
4️⃣ TransformAgent
5️⃣ LLM Planner

- add transform agent
- add fill_missing
- add convert_numeric
- add histogram plot
- integrate LLM planner


Issue: operation planner schema
Issue: dataset detection
Issue: transformation agent
Issue: visualization agent


project/
│
├─ agents/
│   ├─ metadata_agent.py
│   ├─ dataframe_agent.py
│   ├─ transform_agent.py
│
├─ core/
│   ├─ router.py
│   ├─ executor.py
│
├─ data/
│
├─ experiments/
│   ├─ test_operations.py
│   ├─ llm_planner_test.py
│   ├─ column_matching_test.py
│
├─ main.py
└─ README.md



## Transformer agent
fill missing
drop missing
convert types
rename columns
remove duplicates







eda_explorer/
│
├── main.py                 # entrypoint
├── cli.py                  # GPT-style CLI interface
│
├── config/
│   ├── settings.py
│   └── model_config.py
│
├── core/
│   ├── orchestrator.py     # routes user requests
│   ├── session_state.py    # remembers variables/context
│   └── memory.py           # chat history / summaries
│
├── agents/
│   ├── query_agent.py      # understand user intent
│   ├── code_agent.py       # generate pandas code
│   ├── viz_agent.py        # generate visualization code
│   └── metadata_agent.py   # schema & dataset understanding
│
├── execution/
│   ├── sandbox.py          # controlled code execution
│   ├── validator.py        # syntax & security checks
│   └── runner.py           # executes pandas code
│
├── data/
│   ├── loader.py           # CSV/Excel/Parquet ingestion
│   ├── metadata.py         # dataset schema extractor
│   └── dataframe_store.py  # manages multiple dataframes
│
├── llm/
│   ├── llm_client.py       # model interface
│   ├── prompts.py          # prompt templates
│   └── embeddings.py       # optional RAG embeddings
│
├── visualization/
│   ├── plotter.py          # matplotlib/seaborn wrapper
│   └── render.py           # output plots to CLI
│
├── guardrails/
│   ├── code_security.py
│   ├── column_validator.py
│   └── output_validator.py
│
├── utils/
│   ├── logger.py
│   ├── exceptions.py
│   └── helpers.py
│
├── logs/
│
└── tests/
















CLI
↓
Query Agent
↓
Orchestrator
↓
(Deterministic Function OR LLM Agent)
↓
Code Validator
↓
Sandbox Execution
↓
DataFrame Engine
↓
Result 




Current problems: 
missing values, encoding, drop duplicates not working properly. 
delete duplicates and encode, this doesn't work. 
encode gender into numbers, country into cat, this doesn't work. 
Visulization is not running in terminal. 



ollama 
git
venv


# EDA Explorer – AI-Powered Data Analysis CLI

## ✅ Core Features
- [x] Load dataset (CSV → Parquet)
- [x] Dataset registry system
- [x] Analyze command
- [x] Missing value detection
- [x] Duplicate detection
- [x] Column type detection
- [x] Warnings system

## 📊 Analysis Features
- [x] Correlation analysis
- [x] Outlier detection (IQR)
- [x] Feature importance (auto target detection)
- [x] High-cardinality filtering
- [x] ID column detection

## 📈 Visualization
- [x] Histogram
- [x] Bar chart

## 📁 Reporting
- [x] Export analysis to .txt
- [ ] Export to JSON (optional)
- [ ] Export to HTML (optional)

## 🧠 AI Layer (IMPORTANT)
- [x] Auto target selection
- [x] Feature importance explanation
- [ ] RAG-based suggestions (missing values, outliers)

## ⚡ Performance
- [x] Parquet storage
- [x] Large dataset handling (sampling)
- [ ] Chunk processing (future)

## 🛠️ System Design
- [x] Command handler
- [x] Registry system
- [x] Modular agents (AnalysisAgent, etc.)
- [x] Logger integration

## 🎬 Demo Preparation
- [ ] CLI demo recording
- [ ] Feature walkthrough
- [ ] Large dataset example

## 📦 Datasets
- [x] Titanic
- [x] Customer Churn
- [x] Credit Card Fraud
- [ ] Add 1 more strong dataset

## 🚀 Future Enhancements
- [ ] RAG-based EDA advisor
- [ ] SQL query generator
- [ ] Model training pipeline
- [ ] Web UI

## 🧪 Testing
- [ ] Test on small dataset
- [ ] Test on large dataset
- [ ] Edge cases (empty, high missing)
