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