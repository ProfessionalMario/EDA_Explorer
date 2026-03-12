


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