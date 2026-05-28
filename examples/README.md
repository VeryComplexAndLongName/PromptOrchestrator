# examples

This folder contains small, executable prompt_orchestrator usage examples with measurable outputs: stats, safety indicators, RAG impact, and prompt-evolution snapshots.

## Run

From the project root:

```bash
python examples/basic_stats_example.py
python examples/multi_turn_metrics_example.py
python examples/safety_metrics_example.py
python examples/rag_metrics_example.py
python examples/summary_limits_example.py
```

## What Each Example Shows

- basic_stats_example.py: single-request metrics (tokens/chars by section, efficiency, safety score).
- multi_turn_metrics_example.py: aggregated KPI across multiple turns (average tokens, warnings, severity distribution).
- safety_metrics_example.py: safe vs injection-like inputs and safety-engine behavior.
- rag_metrics_example.py: side-by-side comparison of use_rag=False vs use_rag=True on a real PyPI RAG database.
- summary_limits_example.py: how summary appears and how sections are compacted when prompt limits are tight.

## RAG Example Prerequisites

- Default rag_orchestrator source: D:\Prog\AI\RagOrchestrator\src
- Default DB: D:\Prog\AI\RagOrchestrator\scripts\pypi_demo\pypi.sqlite (table pypi_chunks)
- Ollama available for embeddings: http://localhost:11434
- Embedding model: nomic-embed-text:latest

Explicit run with paths:

```bash
python examples/rag_metrics_example.py ^
  --rag-src "D:\Prog\AI\RagOrchestrator\src" ^
  --rag-db "D:\Prog\AI\RagOrchestrator\scripts\pypi_demo\pypi.sqlite" ^
  --table-name pypi_chunks ^
  --embed-model nomic-embed-text:latest ^
  --ollama-url http://localhost:11434
```

## Compact Prompt Diff Table

The table below shows what changed across steps, section by section.

| Scenario | Step | Added | Removed / Trimmed |
|---|---|---|---|
| basic_stats_example | single turn | user message | none |
| multi_turn_metrics_example | turn 2 vs turn 1 | 1 prior user message in Recent Messages | none |
| multi_turn_metrics_example | turn 3 vs turn 2 | another prior user message in Recent Messages | none |
| rag_metrics_example | RAG ON vs RAG OFF | retrieved package chunks in Relevant Docs (RAG) | placeholder text "RAG disabled or no relevant docs." |
| safety_metrics_example | unsafe input | injection-like user content enters prompt text | sanitized prompt masks dangerous patterns |
| summary_limits_example | turn 3+ under tight limits | generated Summary section with condensed transcript | Recent Messages and Summary sections are truncated to fit max chars/tokens |

## Prompt Snapshots and Evolution

### Multi-turn (no RAG)

Turn 1:

```text
Summary (fixed format):
No summary yet.

Recent Messages (window):
No recent messages.
```

Turn 2:

```text
Summary (fixed format):
No summary yet.

Recent Messages (window):
user: Сделай план измерения retention для SaaS-продукта.
```

Turn 3:

```text
Summary (fixed format):
No summary yet.

Recent Messages (window):
user: Сделай план измерения retention для SaaS-продукта.
user: Добавь KPI для активации в первую неделю.
```

### RAG OFF vs RAG ON

RAG OFF:

```text
Relevant Docs (RAG):
RAG disabled or no relevant docs.
```

RAG ON:

```text
Relevant Docs (RAG):
Package: pydantic Version: 2.13.4 Key URLs: ...
Package: pydantic Project URLs: ...
```

### Summary under tight limits

Turn 1:

```text
Summary (fixed format):
No summary yet.
```

Turn 2:

```text
Summary (fixed format):
No summary yet.
```

Turn 3:

```text
Summary (fixed format):
user: We need a launch checklist for the billing service: ... assistant: Acknowledged. I will preserve decision...
```

What changed:

- Summary starts appearing once summary_trigger_messages is reached.
- With strict max_prompt_chars/max_prompt_tokens, the fitter trims Recent Messages first and then shortens Summary if needed.

## Example Run Results

Collected from actual runs in this environment.

### 1) basic_stats_example.py

- total_chars: 508
- total_tokens: 110
- tokens_by_section: {'static': 63, 'summary': 9, 'recent': 24, 'rag': 14}
- efficiency_score: 1.0
- safety_score: 1.0
- warnings: []

### 2) multi_turn_metrics_example.py

- turns: 4
- avg_tokens: 130.75
- max_tokens: 180
- avg_efficiency_score: 0.9250
- avg_safety_score: 1.0000
- total_warnings: 2
- safety_severity_distribution: {'none': 4, 'low': 0, 'medium': 0, 'high': 0}

### 3) safety_metrics_example.py

- case 1 (safe): severity=none, is_safe=True, issues_count=0, safety_score=1.0
- case 2 (injection): severity=high, is_safe=False, issues_count=2, safety_score=0.1
- case 3 (injection): severity=high, is_safe=False, issues_count=4, safety_score=0.1
- unsafe cases produced sanitized_prompt output.

### 4) rag_metrics_example.py

RAG OFF:

- turn 1: total_tokens=104, rag_tokens=14, rag_share=13.46%, efficiency=1.0000
- turn 2: total_tokens=120, rag_tokens=14, rag_share=11.67%, efficiency=1.0000
- turn 3: total_tokens=135, rag_tokens=14, rag_share=10.37%, efficiency=0.8500

RAG ON:

- turn 1: total_tokens=502, rag_tokens=412, rag_share=82.07%, efficiency=0.8500
- turn 2: total_tokens=682, rag_tokens=576, rag_share=84.46%, efficiency=0.8500
- turn 3: total_tokens=415, rag_tokens=294, rag_share=70.84%, efficiency=0.8500

Delta (RAG ON - RAG OFF):

- avg_total_tokens_delta: 413.33
- avg_rag_tokens_delta: 413.33
- avg_rag_share_delta: 68.48%
- avg_efficiency_delta: -0.1000
- warnings_delta: 5

### 5) summary_limits_example.py

- turn 1: total_tokens=116, summary_present=False, summary_len=0, summary_truncated=False, recent_truncated=False
- turn 2: total_tokens=154, summary_present=True, summary_len=220, summary_truncated=False, recent_truncated=False
- turn 3: total_tokens=162, summary_present=True, summary_len=220, summary_truncated=True, recent_truncated=True
- turn 4: total_tokens=166, summary_present=True, summary_len=220, summary_truncated=True, recent_truncated=True
- turn 5: total_tokens=177, summary_present=True, summary_len=220, summary_truncated=True, recent_truncated=True

Key observation:

- This run demonstrates both summary generation and compaction behavior when limits are reached.
