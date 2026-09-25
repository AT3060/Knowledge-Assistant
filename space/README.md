---
title: Deutscher Enterprise Knowledge Copilot
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: German RAG for hospital IT docs with page citations
models:
  - Qwen/Qwen2.5-7B-Instruct
  - BAAI/bge-reranker-v2-m3
  - intfloat/multilingual-e5-small
---

# Deutscher Enterprise Knowledge Copilot

Ask a question in German about the IT documentation of the fictional *Klinikum Musterstadt*.
The answer is generated only from the 12 handbooks, and every source link opens the PDF at the cited page.

**Pipeline:** BM25 + multilingual-e5-small (weighted hybrid) → bge-reranker-v2-m3 (top 20 → top 5)
→ Qwen2.5-7B-Instruct with a grounded prompt and [n] citations → document + page.

**Evaluation (130 answerable + 12 unanswerable test questions):** Recall@5 0.949 after reranking,
83.8% end-to-end correct answers (LLM judge, 95% CI 77–90%), 12/12 unanswerable questions refused.

Code, data and full evaluation: https://github.com/AT3060/Knowledge-Assistant

All documents are synthetic. No real hospital or patient data is used.
