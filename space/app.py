"""
Step 11 - Demo: German Enterprise Knowledge Copilot (Hugging Face Space, ZeroGPU).

ZeroGPU in short: the Space runs on a normal machine; a GPU is attached only while a
function decorated with @spaces.GPU runs. Models are loaded once at startup and moved
with .to("cuda"); the move actually happens when a GPU is attached.

Local test on a laptop (CPU, small models, no GPU needed):
    LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct RERANKER=small python space/app.py
"""
import json
import os
import sys
import traceback
from pathlib import Path
from urllib.parse import quote

try:                                    # 'spaces' exists only on Hugging Face; import it before CUDA use
    import spaces
    gpu = spaces.GPU(duration=30)
    ON_SPACES = True
except ImportError:
    gpu = lambda f: f                   # noqa: E731  (on a laptop the decorator does nothing)
    ON_SPACES = False

import gradio as gr                     # noqa: E402
import torch                            # noqa: E402

# Works both in the Space (app.py next to src/) and in the GitHub repo (space/app.py).
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import NO_ANSWER, Generator          # noqa: E402
from pipeline import RAGPipeline                   # noqa: E402

LLM = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
RERANKER = os.getenv("RERANKER", "large")
DOCS = ROOT / "data" / "docs"

# ---------------------------------------------------------------- load everything once
with open(ROOT / "data" / "chunks.jsonl", encoding="utf-8") as f:
    CHUNKS = [json.loads(line) for line in f]

use_gpu = ON_SPACES or torch.cuda.is_available()
generator = Generator(LLM, device_map=None, dtype=torch.bfloat16 if use_gpu else torch.float32)
# ZeroGPU rule (found by bisection): at startup only LOAD models and move them with .to("cuda").
# Running any model at startup (e.g. encoding the chunks) breaks the GPU worker, so the chunk
# embeddings are computed lazily on the first question, inside the @spaces.GPU function.
rag = RAGPipeline(CHUNKS, generator, reranker_key=RERANKER, device="cpu", lazy_index=True)
if use_gpu:
    generator.model.to("cuda")
    rag.reranker.model.to("cuda")
    rag.hybrid.dense.model.to("cuda")


@gpu
def run_pipeline(question):
    """Everything that needs the GPU happens inside this one call."""
    return rag.answer(question)


# ---------------------------------------------------------------- presentation helpers
def pdf_link(doc, page=None):
    """Link that opens the PDF in the browser - at the cited page if given."""
    url = f"/gradio_api/file={quote(str(DOCS / doc))}" + (f"#page={page}" if page else "")
    return url


def confidence_html(score):
    # Thresholds from the Step 10 analysis: above 0.5 about 91% of answers were correct, below about 65%.
    level, label, hint = (("high", "hoch", "") if score >= 0.5 else
                          ("mid", "mittel", " – bitte anhand der Quelle prüfen") if score >= 0.1 else
                          ("low", "niedrig", " – bitte anhand der Quelle prüfen"))
    return (f'<div class="conf conf-{level}"><div class="track"><div class="fill" '
            f'style="width:{max(score, 0.02) * 100:.0f}%"></div></div>'
            f'<span>Retrieval confidence: <b>{score:.2f}</b> ({label}){hint}</span></div>')


def sources_html(result):
    if result["abstained"] or not result["citations"]:
        return ""
    seen, items = set(), []
    for c in result["citations"]:
        if c["page_id"] in seen:
            continue
        seen.add(c["page_id"])
        items.append(f'<li><a href="{pdf_link(c["doc"], c["page"])}" target="_blank">'
                     f'📄 {c["doc"]} — Seite {c["page"]}</a></li>')
    return "<div class='sources'><h4>Quellen</h4><ul>" + "".join(items) + "</ul></div>"


def passages_md(result):
    parts = []
    for n, p in enumerate(result["passages"], start=1):
        parts.append(f"**[{n}] {p['doc']} — Seite {p['page']} · {p['title']}** "
                     f"(Relevanz {p['score']:.2f})\n\n> {p['text']}")
    return "\n\n".join(parts)


def ask(question):
    question = (question or "").strip()
    if not question:
        return "Bitte geben Sie eine Frage ein.", "", "", "", {}
    try:
        result = run_pipeline(question)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()                           # full details go to the Space logs
        return ("Es ist ein technischer Fehler aufgetreten. Bitte versuchen Sie es in einem Moment erneut.",
                "", "", "", {})
    answer = NO_ANSWER + " Wenden Sie sich bei Bedarf an den IT-Servicedesk (Durchwahl 4444)." \
        if result["abstained"] else result["raw_answer"]
    timing = (f"<span class='timing'>Suche {result['t_retrieve']:.1f} s · "
              f"Antwort {result['t_generate']:.1f} s</span>")
    state = {"question": question, "answer": result["raw_answer"]}
    return (answer, sources_html(result), confidence_html(result["top_score"]) + timing,
            passages_md(result), state)


def feedback(state, helpful):
    if not state:
        return "Bitte zuerst eine Frage stellen."
    # Printed to the Space logs. (A persistent store would need a dataset repo - out of scope here.)
    print(f"FEEDBACK helpful={helpful} | {state['question']} | {state['answer'][:200]}", flush=True)
    return "Danke für Ihr Feedback!"


HANDBOOKS = "".join(f'<li><a href="{pdf_link(p.name)}" target="_blank">📄 {p.name}</a></li>'
                    for p in sorted(DOCS.glob("*.pdf"), key=lambda p: p.name))

CSS = """
.gradio-container {max-width: 820px !important; margin: 0 auto;}
#title h1 {margin-bottom: 0.2rem;}
#answer {font-size: 1.05rem; line-height: 1.6;}
.sources h4 {margin: 0.6rem 0 0.3rem;}
.sources ul, .books ul {list-style: none; padding-left: 0; margin: 0;}
.sources li, .books li {margin: 0.25rem 0;}
.conf {display: flex; align-items: center; gap: 0.7rem; margin: 0.5rem 0;}
.conf .track {width: 150px; height: 6px; border-radius: 3px; background: var(--border-color-primary); overflow: hidden;}
.conf .fill {height: 100%;}
.conf-high .fill {background: #2e7d5b;} .conf-mid .fill {background: #b26b00;} .conf-low .fill {background: #a33a3a;}
.timing {font-size: 0.85rem; opacity: 0.7;}
"""

# ---------------------------------------------------------------- the interface (as in the mockup)
with gr.Blocks(title="Deutscher Enterprise Knowledge Copilot") as demo:
    gr.Markdown("# Deutscher Enterprise Knowledge Copilot\n"
                "Fragen zur IT-Dokumentation des **fiktiven** Klinikum Musterstadt. "
                "Jede Antwort nennt Dokument und Seite – ein Klick öffnet die Quelle.", elem_id="title")
    state = gr.State({})
    question = gr.Textbox(label="Frage", placeholder="Wie beantrage ich Zugriff auf das PACS?", lines=2)
    ask_btn = gr.Button("Frage stellen", variant="primary")
    gr.Examples(["Wie beantrage ich Zugriff auf das PACS?", "KIS geht nicht",
                 "Komische Mail mit Link bekommen, was mache ich?",
                 "Wie lange werden die VPN-Protokolle aufbewahrt?",
                 "Wie drucke ich in Farbe?"], inputs=question)

    gr.Markdown("### Antwort")
    answer = gr.Markdown(elem_id="answer")
    sources = gr.HTML()
    confidence = gr.HTML()
    with gr.Accordion("Quellen anzeigen", open=False):
        passages = gr.Markdown()
    with gr.Accordion("Feedback geben", open=False):
        with gr.Row():
            up = gr.Button("👍 Hilfreich")
            down = gr.Button("👎 Nicht hilfreich")
        fb = gr.Markdown()
    with gr.Accordion("Alle Handbücher (PDF)", open=False):
        gr.HTML(f"<div class='books'><ul>{HANDBOOKS}</ul></div>")
    gr.Markdown(f"<small>Pipeline: BM25 + multilingual-e5-small (hybrid) → bge-reranker-v2-m3 → {LLM}. "
                "Alle Dokumente sind synthetisch; es werden keine echten Patientendaten verwendet.</small>")

    outputs = [answer, sources, confidence, passages, state]
    ask_btn.click(ask, question, outputs)
    question.submit(ask, question, outputs)
    up.click(lambda s: feedback(s, True), state, fb)
    down.click(lambda s: feedback(s, False), state, fb)

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(DOCS)], css=CSS,
                theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"))
