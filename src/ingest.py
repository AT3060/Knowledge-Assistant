"""
Step 3 - Read the PDFs and turn them into searchable chunks.

In a real company you only get the PDFs. So from here on we treat the PDFs as the source
of truth, and use data/pages.jsonl only to CHECK that our extraction is correct.

Output:  data/chunks.jsonl   one line per chunk:
         {"chunk_id", "page_id", "doc", "page", "title", "text", "n_words"}

Run:     python src/ingest.py              (defaults: max 60 words, 1 sentence overlap)
         python src/ingest.py --max-words 40
"""
import argparse
import difflib
import json
import re
from pathlib import Path

from pypdf import PdfReader

DATA = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- 1. extraction
def extract_pages(pdf_path):
    """Return [(page_number, raw_text)] for every page. Page numbers start at 1, like in a PDF viewer."""
    reader = PdfReader(pdf_path)
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


# ---------------------------------------------------------------- 2. cleaning
HEADING = re.compile(r"^\d+\.\s+\S")      # e.g. "2. Zugriff beantragen"


def is_content_page(raw):
    """A content page starts with a numbered section heading. That rule skips the title page.
    The table of contents also starts with numbered lines, so we skip pages with dotted leaders '....'."""
    first_line = raw.strip().split("\n", 1)[0]
    return bool(HEADING.match(first_line)) and raw.count("....") < 3


def split_heading(raw):
    """The first line of a content page is the section heading, e.g. '2. Zugriff beantragen'.
    Return (title without number, rest of the page)."""
    first, _, rest = raw.strip().partition("\n")
    title = re.sub(r"^\d+\.\s*", "", first).strip()
    return title, rest


def clean(text):
    """A PDF stores lines, not paragraphs. Re-join the lines into running text.
    A line ending in '-' is ambiguous in German:
      'Self-Service-' + 'Portal'  -> one word, join WITHOUT a space  (next word is capitalised)
      'Planungs-'     + 'und ...' -> 'Planungs- und', keep the space (next word is lower case)"""
    text = re.sub(r"-\n(?=[A-ZÄÖÜ0-9])", "-", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()     # collapse repeated whitespace


# ---------------------------------------------------------------- 3. chunking
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ„(\[])")


def split_sentences(text):
    """Split after . ! ? when the next word starts with a capital letter.
    Simple rule; it would wrongly split abbreviations like 'z. B.' (our corpus avoids them)."""
    return [s for s in SENTENCE_END.split(text) if s]


def chunk_sentences(sentences, max_words, overlap_sentences):
    """Group whole sentences into chunks of at most ~max_words words.
    Consecutive chunks share `overlap_sentences` sentences, so an answer that sits
    at a chunk border is still fully contained in at least one chunk."""
    chunks, current = [], []
    for sent in sentences:
        n_current = sum(len(s.split()) for s in current)
        if current and n_current + len(sent.split()) > max_words:
            chunks.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sent)
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


# ---------------------------------------------------------------- 4. pipeline
def build_chunks(max_words, overlap_sentences):
    chunks, skipped, extracted = [], [], {}
    for pdf in sorted((DATA / "docs").glob("*.pdf")):
        for page_no, raw in extract_pages(pdf):
            page_id = f"{pdf.name}#{page_no}"
            if not is_content_page(raw):
                skipped.append(page_id)
                continue
            title, body = split_heading(raw)
            body = clean(body)
            extracted[page_id] = body
            pieces = chunk_sentences(split_sentences(body), max_words, overlap_sentences)
            for k, piece in enumerate(pieces):
                chunks.append(dict(chunk_id=f"{page_id}#c{k}", page_id=page_id, doc=pdf.name,
                                   page=page_no, title=title, text=piece,
                                   n_words=len(piece.split())))
    return chunks, skipped, extracted


def validate(extracted):
    """Compare our extracted text with the original text from build_corpus.py.
    A similarity of 1.0 means the PDF round trip lost nothing."""
    truth = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(DATA / "pages.jsonl", encoding="utf-8")}
    missing = set(truth) - set(extracted)
    extra = set(extracted) - set(truth)
    scores = {pid: difflib.SequenceMatcher(None, extracted[pid], truth[pid]).ratio()
              for pid in truth if pid in extracted}
    return missing, extra, scores, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--overlap", type=int, default=1, help="sentences shared by neighbouring chunks")
    args = ap.parse_args()

    chunks, skipped, extracted = build_chunks(args.max_words, args.overlap)
    with open(DATA / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- report
    n_pages = len({c["page_id"] for c in chunks})
    words = sorted(c["n_words"] for c in chunks)
    split_pages = len({c["page_id"] for c in chunks if not c["chunk_id"].endswith("#c0")})
    print(f"Skipped {len(skipped)} pages (title + table of contents)")
    print(f"{n_pages} content pages -> {len(chunks)} chunks  "
          f"({split_pages} pages were split into more than one chunk)")
    print(f"Words per chunk: min {words[0]}, median {words[len(words) // 2]}, max {words[-1]}")

    missing, extra, scores, truth = validate(extracted)
    worst = sorted(scores, key=scores.get)[:3]
    print(f"\nValidation against pages.jsonl: {len(missing)} missing, {len(extra)} unexpected pages")
    print(f"Text similarity: mean {sum(scores.values()) / len(scores):.4f}, "
          f"min {min(scores.values()):.4f}")
    for pid in worst:
        if scores[pid] < 1.0:
            diff = [d for d in difflib.ndiff(truth[pid].split(), extracted[pid].split()) if d[0] in "+-"]
            print(f"  {pid}  {scores[pid]:.4f}  differences: {diff[:6]}")

    example = next(c for c in chunks if c["page_id"] == "PACS_Handbuch.pdf#4")
    print("\nExample chunk:\n" + json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
