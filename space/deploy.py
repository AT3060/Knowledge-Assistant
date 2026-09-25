"""
Upload the demo to a Hugging Face Space (the Space must already exist: Gradio SDK, ZeroGPU).

It collects exactly the files the app needs into one folder and uploads them in one commit:
    app.py, README.md, requirements.txt      (from space/)
    src/*.py                                  (the pipeline modules)
    data/chunks.jsonl, data/docs/*.pdf        (the knowledge base)

Run from the project root:   python space/deploy.py --repo Areftawana3/Knowledge-Assistant
Needs a Hugging Face WRITE token in the environment variable HF_TOKEN.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SRC_FILES = ["bm25.py", "dense.py", "hybrid.py", "rerank.py", "generate.py", "pipeline.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. Areftawana3/Knowledge-Assistant")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("app.py", "README.md", "requirements.txt"):
            shutil.copy(ROOT / "space" / name, stage / name)
        (stage / "src").mkdir()
        for name in SRC_FILES:
            shutil.copy(ROOT / "src" / name, stage / "src" / name)
        (stage / "data" / "docs").mkdir(parents=True)
        shutil.copy(ROOT / "data" / "chunks.jsonl", stage / "data" / "chunks.jsonl")
        for pdf in (ROOT / "data" / "docs").glob("*.pdf"):
            shutil.copy(pdf, stage / "data" / "docs" / pdf.name)

        files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
        print(f"Uploading {len(files)} files to spaces/{args.repo}:")
        for f in files:
            print("  ", f)
        HfApi().upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="space",
                              commit_message="Deploy demo from GitHub project")
    print(f"\nDone. Open https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
