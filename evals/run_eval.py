"""Retrieval (and optional answer) evaluation on the generated fixture PDF.

What it measures
----------------
Retrieval — offline, no API key needed:
    The fixture is OCR'd and indexed with the real ingestion code
    (``ingest.build_index``), reopened the way the app opens it
    (``rag.load_index``) and queried with the same retriever. For each
    in-scope question in ``questions.json``:
      hit@1  the expected page is the first page retrieved
      hit@3  the expected page is among the first three unique pages
      MRR    mean reciprocal rank of the expected page

Answers — only when GROQ_API_KEY is set and ``--no-llm`` is not given:
    The grounded query engine (``rag.build_query_engine``) answers every
    question. In-scope answers should not be the refusal sentence and should
    cite the expected page inline as ``[sample_notes.pdf p.N]``; out-of-scope
    questions should get exactly ``prompts.NOT_COVERED``.

Usage (from the repository root):

    python -m evals.run_eval            # retrieval, plus answers if key set
    python -m evals.run_eval --no-llm   # retrieval only
    pytest -m eval tests/test_eval.py -v -s   # gated version (retrieval)

Results are written to evals/results.json (gitignored).
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# Allow ``python evals/run_eval.py`` as well as ``python -m evals.run_eval``.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import EMBED_MODEL, LLM_MODEL, TOP_K
from evals.make_fixture import FIXTURE_NAME, fixture_pages, write_fixture
from ingest import CHUNK_OVERLAP, CHUNK_SIZE, build_index
from prompts import NOT_COVERED, is_not_covered
from rag import (
    build_query_engine,
    build_retriever,
    load_index,
    make_llm,
    resolve_embed_model,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(EVAL_DIR, ".tmp")
QUESTIONS_PATH = os.path.join(EVAL_DIR, "questions.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "results.json")
COLLECTION = "eval_notes"

# Inline citation as requested by the prompt: [<file_name> p.<page>]
CITATION_RE = re.compile(r"\[\s*([^\[\]]+?)\s*,?\s*p\.\s*(\d+)\s*\]")


def load_questions(path=QUESTIONS_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    for q in questions:
        q.setdefault("in_scope", True)
        if q["in_scope"] and not q.get("expected_pages"):
            raise ValueError(f"{q['id']}: in-scope questions need pages")
    return questions


def ranked_pages(nodes):
    """Unique page numbers in retrieval order."""
    pages = []
    for node in nodes:
        label = (node.node.metadata or {}).get("page_label")
        try:
            page = int(label)
        except (TypeError, ValueError):
            continue
        if page not in pages:
            pages.append(page)
    return pages


def score_retrieval(pages, expected):
    rank = next(
        (i + 1 for i, page in enumerate(pages) if page in expected), None
    )
    return {
        "hit_at_1": rank == 1,
        "hit_at_3": rank is not None and rank <= 3,
        "rank": rank,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
    }


def cited_pages(answer, file_name=FIXTURE_NAME):
    """Page numbers the answer cites for ``file_name``."""
    return sorted(
        {
            int(page)
            for name, page in CITATION_RE.findall(answer or "")
            if name.strip() == file_name
        }
    )


def _mean(values):
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def prepare_index(work_dir, embed_model=EMBED_MODEL):
    """Generate the fixture, ingest it, and reopen the index from disk."""
    pdf_dir = os.path.join(work_dir, "pdfs")
    db_dir = os.path.join(work_dir, "chroma_db")
    write_fixture(pdf_dir)
    embed = resolve_embed_model(embed_model)
    build_index(pdf_dir, db_dir, COLLECTION, embed_model=embed)
    # Reopen rather than reuse the in-memory index: this is the app's path.
    return load_index(db_dir, COLLECTION, embed)


def evaluate_retrieval(index, questions, top_k):
    retriever = build_retriever(index, top_k=max(top_k, 3))
    rows = []
    for q in questions:
        if not q["in_scope"]:
            continue
        nodes = retriever.retrieve(q["question"])
        pages = ranked_pages(nodes)
        top_score = (
            round(nodes[0].score, 4)
            if nodes and nodes[0].score is not None
            else None
        )
        row = {
            "id": q["id"],
            "question": q["question"],
            "expected_pages": q["expected_pages"],
            "retrieved_pages": pages,
            "top_score": top_score,
        }
        row.update(score_retrieval(pages, q["expected_pages"]))
        rows.append(row)
    return {
        "n": len(rows),
        "hit_at_1": _mean(r["hit_at_1"] for r in rows),
        "hit_at_3": _mean(r["hit_at_3"] for r in rows),
        "mrr": _mean(r["reciprocal_rank"] for r in rows),
        "rows": rows,
    }


def evaluate_answers(index, questions, top_k, api_key, llm_model=LLM_MODEL):
    llm = make_llm(api_key, llm_model)
    engine = build_query_engine(index, top_k=top_k, llm=llm)
    rows = []
    for q in questions:
        row = {
            "id": q["id"],
            "in_scope": q["in_scope"],
            "question": q["question"],
        }
        try:
            answer = str(engine.query(q["question"]))
        except Exception as exc:  # keep going; report at the end
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        refused = is_not_covered(answer)
        row["answer"] = answer
        row["refused"] = refused
        if q["in_scope"]:
            pages = cited_pages(answer)
            row["cited_pages"] = pages
            row["cites_expected_page"] = not refused and any(
                page in q["expected_pages"] for page in pages
            )
        rows.append(row)

    in_rows = [r for r in rows if r["in_scope"] and "error" not in r]
    out_rows = [r for r in rows if not r["in_scope"] and "error" not in r]
    return {
        "model": llm_model,
        "refusal_sentence": NOT_COVERED,
        "n_in_scope": len(in_rows),
        "answered": _mean(not r["refused"] for r in in_rows),
        "cites_expected_page": _mean(
            r["cites_expected_page"] for r in in_rows
        ),
        "n_out_of_scope": len(out_rows),
        "refused": _mean(r["refused"] for r in out_rows),
        "errors": sum("error" in r for r in rows),
        "rows": rows,
    }


def run(top_k=TOP_K, with_llm=None, keep=False, results_path=RESULTS_PATH):
    """Run the evaluation; returns the results dict (also written to JSON).

    ``with_llm=None`` means "if GROQ_API_KEY is set".
    """
    questions = load_questions()
    api_key = os.getenv("GROQ_API_KEY", "")
    if with_llm is None:
        with_llm = bool(api_key)
    if with_llm and not api_key:
        raise SystemExit("GROQ_API_KEY is not set (use --no-llm).")

    os.makedirs(TMP_DIR, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="run-", dir=TMP_DIR)
    started = time.time()
    try:
        index = prepare_index(work_dir)
        results = {
            "generated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "fixture": FIXTURE_NAME,
            "fixture_pages": len(fixture_pages()),
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": top_k,
            "retrieval": evaluate_retrieval(index, questions, top_k),
            "answers": (
                evaluate_answers(index, questions, top_k, api_key)
                if with_llm
                else None
            ),
        }
        results["seconds"] = round(time.time() - started, 1)
    finally:
        if keep:
            print(f"Kept temporary index at {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    if results_path:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
    return results


def _yes(flag):
    return "yes" if flag else "no"


def print_report(results):
    r = results["retrieval"]
    print(
        f"\nRetrieval on {results['fixture_pages']} fixture pages "
        f"(top_k={results['top_k']}, embed={results['embed_model']}, "
        f"chunk={results['chunk_size']}/{results['chunk_overlap']})"
    )
    print(f"{'id':<5}{'expected':<10}{'retrieved':<14}{'hit@1':<7}hit@3")
    for row in r["rows"]:
        expected = ",".join(map(str, row["expected_pages"]))
        retrieved = ",".join(map(str, row["retrieved_pages"]))
        print(
            f"{row['id']:<5}{expected:<10}{retrieved:<14}"
            f"{_yes(row['hit_at_1']):<7}{_yes(row['hit_at_3'])}"
        )
    print(
        f"\nhit@1 = {r['hit_at_1']:.2f}   hit@3 = {r['hit_at_3']:.2f}   "
        f"MRR = {r['mrr']:.2f}   (n = {r['n']})"
    )

    a = results.get("answers")
    if not a:
        print("\nAnswer checks skipped (no GROQ_API_KEY, or --no-llm).")
        return
    print(
        f"\nAnswers via {a['model']}: in-scope answered "
        f"{a['answered']:.2f}, cites expected page "
        f"{a['cites_expected_page']:.2f} (n = {a['n_in_scope']}); "
        f"out-of-scope refused {a['refused']:.2f} "
        f"(n = {a['n_out_of_scope']}); errors {a['errors']}"
    )
    for row in a["rows"]:
        if "error" in row:
            print(f"  {row['id']}: ERROR {row['error']}")
        elif row["in_scope"]:
            print(
                f"  {row['id']}: cites {row['cited_pages']} "
                f"-> {_yes(row['cites_expected_page'])}"
            )
        else:
            print(f"  {row['id']}: refused -> {_yes(row['refused'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval (and answers) on the fixture PDF."
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="skip the answer checks even if GROQ_API_KEY is set",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K, help=f"default {TOP_K}"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary index under evals/.tmp/",
    )
    parser.add_argument(
        "--results", default=RESULTS_PATH, help="where to write the JSON"
    )
    args = parser.parse_args(argv)

    results = run(
        top_k=args.top_k,
        with_llm=False if args.no_llm else None,
        keep=args.keep,
        results_path=args.results,
    )
    print_report(results)
    print(f"\nWrote {args.results}")


if __name__ == "__main__":
    main()
