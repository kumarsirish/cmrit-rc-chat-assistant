"""Single entry point covering every stage of the RAG pipeline —
both the CLI (`python main.py <stage>`) and the Streamlit dashboard
(`streamlit run main.py`, or `python main.py ui`, which just shells
out to that).

Each subcommand maps to one numbered package under rag/:

    01_ingestion -> ingest     load PDFs, chunk, embed
    02_indexing  -> ingest     write embedded chunks to Chroma
    03_routing   -> route      show which topic/document a query maps to
    04_retrieval -> retrieve   run retrieval only, print the matched chunks
    05_generation
    06_response  -> query      full retrieval + LLM answer + citations
    07_evaluation-> evaluate   score retrieval against the labeled test set
    08_auth      -> auth       validate/normalize a user email
    09_common    -> config     print the effective AppConfig
    10_ui        -> ui         launch the Streamlit dashboard

`ingest` builds the whole index in one pass (01 and 02 are one
pipeline — see helper_scripts/ingest.py), and `query` runs 05+06
together (see rag/_05_generation/rag_pipeline.py). Stage wiring itself
lives in rag/pipeline_factory.py, shared with helper_scripts/.

Streamlit re-executes its target script from scratch on every
interaction with no CLI args of its own, which the argparse-based CLI
below can't satisfy — so when this file is run under `streamlit run`,
streamlit.runtime.exists() is used to detect that and skip straight to
the dashboard instead of parsing argv as a subcommand (see the
__main__ guard at the bottom).
"""

import argparse
import dataclasses
import json
import subprocess
import sys

from rag._09_common.config import load_config
from rag.pipeline_factory import build_retrieval_service, build_rag_pipeline


def cmd_ingest(args) -> None:
    from helper_scripts.ingest import run_ingestion

    run_ingestion()


def cmd_route(args) -> None:
    retrieval_service, _ = build_retrieval_service(load_config())
    matched_topic = retrieval_service.match_topic(args.question)
    if matched_topic is None:
        print("No topic matched above the similarity threshold.")
        return
    print(f"Topic: {matched_topic.topic}")
    print(f"Link: {matched_topic.link}")
    if matched_topic.docs_filename:
        print(f"Locally-indexed document(s): {', '.join(matched_topic.docs_filename)}")


def cmd_retrieve(args) -> None:
    retrieval_service, _ = build_retrieval_service(load_config())
    chunks, matched_topic = retrieval_service.retrieve(args.question)
    print(f"Matched topic: {matched_topic.topic if matched_topic else None}")
    for rank, chunk in enumerate(chunks, start=1):
        print(f"\n[{rank}] {chunk.source_file} (page {chunk.page_number})")
        print(chunk.text[:300])


def cmd_query(args) -> None:
    pipeline = build_rag_pipeline(load_config())
    result = pipeline.answer(args.question)
    print(result.answer_text)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"- {source.source_file}")


def cmd_evaluate(args) -> None:
    from rag._07_evaluation.evaluate_retrieval import evaluate_retrieval

    config = load_config()
    retrieval_service, _ = build_retrieval_service(config)
    with open(config.evaluation.test_data_path, "r", encoding="utf-8") as test_file:
        test_cases = [case for case in json.load(test_file) if case.get("relevant_sources")]

    result = evaluate_retrieval(retrieval_service, test_cases, k=config.retrieval.top_k)
    print(json.dumps({"k": result["k"], "mean_scores": result["mean_scores"]}, indent=2))


def cmd_auth(args) -> None:
    from rag._08_auth.user_identity import is_valid_email, normalize_email

    if not is_valid_email(args.email):
        print(f"{args.email!r} is not a valid email address.")
        return
    print(normalize_email(args.email))


def cmd_config(args) -> None:
    config = load_config()
    database = dataclasses.replace(config.database, password="***")
    print(dataclasses.replace(config, database=database))


def cmd_ui(args) -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=True)


STAGES = {
    "ingest": ("01_ingestion + 02_indexing: build the vector index", cmd_ingest, None),
    "route": ("03_routing: show the topic a question maps to", cmd_route, "question"),
    "retrieve": ("04_retrieval: run retrieval only", cmd_retrieve, "question"),
    "query": ("05_generation + 06_response: full RAG answer", cmd_query, "question"),
    "evaluate": ("07_evaluation: score retrieval against the test set", cmd_evaluate, None),
    "auth": ("08_auth: validate/normalize a user email", cmd_auth, "email"),
    "config": ("09_common: print the effective AppConfig", cmd_config, None),
    "ui": ("10_ui: launch the Streamlit dashboard", cmd_ui, None),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    for name, (help_text, func, positional_arg) in STAGES.items():
        subparser = subparsers.add_parser(name, help=help_text)
        if positional_arg:
            subparser.add_argument(positional_arg)
        subparser.set_defaults(func=func)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    import streamlit.runtime

    if streamlit.runtime.exists():
        from rag._10_ui.dashboard import main as run_dashboard

        run_dashboard()
    else:
        main()
