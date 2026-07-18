"""
Runs the test question set against the live RAG pipeline and scores:
1. Refusal accuracy - does it correctly say "I don't know" for
   out-of-scope questions, and correctly NOT refuse for in-scope ones?
2. Answer correctness - for answerable questions, does the generated
   answer actually match the expected answer? (judged by the LLM itself)

Run with: python -m evaluation.run_eval
(Run from the project root, with the venv active and an index already
built from data/raw, OR point it at a saved index via app/config.py)
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import INDEX_DIR
from app.retrieval.vectorstore import load_vectorstore
from app.generation.qa_chain import answer_question, get_llm

REFUSAL_PHRASES = [
    "don't know", "do not know", "not stated", "cannot find",
    "not contained", "no information", "not mentioned", "not provided",
]


def looks_like_refusal(answer: str) -> bool:
    """Simple keyword check for whether the model refused to answer."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def judge_correctness(question, expected_answer, actual_answer, llm):
    """
    Uses the LLM itself as a judge: does the actual answer correctly
    convey the expected answer's information? Returns True/False.
    """
    judge_prompt = f"""You are grading whether an AI's answer is factually
correct compared to a known correct answer. Respond with ONLY "YES" or "NO".

Question: {question}
Expected answer: {expected_answer}
AI's actual answer: {actual_answer}

Does the AI's answer correctly convey the same information as the expected
answer, even if worded differently? Respond with ONLY "YES" or "NO"."""

    response = llm.invoke(judge_prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return "YES" in content.upper()


def run_evaluation():
    vectorstore = load_vectorstore(INDEX_DIR)
    if vectorstore is None:
        print("No saved index found. Build an index in the app first "
              "(upload test-handbook.txt and click 'Build / Rebuild Index'), "
              "then run this script again.")
        return

    with open(
        os.path.join(os.path.dirname(__file__), "test_questions.json"),
        encoding="utf-8",
    ) as f:
        test_questions = json.load(f)

    llm = get_llm()

    results = []
    for item in test_questions:
        question = item["question"]
        expected = item["expected_answer"]
        should_refuse = item["should_refuse"]

        result = answer_question(vectorstore, question)
        actual_answer = result["answer"]
        refused = looks_like_refusal(actual_answer)

        if should_refuse:
            refusal_correct = refused
            correctness = None  # not applicable for refusal questions
        else:
            refusal_correct = not refused
            correctness = (
                judge_correctness(question, expected, actual_answer, llm)
                if not refused else False
            )

        results.append({
            "id": item["id"],
            "question": question,
            "should_refuse": should_refuse,
            "refused": refused,
            "refusal_correct": refusal_correct,
            "correctness": correctness,
            "actual_answer": actual_answer,
        })

        status = "PASS" if refusal_correct and (correctness is not False) else "FAIL"
        print(f"[{status}] Q{item['id']}: {question}")

    # --- Scoring ---
    total = len(results)
    refusal_correct_count = sum(1 for r in results if r["refusal_correct"])
    answerable = [r for r in results if not r["should_refuse"]]
    correct_answers = sum(1 for r in answerable if r["correctness"])

    refusal_accuracy = refusal_correct_count / total * 100
    answer_accuracy = (correct_answers / len(answerable) * 100) if answerable else 0

    report_lines = [
        "# Evaluation Report",
        "",
        f"- Total questions: {total}",
        f"- Refusal/answer-decision accuracy: {refusal_accuracy:.1f}%",
        f"- Answer correctness (on answerable questions): {answer_accuracy:.1f}% "
        f"({correct_answers}/{len(answerable)})",
        "",
        "## Per-question results",
        "",
        "| ID | Question | Should Refuse | Refused | Correct Decision | Answer Correct |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        report_lines.append(
            f"| {r['id']} | {r['question']} | {r['should_refuse']} | "
            f"{r['refused']} | {r['refusal_correct']} | "
            f"{r['correctness'] if r['correctness'] is not None else 'N/A'} |"
        )

    report = "\n".join(report_lines)
    report_path = os.path.join(os.path.dirname(__file__), "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 50)
    print(f"Refusal/answer-decision accuracy: {refusal_accuracy:.1f}%")
    print(f"Answer correctness: {answer_accuracy:.1f}% ({correct_answers}/{len(answerable)})")
    print(f"Full report written to: {report_path}")


if __name__ == "__main__":
    run_evaluation()