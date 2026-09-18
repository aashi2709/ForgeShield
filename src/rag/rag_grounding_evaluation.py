from pathlib import Path
import json
import re

from safety_assistant import ForgeShieldSafetyAssistant


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "rag"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "grounding_test_results.json"
)


TEST_CASES = [
    {
        "id": "overheating",
        "question": (
            "What should an operator do "
            "when a machine shows overheating?"
        ),
        "event_type": None,
        "expected_supported": True,
    },
    {
        "id": "gas_leakage",
        "question": (
            "What actions should be taken "
            "when a gas leakage is detected?"
        ),
        "event_type": "gas_leakage",
        "expected_supported": True,
    },
    {
        "id": "vibration",
        "question": (
            "How should abnormal machine "
            "vibration be handled?"
        ),
        "event_type": "vibration_anomaly",
        "expected_supported": True,
    },
    {
        "id": "unsupported_chemical_spill",
        "question": (
            "What procedure does ForgeShield "
            "have for a chemical spill?"
        ),
        "event_type": None,
        "expected_supported": False,
    },
]


def extract_ids(text: str) -> set[str]:
    """
    Extract ForgeShield incident and procedure IDs
    mentioned by the generated answer.
    """

    incident_ids = set(
        re.findall(
            r"\bINC-\d{4}\b",
            text,
        )
    )

    procedure_ids = set(
        re.findall(
            r"\bFSP-SOP-\d{3}\b",
            text,
        )
    )

    return incident_ids | procedure_ids


def evaluate_result(
    result: dict,
    expected_supported: bool,
) -> dict:

    answer = result["answer"]

    evidence = result["evidence"]

    answer_upper = answer.upper()

    # ---------------------------------------------------------
    # Required response sections
    # ---------------------------------------------------------

    required_sections = [
        "FACTS",
        "POSSIBLE HYPOTHESES",
        "RECOMMENDED ACTIONS",
        "EVIDENCE SOURCES",
        "LIMITATIONS",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in answer_upper
    ]

    sections_complete = (
        len(missing_sections) == 0
    )

    # ---------------------------------------------------------
    # Retrieved source IDs
    # ---------------------------------------------------------

    retrieved_ids = set()

    for item in evidence:

        if item.get("incident_id"):
            retrieved_ids.add(
                item["incident_id"]
            )

        if item.get("procedure_id"):
            retrieved_ids.add(
                item["procedure_id"]
            )

    # ---------------------------------------------------------
    # IDs cited by the LLM
    # ---------------------------------------------------------

    cited_ids = extract_ids(answer)

    unsupported_ids = (
        cited_ids - retrieved_ids
    )

    citation_grounded = (
        len(unsupported_ids) == 0
    )

    # ---------------------------------------------------------
    # Insufficient-evidence behavior
    # ---------------------------------------------------------

    insufficient_phrase = (
        "The retrieved evidence is insufficient "
        "to answer this safely."
    )

    explicitly_insufficient = (
        insufficient_phrase.lower()
        in answer.lower()
    )

    if expected_supported:

        insufficient_behavior_correct = not (
            explicitly_insufficient
        )

    else:

        insufficient_behavior_correct = (
            explicitly_insufficient
            or (
                "does not include"
                in answer.lower()
            )
            or (
                "no "
                in answer.lower()
                and "evidence"
                in answer.lower()
            )
        )

    # ---------------------------------------------------------
    # Synthetic evidence disclosure
    # ---------------------------------------------------------

    synthetic_disclosed = (
        "synthetic"
        in answer.lower()
    )

    # ---------------------------------------------------------
    # Overall checks
    # ---------------------------------------------------------

    checks = {
        "sections_complete":
            sections_complete,

        "citation_grounded":
            citation_grounded,

        "insufficient_behavior_correct":
            insufficient_behavior_correct,

        "synthetic_disclosed":
            synthetic_disclosed,
    }

    passed_checks = sum(
        checks.values()
    )

    total_checks = len(checks)

    return {
        "checks": checks,

        "passed_checks":
            passed_checks,

        "total_checks":
            total_checks,

        "grounding_score":
            round(
                passed_checks
                / total_checks,
                3,
            ),

        "retrieved_ids":
            sorted(retrieved_ids),

        "cited_ids":
            sorted(cited_ids),

        "unsupported_ids":
            sorted(unsupported_ids),

        "missing_sections":
            missing_sections,

        "explicitly_insufficient":
            explicitly_insufficient,
    }


def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "FORGESHIELD | RAG GROUNDING EVALUATION"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assistant = (
        ForgeShieldSafetyAssistant()
    )

    evaluation_results = []

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print(
            f"\n[{index}/{len(TEST_CASES)}] "
            f"{test_case['id']}"
        )

        print(
            f"Question: "
            f"{test_case['question']}"
        )

        result = assistant.ask(
            question=test_case["question"],
            top_k=5,
            event_type=test_case["event_type"],
        )

        evaluation = evaluate_result(
            result=result,
            expected_supported=(
                test_case["expected_supported"]
            ),
        )

        record = {
            "test_id":
                test_case["id"],

            "question":
                test_case["question"],

            "event_type":
                test_case["event_type"],

            "expected_supported":
                test_case["expected_supported"],

            "answer":
                result["answer"],

            "evaluation":
                evaluation,

            "retrieved_evidence": [
                {
                    "rank":
                        item["rank"],

                    "document_type":
                        item["document_type"],

                    "evidence_type":
                        item["evidence_type"],

                    "incident_id":
                        item["incident_id"],

                    "procedure_id":
                        item["procedure_id"],

                    "event_type":
                        item["event_type"],

                    "severity":
                        item["severity"],

                    "distance":
                        item["distance"],

                    "source":
                        item["source"],
                }
                for item in result["evidence"]
            ],
        }

        evaluation_results.append(
            record
        )

        print(
            f"  Grounding score: "
            f"{evaluation['grounding_score']:.3f}"
        )

        print(
            f"  Unsupported IDs: "
            f"{evaluation['unsupported_ids']}"
        )

        print(
            f"  Sections complete: "
            f"{evaluation['checks']['sections_complete']}"
        )

        print(
            f"  Citation grounded: "
            f"{evaluation['checks']['citation_grounded']}"
        )

        print(
            f"  Insufficient-evidence behavior: "
            f"{evaluation['checks']['insufficient_behavior_correct']}"
        )

    # ---------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------

    scores = [
        item["evaluation"]["grounding_score"]
        for item in evaluation_results
    ]

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    all_checks = {
        "sections_complete": [],
        "citation_grounded": [],
        "insufficient_behavior_correct": [],
        "synthetic_disclosed": [],
    }

    for item in evaluation_results:

        checks = item[
            "evaluation"
        ]["checks"]

        for key in all_checks:

            all_checks[key].append(
                checks[key]
            )

    check_rates = {}

    for key, values in all_checks.items():

        check_rates[key] = round(
            sum(values) / len(values),
            3,
        )

    summary = {
        "test_cases":
            len(evaluation_results),

        "average_grounding_score":
            round(
                average_score,
                3,
            ),

        "check_pass_rates":
            check_rates,
    }

    output = {
        "project":
            "ForgeShield",

        "evaluation_type":
            "RAG grounding evaluation",

        "summary":
            summary,

        "results":
            evaluation_results,
    }

    RESULTS_PATH.write_text(
        json.dumps(
            output,
            indent=4,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "GROUNDING EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAverage grounding score: "
        f"{average_score:.3f}"
    )

    print(
        "\nCheck pass rates:"
    )

    for key, value in check_rates.items():

        print(
            f"  {key}: "
            f"{value:.3f}"
        )

    print(
        f"\nResults saved to:"
    )

    print(
        f"  {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
