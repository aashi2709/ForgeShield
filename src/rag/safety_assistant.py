from pathlib import Path
from typing import Any

from retriever import ForgeShieldRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ForgeShieldSafetyAssistant:
    """
    Evidence-grounded safety assistant for ForgeShield.

    The assistant retrieves relevant safety evidence first.
    LLM generation is intentionally kept separate from retrieval
    so that evidence can be inspected and evaluated independently.
    """

    def __init__(
        self,
        retriever: ForgeShieldRetriever | None = None,
    ):
        self.retriever = (
            retriever
            if retriever is not None
            else ForgeShieldRetriever()
        )

    def retrieve_evidence(
        self,
        question: str,
        top_k: int = 5,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve evidence relevant to a safety question.
        """

        return self.retriever.retrieve(
            query=question,
            top_k=top_k,
            event_type=event_type,
        )

    def build_prompt(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Build a strict evidence-grounded prompt.

        The prompt explicitly prevents unsupported claims
        and requires facts and hypotheses to be separated.
        """

        evidence_text = (
            self.retriever.format_evidence(
                evidence
            )
        )

        prompt = f"""
You are the ForgeShield Industrial Safety Assistant.

Your task is to answer the user's safety question using ONLY
the retrieved evidence provided below.

IMPORTANT RULES:

1. Do not invent facts, procedures, sources, or incident details.
2. Do not use outside knowledge when the retrieved evidence is
   insufficient.
3. Clearly distinguish observed facts from possible hypotheses.
4. Recommendations must be grounded in the retrieved evidence.
5. Every factual statement should be traceable to the evidence.
6. If the evidence is insufficient, explicitly say:
   "The retrieved evidence is insufficient to answer this safely."
7. Remember that the incident records may be synthetic research
   data and must not be presented as real-world incidents.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence_text}

Respond using exactly this structure:

FACTS
- List the relevant evidence-supported facts.

POSSIBLE HYPOTHESES
- List possible contributing factors or root causes.
- Clearly state that these are hypotheses, not confirmed causes.

RECOMMENDED ACTIONS
- List actions explicitly supported by the retrieved evidence.

EVIDENCE SOURCES
- Give the Incident ID and source path for the evidence used.

LIMITATIONS
- State any important limitations or missing information.
"""

        return prompt.strip()

    def answer_without_llm(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Deterministic demo mode.

        This allows the complete retrieval + evidence pipeline
        to be demonstrated without requiring an external LLM API.
        """

        if not evidence:
            return (
                "The retrieved evidence is insufficient "
                "to answer this safely."
            )

        lines = []

        lines.append("FACTS")

        for result in evidence[:3]:
            text = result["text"]

            summary = text.split(
                "## Possible Contributing Factors"
            )[0]

            summary = summary.replace(
                "# Machine Overheating Incident",
                "",
            ).strip()

            if summary:
                lines.append(
                    f"- Evidence retrieved from "
                    f"{result['incident_id']} indicates "
                    f"{result['event_type']} with "
                    f"{result['severity']} severity."
                )

                break

        lines.append("")
        lines.append("POSSIBLE HYPOTHESES")

        hypothesis_added = False

        for result in evidence[:3]:

            text = result["text"]

            if "## Potential Root Causes" in text:

                section = text.split(
                    "## Potential Root Causes",
                    1,
                )[1]

                if "## Corrective Actions" in section:

                    section = section.split(
                        "## Corrective Actions",
                        1,
                    )[0]

                causes = [
                    line.strip()
                    for line in section.splitlines()
                    if line.strip().startswith("-")
                ]

                for cause in causes[:4]:

                    lines.append(
                        f"- {cause[1:].strip()} "
                        "(possible cause, not confirmed)."
                    )

                hypothesis_added = True
                break

        if not hypothesis_added:

            lines.append(
                "- No root-cause hypotheses were "
                "available in the retrieved evidence."
            )

        lines.append("")
        lines.append("RECOMMENDED ACTIONS")

        actions_added = False

        for result in evidence[:3]:

            text = result["text"]

            if "## Corrective Actions" in text:

                section = text.split(
                    "## Corrective Actions",
                    1,
                )[1]

                if "## Preventive Actions" in section:

                    section = section.split(
                        "## Preventive Actions",
                        1,
                    )[0]

                actions = [
                    line.strip()
                    for line in section.splitlines()
                    if line.strip().startswith("-")
                ]

                for action in actions[:5]:

                    lines.append(
                        f"- {action[1:].strip()}"
                    )

                actions_added = True
                break

        if not actions_added:

            lines.append(
                "- No corrective actions were "
                "available in the retrieved evidence."
            )

        lines.append("")
        lines.append("EVIDENCE SOURCES")

        seen_sources = set()

        for result in evidence:

            key = (
                result["incident_id"],
                result["source"],
            )

            if key in seen_sources:
                continue

            seen_sources.add(key)

            lines.append(
                f"- {result['incident_id']} | "
                f"{result['source']}"
            )

        lines.append("")
        lines.append("LIMITATIONS")

        lines.append(
            "- This response uses retrieved synthetic "
            "research records."
        )

        lines.append(
            "- The retrieved evidence does not establish "
            "causality or represent a real industrial incident."
        )

        return "\n".join(lines)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Complete evidence-grounded question-answering pipeline.

        Currently uses deterministic demo generation.
        A real LLM can be connected later without changing
        the retrieval interface.
        """

        evidence = self.retrieve_evidence(
            question=question,
            top_k=top_k,
            event_type=event_type,
        )

        prompt = self.build_prompt(
            question=question,
            evidence=evidence,
        )

        answer = self.answer_without_llm(
            question=question,
            evidence=evidence,
        )

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "prompt": prompt,
        }


def print_answer(result: dict[str, Any]):

    print("\n" + "=" * 70)
    print("FORGESHIELD | SAFETY ASSISTANT")
    print("=" * 70)

    print(
        f"\nQUESTION:\n{result['question']}"
    )

    print("\n" + "-" * 70)
    print("GENERATED RESPONSE")
    print("-" * 70)

    print(
        result["answer"]
    )

    print("\n" + "-" * 70)
    print("RETRIEVED EVIDENCE")
    print("-" * 70)

    for item in result["evidence"]:

        print(
            f"\n[{item['rank']}] "
            f"{item['incident_id']}"
        )

        print(
            f"Event: {item['event_type']}"
        )

        print(
            f"Severity: {item['severity']}"
        )

        print(
            f"Distance: {item['distance']:.4f}"
        )

        print(
            f"Source: {item['source']}"
        )

    print("\n" + "=" * 70)
    print("SAFETY ASSISTANT TEST COMPLETE")
    print("=" * 70)


def main():

    assistant = ForgeShieldSafetyAssistant()

    questions = [
        (
            "What should an operator do "
            "when a machine shows overheating?",
            None,
        ),
        (
            "What actions should be taken "
            "when a gas leakage is detected?",
            "gas_leakage",
        ),
    ]

    for question, event_type in questions:

        result = assistant.ask(
            question=question,
            top_k=3,
            event_type=event_type,
        )

        print_answer(result)


if __name__ == "__main__":
    main()
