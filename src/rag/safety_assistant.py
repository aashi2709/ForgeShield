from pathlib import Path
from typing import Any

import ollama

from retriever import ForgeShieldRetriever


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

OLLAMA_MODEL = "qwen3:8b"


class ForgeShieldSafetyAssistant:
    """
    Evidence-grounded safety assistant for ForgeShield.

    Pipeline:

        User Question
              ↓
        Evidence Retrieval
              ↓
        Incident / Procedure Separation
              ↓
        Structured Grounding Prompt
              ↓
        Local Qwen LLM
              ↓
        Evidence-grounded Response

    The assistant deliberately separates:

    - incident evidence: observations from synthetic records
    - procedure evidence: recommended actions from procedures
    """

    def __init__(
        self,
        retriever: ForgeShieldRetriever | None = None,
        model: str = OLLAMA_MODEL,
    ):

        self.retriever = (
            retriever
            if retriever is not None
            else ForgeShieldRetriever()
        )

        self.model = model

        print(
            f"Ollama model: {self.model}"
        )

    def retrieve_evidence(
        self,
        question: str,
        top_k: int = 5,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve a balanced evidence set.

        Procedure and incident evidence are retrieved separately
        so that one evidence type cannot completely dominate the
        context supplied to the LLM.
        """

        # -----------------------------------------------------
        # Retrieve procedures
        # -----------------------------------------------------

        procedure_results = (
            self.retriever.retrieve(
                query=question,
                top_k=max(2, top_k // 2),
                document_type="safety_procedure",
            )
        )

        # -----------------------------------------------------
        # Retrieve incidents
        # -----------------------------------------------------

        incident_results = (
            self.retriever.retrieve(
                query=question,
                top_k=max(2, top_k // 2),
                event_type=event_type,
                document_type="synthetic_incident",
            )
        )

        # -----------------------------------------------------
        # Combine evidence
        # -----------------------------------------------------

        combined = []

        for item in procedure_results:

            item = item.copy()

            item[
                "retrieval_role"
            ] = "procedure"

            combined.append(item)

        for item in incident_results:

            item = item.copy()

            item[
                "retrieval_role"
            ] = "incident"

            combined.append(item)

        # Re-rank only for display.

        combined.sort(
            key=lambda item: (
                item["distance"]
                if item["distance"] is not None
                else float("inf")
            )
        )

        # Assign final ranks.

        for index, item in enumerate(
            combined[:top_k]
        ):

            item["rank"] = index + 1

        return combined[:top_k]

    def build_prompt(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Construct a strict evidence-grounded prompt.

        The prompt explicitly separates procedural guidance from
        historical incident observations.
        """

        if not evidence:

            evidence_text = (
                "No relevant evidence was retrieved."
            )

        else:

            sections = []

            for index, item in enumerate(
                evidence,
                start=1,
            ):

                metadata = item.get(
                    "metadata",
                    {},
                )

                source = metadata.get(
                    "source",
                    "Unknown source",
                )

                document_type = metadata.get(
                    "document_type",
                    "Unknown document type",
                )

                evidence_type = metadata.get(
                    "evidence_type",
                    "Unknown evidence type",
                )

                incident_id = metadata.get(
                    "incident_id",
                    "Not applicable",
                )

                procedure_id = metadata.get(
                    "procedure_id",
                    "Not applicable",
                )

                event_type = metadata.get(
                    "event_type",
                    "Not specified",
                )

                severity = metadata.get(
                    "severity",
                    "Not specified",
                )

                synthetic = metadata.get(
                    "is_synthetic",
                    True,
                )

                content = item.get(
                    "text",
                    "",
                )

                sections.append(
                    f"""
[Evidence {index}]

Retrieval Role:
{item.get("retrieval_role", "unknown")}

Document Type:
{document_type}

Evidence Type:
{evidence_type}

Source:
{source}

Incident ID:
{incident_id}

Procedure ID:
{procedure_id}

Event Type:
{event_type}

Severity:
{severity}

Synthetic:
{synthetic}

Content:
{content}
""".strip()
                )

            evidence_text = (
                "\n\n".join(sections)
            )

        prompt = f"""
You are the ForgeShield Industrial Safety Assistant.

Answer the user's question using ONLY the retrieved
evidence below.

You have NO other knowledge.

============================================================
GROUNDING RULES
============================================================

1. Do not invent facts, procedures, sensor values,
   causes, recommendations, or sources.

2. Do not use outside knowledge, even if you believe
   it would be useful.

3. Every factual claim must be directly supported by
   the retrieved evidence.

4. Do not create a new fact by combining unrelated
   observations from different incident records.

5. Never attribute a sensor value, machine condition,
   or event to an incident unless that exact evidence
   appears in that incident's retrieved content.

6. NEVER invent or modify numerical values.

7. Only mention numerical values when they explicitly
   appear in the retrieved evidence.

8. Procedure evidence describes recommended actions.
   Incident evidence describes recorded observations.

9. Do not treat an incident record as an official
   safety procedure.

10. Do not treat a procedure as evidence that an incident
    actually occurred.

11. Possible contributing factors and root causes must
    remain explicitly unconfirmed unless the evidence
    itself establishes otherwise.

12. Recommendations must come from retrieved procedure
    evidence or explicitly stated corrective/preventive
    actions in retrieved incident evidence.

13. Do not add examples, explanations, technical details,
    or safety practices that are not present in the evidence.

14. Do not invent section names, document names,
    incident IDs, or procedure IDs.

15. Cite the actual Incident ID or Procedure ID when
    referring to evidence.

16. All retrieved records in this proof of concept are
    synthetic unless the evidence explicitly states otherwise.

17. If the evidence does not support an answer, say exactly:

    "The retrieved evidence is insufficient to answer this safely."

18. When evidence is insufficient, do not fill the gap
    using general knowledge.

19. Keep incident-specific facts tied to their source.

20. Prefer omission over unsupported detail.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED EVIDENCE
============================================================

{evidence_text}

============================================================
RESPONSE FORMAT
============================================================

Return EXACTLY these five sections:

FACTS

- State only evidence-supported observations.
- Tie incident-specific observations to their Incident ID.
- Do not merge observations from separate incidents.

POSSIBLE HYPOTHESES

- State possible contributing factors or root causes
  only when explicitly present in the evidence.
- Label all of them as unconfirmed.

RECOMMENDED ACTIONS

- State only actions supported by the retrieved evidence.
- Prefer actions from safety procedure evidence.
- Do not add actions from outside knowledge.

EVIDENCE SOURCES

- List the Incident IDs and/or Procedure IDs actually used.
- Include the source path when available.

LIMITATIONS

- State that the evidence is synthetic.
- State important missing information or uncertainty.
- Do not claim that the answer is universally applicable.

Do not add any other sections.
"""

        return prompt.strip()

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate an answer using the local Ollama model.
        """

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evidence-grounded "
                        "industrial safety assistant. "
                        "Use only the evidence supplied by "
                        "the user prompt. "
                        "Never invent missing information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.0,
            },
        )

        message = response.get(
            "message",
            {},
        )

        answer = message.get(
            "content",
            "",
        )

        if not answer.strip():

            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return answer.strip()

    def ask(
        self,
        question: str,
        top_k: int = 5,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Complete RAG question-answering pipeline.
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

        answer = self.generate(
            prompt=prompt,
        )

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "prompt": prompt,
        }


def print_answer(
    result: dict[str, Any],
):

    print(
        "\n" + "=" * 70
    )

    print(
        "FORGESHIELD | LOCAL RAG SAFETY ASSISTANT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nMODEL:\n{OLLAMA_MODEL}"
    )

    print(
        f"\nQUESTION:\n{result['question']}"
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "LLM RESPONSE"
    )

    print(
        "-" * 70
    )

    print(
        result["answer"]
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "RETRIEVED EVIDENCE"
    )

    print(
        "-" * 70
    )

    for item in result["evidence"]:

        print(
            f"\n[{item['rank']}] "
            f"{item['retrieval_role'].upper()}"
        )

        print(
            f"Document Type: "
            f"{item['document_type']}"
        )

        print(
            f"Incident: "
            f"{item['incident_id'] or '-'}"
        )

        print(
            f"Procedure: "
            f"{item['procedure_id'] or '-'}"
        )

        print(
            f"Event: "
            f"{item['event_type'] or '-'}"
        )

        print(
            f"Severity: "
            f"{item['severity'] or '-'}"
        )

        print(
            f"Distance: "
            f"{item['distance']:.4f}"
        )

        print(
            f"Source: "
            f"{item['source']}"
        )

    print(
        "\n" + "=" * 70
    )


def main():

    assistant = (
        ForgeShieldSafetyAssistant()
    )

    test_cases = [

        {
            "question": (
                "What should an operator do "
                "when a machine shows overheating?"
            ),
            "event_type": None,
        },

        {
            "question": (
                "What actions should be taken "
                "when a gas leakage is detected?"
            ),
            "event_type": "gas_leakage",
        },

        {
            "question": (
                "How should abnormal machine "
                "vibration be handled?"
            ),
            "event_type": "vibration_anomaly",
        },

        {
            "question": (
                "What procedure does ForgeShield "
                "have for a chemical spill?"
            ),
            "event_type": None,
        },
    ]

    for test_case in test_cases:

        result = assistant.ask(
            question=test_case["question"],
            top_k=5,
            event_type=test_case["event_type"],
        )

        print_answer(
            result
        )


if __name__ == "__main__":
    main()