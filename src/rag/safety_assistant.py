from pathlib import Path
from typing import Any

import ollama

from retriever import ForgeShieldRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OLLAMA_MODEL = "qwen3:8b"


class ForgeShieldSafetyAssistant:
    """
    Evidence-grounded safety assistant for ForgeShield.

    Pipeline:

        User Question
              ↓
        Semantic Retrieval
              ↓
        ChromaDB Evidence
              ↓
        Grounding Prompt
              ↓
        Local Qwen LLM
              ↓
        Evidence-grounded Response
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
        Retrieve evidence from the ForgeShield knowledge base.
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
        Construct a strict evidence-grounded prompt.
        """

        evidence_text = (
            self.retriever.format_evidence(
                evidence
            )
        )

        prompt = f"""
You are the ForgeShield Industrial Safety Assistant.

You must answer the user's question using ONLY the
retrieved evidence provided below.

GROUNDING RULES:

1. Do not invent facts, procedures, sources, or incident details.
2. Do not use outside knowledge.
3. Treat retrieved evidence as the only knowledge available.
4. Separate observed facts from possible hypotheses.
5. Never present a possible root cause as a confirmed cause.
6. Recommendations must come from the retrieved evidence.
7. Preserve the distinction between synthetic records and real incidents.
8. If the retrieved evidence is insufficient, explicitly state:
   "The retrieved evidence is insufficient to answer this safely."
9. Cite the Incident ID when referring to evidence.
10. Do not fabricate citations or source names.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence_text}

Return the answer using exactly these sections:

FACTS
- Evidence-supported observations.

POSSIBLE HYPOTHESES
- Possible contributing factors or root causes.
- Explicitly label them as unconfirmed hypotheses.

RECOMMENDED ACTIONS
- Actions supported by the retrieved evidence.

EVIDENCE SOURCES
- Incident ID and source for the evidence used.

LIMITATIONS
- Mention that the evidence may be synthetic.
- Mention important missing information or uncertainty.

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
                        "You are a careful industrial safety "
                        "assistant. Follow the supplied evidence "
                        "strictly and never invent information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.1,
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
    ]

    for test_case in test_cases:

        result = assistant.ask(
            question=test_case["question"],
            top_k=3,
            event_type=test_case["event_type"],
        )

        print_answer(
            result
        )


if __name__ == "__main__":
    main()