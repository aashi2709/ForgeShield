from pathlib import Path

import ollama
import streamlit as st

from dashboard.components.theme import (
    BORDER,
    BORDER_STRONG,
    SURFACE,
    TEXT,
    MUTED,
    BRAND,
    LOW,
)
from dashboard.components.ui import render_html
from rag.retriever import ForgeShieldRetriever


OLLAMA_MODEL = "qwen3:8b"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _initialise_state() -> None:
    if "safety_copilot_messages" not in st.session_state:
        st.session_state.safety_copilot_messages = []

    if "safety_copilot_retriever" not in st.session_state:
        st.session_state.safety_copilot_retriever = None


def _get_retriever():
    _initialise_state()

    if st.session_state.safety_copilot_retriever is None:
        st.session_state.safety_copilot_retriever = ForgeShieldRetriever()

    return st.session_state.safety_copilot_retriever


def _render_kpi(label: str, value: str, description: str) -> None:
    render_html(
        f"""
        <div style="
            min-height:145px;
            padding:4px 0 0 0;
        ">
            <div style="
                color:{MUTED};
                font-size:12px;
                font-weight:700;
                letter-spacing:1.5px;
                text-transform:uppercase;
            ">
                {label}
            </div>

            <div style="
                margin-top:14px;
                color:{TEXT};
                font-size:30px;
                font-weight:700;
                line-height:1.1;
            ">
                {value}
            </div>

            <div style="
                margin-top:10px;
                color:{MUTED};
                font-size:13px;
                line-height:1.5;
            ">
                {description}
            </div>
        </div>
        """,
    )


def _render_pipeline_step(number: str, title: str, description: str) -> None:
    render_html(
        f"""
        <div style="
            height:235px;
            padding:20px 18px;
            border:1px solid {BORDER};
            border-radius:8px;
            background:{SURFACE};
            box-sizing:border-box;
        ">
            <div style="
                color:{MUTED};
                font-size:11px;
                font-weight:700;
                letter-spacing:1.5px;
            ">
                {number}
            </div>

            <div style="
                margin-top:18px;
                color:{TEXT};
                font-size:16px;
                font-weight:700;
            ">
                {title}
            </div>

            <div style="
                margin-top:10px;
                color:{MUTED};
                font-size:12px;
                line-height:1.6;
            ">
                {description}
            </div>
        </div>
        """,
    )


def _render_response_block(
    title: str,
    description: str,
) -> None:
    render_html(
        f"""
        <div style="
            min-height:170px;
            padding:20px;
            border:1px solid {BORDER};
            border-radius:8px;
            background:{SURFACE};
        ">
            <div style="
                color:{TEXT};
                font-size:13px;
                font-weight:700;
                letter-spacing:.5px;
                text-transform:uppercase;
            ">
                {title}
            </div>

            <div style="
                margin-top:12px;
                color:{MUTED};
                font-size:13px;
                line-height:1.6;
            ">
                {description}
            </div>
        </div>
        """,
    )


def _render_footer() -> None:
    render_html(
        f"""
        <div class="fs-footer">
            <span>
                FORGESHIELD · SAFETY COPILOT
            </span>

            <span class="fs-footer-mono">
                Research Proof of Concept · Retrieval-Grounded AI
            </span>
        </div>
        """,
    )


def _render_evidence(evidence) -> None:
    if not evidence:
        st.info("No retrieved evidence was returned for this query.")
        return

    rows = []

    for item in evidence:
        if isinstance(item, dict):
            metadata = item.get("metadata", {}) or {}
            content = item.get("content", item.get("text", ""))
        else:
            metadata = getattr(item, "metadata", {}) or {}
            content = getattr(item, "page_content", "")

        incident_id = metadata.get("incident_id")
        procedure_id = metadata.get("procedure_id")

        if incident_id:
            evidence_id = incident_id
        elif procedure_id:
            evidence_id = procedure_id
        else:
            evidence_id = "-"

        rows.append(
            {
                "Evidence ID": evidence_id,
                "Evidence Type": metadata.get(
                    "evidence_type",
                    metadata.get("document_type", "Unknown"),
                ),
                "Event Type": metadata.get(
                    "event_type",
                    "Unknown",
                ),
                "Content": content,
            }
        )

    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

def _extract_response(response):
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")

    message = getattr(response, "message", None)
    if message is not None:
        return getattr(message, "content", "")

    return ""


def _run_llm(question: str, evidence) -> str:
    evidence_text = []

    for index, item in enumerate(evidence, start=1):
        if isinstance(item, dict):
            metadata = item.get("metadata", {}) or {}
            content = item.get("content", item.get("text", ""))
        else:
            metadata = getattr(item, "metadata", {}) or {}
            content = getattr(item, "page_content", "")

        incident_id = metadata.get("incident_id")
        procedure_id = metadata.get("procedure_id")

        if incident_id:
            evidence_id = incident_id
        elif procedure_id:
            evidence_id = procedure_id
        else:
            evidence_id = f"EVIDENCE-{index}"

        evidence_text.append(
            f"""
[EVIDENCE {evidence_id}]
Event type: {metadata.get("event_type", "unknown")}
Evidence type: {metadata.get("evidence_type", "unknown")}
Severity: {metadata.get("severity", "unknown")}
Content:
{content}
"""
        )

    context = "\n".join(evidence_text)

    prompt = f"""
You are the ForgeShield Safety Copilot.

You must answer ONLY from the retrieved research evidence supplied below.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Follow these rules strictly:

1. Separate FACTS from POSSIBLE HYPOTHESES.
2. Facts must be directly supported by retrieved evidence.
3. Hypotheses must be clearly labelled as hypotheses.
4. Recommended actions may only come from retrieved safety procedures or evidence.
5. Do not invent operational safety instructions.
6. If the evidence does not directly address the user's question, explicitly state that the evidence is insufficient.
7. If the evidence is unrelated or only partially related to the user's question, do NOT infer connections between them.
8. Do not use general world knowledge, background knowledge, common sense, or unstated assumptions to fill evidence gaps.
9. When evidence is insufficient or unrelated, POSSIBLE HYPOTHESES must be "None supported by the retrieved evidence."
10. When evidence is insufficient or unrelated, RECOMMENDED ACTIONS must be "None supported by the retrieved evidence."
11. Do not introduce examples, scenarios, causes, materials, procedures, or relationships that are not explicitly present in the retrieved evidence.
12. Do not treat synthetic research evidence as validated industrial operating procedure.
13. Cite evidence identifiers when making evidence-supported claims.
14. Evidence sources may be listed for transparency even when they are not relevant, but clearly state that they do not support the user's question.
15. Never convert an unsupported possibility into a hypothesis merely because it sounds plausible.

Return exactly these sections:

FACTS
POSSIBLE HYPOTHESES
RECOMMENDED ACTIONS
EVIDENCE SOURCES
LIMITATIONS
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return _extract_response(response)

def _render_static_response_structure() -> None:
    st.markdown("### Response Structure")

    st.write(
        "The Safety Copilot separates evidence-backed information "
        "from uncertain model-generated interpretation."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**FACTS**")
        _render_response_block(
            "Grounded facts",
            "Information directly supported by retrieved evidence.",
        )

    with col2:
        st.markdown("**POSSIBLE HYPOTHESES**")
        _render_response_block(
            "Hypothesis separation",
            "Potential interpretations explicitly separated from established facts.",
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**RECOMMENDED ACTIONS**")
        _render_response_block(
            "Action grounding",
            "Actions should originate from retrieved procedures or evidence.",
        )

    with col2:
        st.markdown("**LIMITATIONS**")
        _render_response_block(
            "Abstention",
            "Missing evidence or unsupported requests should result in an explicit limitation.",
        )


def render_safety_copilot() -> None:
    _initialise_state()

    render_html(
        """
        <div class="fs-eyebrow">
            SAFETY COPILOT
        </div>
        """,
    )

    render_html(
        """
        <h1 style="
            margin-top:0;
            margin-bottom:8px;
        ">
            Retrieval-Grounded Safety Assistant
        </h1>
        """,
    )

    render_html(
        """
        <div style="
            color:#AAB2BF;
            font-size:14px;
            line-height:1.6;
            margin-bottom:42px;
        ">
            Evidence-grounded safety assistance using the ForgeShield
            incident and procedure knowledge base.
        </div>
        """,
    )

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        _render_kpi(
            "Knowledge Source",
            "RAG",
            "Retrieval-Augmented Generation",
        )

    with kpi2:
        _render_kpi(
            "Evidence Policy",
            "Grounded",
            "Responses constrained by retrieved evidence",
        )

    with kpi3:
        _render_kpi(
            "LLM",
            "Qwen3:8B",
            "Local Ollama inference",
        )

    st.write("")

    st.markdown("### Safety Query")

    st.write(
        "Ask a safety-related question using the ForgeShield research knowledge base."
    )

    question = st.text_area(
        "Enter a safety question",
        placeholder=(
            "Example: What should be checked when a machine "
            "shows abnormal vibration?"
        ),
        height=100,
        key="safety_copilot_question",
    )

    run_query = st.button(
        "Run Safety Copilot",
        type="secondary",
    )

    if run_query:
        if not question.strip():
            st.warning("Enter a safety question first.")
        else:
            with st.spinner("Retrieving evidence and generating response..."):
                try:
                    retriever = _get_retriever()

                    evidence = retriever.retrieve(
                        question.strip()
                    )

                    answer = _run_llm(
                        question.strip(),
                        evidence,
                    )

                    st.session_state.safety_copilot_messages.append(
                        {
                            "role": "user",
                            "content": question.strip(),
                        }
                    )

                    st.session_state.safety_copilot_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "evidence": evidence,
                        }
                    )

                except Exception as exc:
                    st.error(
                        f"Safety Copilot execution failed: {exc}"
                    )

    st.write("")

    if (
        st.session_state.safety_copilot_messages
        and st.session_state.safety_copilot_messages[-1]["role"]
        == "assistant"
    ):
        latest = st.session_state.safety_copilot_messages[-1]

        st.markdown("### Grounded Response")

        st.markdown(latest["content"])

        evidence = latest.get("evidence", [])

        if evidence:
            st.markdown("### Retrieved Evidence")

            with st.expander(
                f"View retrieved evidence ({len(evidence)} chunks)"
            ):
                _render_evidence(evidence)

    st.markdown("### Grounding Pipeline")

    steps = [
        (
            "01",
            "Question",
            "Safety query submitted by the user.",
        ),
        (
            "02",
            "Retrieve",
            "Relevant evidence retrieved from the research knowledge base.",
        ),
        (
            "03",
            "Ground",
            "Retrieved evidence is assembled into the model context.",
        ),
        (
            "04",
            "Generate",
            "Qwen3:8B generates a response constrained by the evidence.",
        ),
        (
            "05",
            "Validate",
            "Evidence identifiers and limitations remain visible to the user.",
        ),
    ]

    cols = st.columns(5)

    for column, step in zip(cols, steps):
        with column:
            _render_pipeline_step(*step)

    st.write("")

    _render_static_response_structure()

    st.write("")

    st.markdown("### Grounded Response Artifact")

    st.write(
        "The interactive RAG execution layer is kept separate from the "
        "dashboard view. The POC page exposes the grounding interface "
        "without inventing an answer when live response evidence is unavailable."
    )

    st.markdown("### Evidence Policy")

    policy_items = [
        (
            "Grounded facts",
            "Statements presented as facts should be supported by retrieved knowledge-base evidence.",
        ),
        (
            "Hypothesis separation",
            "Potential explanations are explicitly labelled as hypotheses rather than established facts.",
        ),
        (
            "Action grounding",
            "Safety actions should originate from retrieved procedures or evidence. The assistant should not manufacture operational instructions when the knowledge base does not support them.",
        ),
        (
            "Abstention",
            "When relevant evidence is unavailable, the system should surface that limitation instead of fabricating an answer.",
        ),
    ]

    for title, description in policy_items:
        render_html(
            f"""
            <div style="
                margin:42px 24px;
                max-width:1000px;
            ">
                <div style="
                    color:{TEXT};
                    font-size:15px;
                    font-weight:700;
                ">
                    {title}
                </div>

                <div style="
                    margin-top:8px;
                    color:{MUTED};
                    font-size:14px;
                    line-height:1.7;
                ">
                    {description}
                </div>
            </div>
            """,
        )

    st.markdown("### Research Contribution")

    render_html(
        f"""
        <div style="
            margin:28px 24px;
            max-width:1100px;
        ">
            <div style="
                color:{TEXT};
                font-size:17px;
                line-height:1.8;
            ">
                ForgeShield extends predictive risk detection with a
                retrieval-grounded generative layer. Instead of treating
                the language model as an independent source of safety
                knowledge, the architecture supplies retrieved incident
                and procedure evidence as context.
            </div>

            <div style="
                margin-top:42px;
                color:{MUTED};
                font-size:14px;
            ">
                This creates a research pathway from:
            </div>

            <div style="
                margin-top:16px;
                color:{TEXT};
                font-size:16px;
                font-weight:700;
                line-height:1.8;
            ">
                Sensor Risk → Anomaly Evidence → Retrieval →
                Grounded Explanation → Safety Assistance
            </div>

            <div style="
                margin-top:42px;
                color:{MUTED};
                font-size:14px;
                line-height:1.7;
            ">
                The current implementation is a proof of concept.
                The knowledge base contains synthetic research evidence
                and therefore should not be treated as a validated
                industrial operating procedure.
            </div>
        </div>
        """,
    )

    _render_footer()


if __name__ == "__main__":
    render_safety_copilot()
