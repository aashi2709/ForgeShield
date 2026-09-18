import json
import re
from pathlib import Path

import ollama

from retriever import ForgeShieldRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OLLAMA_MODEL = "qwen3:8b"

EVENT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integration"
    / "demo_event.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "integration"
)


class ForgeShieldIncidentReportGenerator:

    def __init__(self):

        print(
            "Loading ForgeShield incident report generator..."
        )

        self.retriever = ForgeShieldRetriever()

        print(
            f"Ollama model: {OLLAMA_MODEL}"
        )

    # =========================================================
    # EVENT LOADING
    # =========================================================

    @staticmethod
    def load_event():

        if not EVENT_PATH.exists():

            raise FileNotFoundError(
                "ForgeShield integration event not found:\n"
                f"{EVENT_PATH}\n\n"
                "Run src/forge_shield_pipeline.py first."
            )

        with open(
            EVENT_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            event = json.load(file)

        required_fields = [
            "machine_id",
            "event_type",
            "risk_score",
            "risk_band",
            "failure_probability",
            "anomaly_score",
        ]

        missing = [
            field
            for field in required_fields
            if field not in event
        ]

        if missing:

            raise ValueError(
                "Integration event is missing required fields: "
                f"{missing}"
            )

        return event

    # =========================================================
    # EVIDENCE RETRIEVAL
    # =========================================================

    def retrieve_evidence(
        self,
        event_type=None,
        severity=None,
        top_k=6,
    ):

        """
        Retrieve historical incident evidence and safety
        procedure evidence through independent retrieval paths.
        """

        if event_type:

            readable_event = (
                event_type.replace(
                    "_",
                    " ",
                )
            )

            incident_query = (
                "historical industrial safety incident "
                f"machine {readable_event}"
            )

            procedure_query = (
                "safety procedure SOP "
                f"machine {readable_event} "
                "operator response corrective actions "
                "preventive actions"
            )

        else:

            incident_query = (
                "historical industrial machine safety "
                "incident abnormal operating condition"
            )

            procedure_query = (
                "industrial machine safety procedure SOP "
                "operator response corrective actions "
                "preventive actions"
            )

        # -----------------------------------------------------
        # Historical incidents
        # -----------------------------------------------------

        incident_evidence = (
            self.retriever.retrieve(
                query=incident_query,
                top_k=top_k,
                document_type="synthetic_incident",
                event_type=event_type,
                severity=severity,
            )
        )

        # -----------------------------------------------------
        # Fallback incident retrieval
        # -----------------------------------------------------

        if len(incident_evidence) < 2:

            fallback_incidents = (
                self.retriever.retrieve(
                    query=incident_query,
                    top_k=top_k,
                    document_type="synthetic_incident",
                )
            )

            existing_ids = {
                item.get("incident_id")
                for item in incident_evidence
                if item.get("incident_id")
            }

            for item in fallback_incidents:

                incident_id = item.get(
                    "incident_id"
                )

                if (
                    incident_id
                    and incident_id not in existing_ids
                ):

                    incident_evidence.append(
                        item
                    )

                    existing_ids.add(
                        incident_id
                    )

                if len(incident_evidence) >= top_k:
                    break

        # -----------------------------------------------------
        # Safety procedures
        # -----------------------------------------------------

        procedure_evidence = (
            self.retriever.retrieve(
                query=procedure_query,
                top_k=top_k,
                document_type="safety_procedure",
            )
        )

        # -----------------------------------------------------
        # Event-specific procedure filtering
        # -----------------------------------------------------

        if event_type:

            event_terms = [
                term
                for term in (
                    event_type
                    .replace(
                        "_",
                        " ",
                    )
                    .lower()
                    .split()
                )
                if len(term) > 3
            ]

            relevant_procedures = []

            for item in procedure_evidence:

                source = (
                    item.get("source")
                    or ""
                ).lower()

                content = (
                    item.get("content")
                    or ""
                ).lower()

                searchable_text = (
                    source
                    + " "
                    + content
                )

                if any(
                    term in searchable_text
                    for term in event_terms
                ):

                    relevant_procedures.append(
                        item
                    )

            if relevant_procedures:

                procedure_evidence = (
                    relevant_procedures
                )

        # -----------------------------------------------------
        # Deduplicate incidents by document
        # -----------------------------------------------------

        unique_incidents = []

        seen_incidents = set()

        for item in incident_evidence:

            key = (
                item.get("incident_id"),
                item.get("source"),
            )

            if key not in seen_incidents:

                unique_incidents.append(
                    item
                )

                seen_incidents.add(
                    key
                )

        incident_evidence = (
            unique_incidents
        )

        # -----------------------------------------------------
        # Deduplicate procedures
        # -----------------------------------------------------

        unique_procedures = []

        seen_procedures = set()

        for item in procedure_evidence:

            key = (
                item.get("procedure_id"),
                item.get("source"),
            )

            if key not in seen_procedures:

                unique_procedures.append(
                    item
                )

                seen_procedures.add(
                    key
                )

        procedure_evidence = (
            unique_procedures
        )

        # -----------------------------------------------------
        # Sort by retrieval distance
        # -----------------------------------------------------

        incident_evidence.sort(
            key=lambda x: x.get(
                "distance",
                float("inf"),
            )
        )

        procedure_evidence.sort(
            key=lambda x: x.get(
                "distance",
                float("inf"),
            )
        )

        # -----------------------------------------------------
        # Limit evidence
        # -----------------------------------------------------

        incident_evidence = (
            incident_evidence[:top_k]
        )

        procedure_evidence = (
            procedure_evidence[:top_k]
        )

        # -----------------------------------------------------
        # Add retrieval roles
        # -----------------------------------------------------

        evidence = []

        for item in incident_evidence:

            item = dict(item)

            item[
                "retrieval_role"
            ] = "incident"

            evidence.append(
                item
            )

        for item in procedure_evidence:

            item = dict(item)

            item[
                "retrieval_role"
            ] = "procedure"

            evidence.append(
                item
            )

        # -----------------------------------------------------
        # Balanced evidence selection
        # -----------------------------------------------------

        incident_items = [
            item
            for item in evidence
            if item[
                "retrieval_role"
            ] == "incident"
        ]

        procedure_items = [
            item
            for item in evidence
            if item[
                "retrieval_role"
            ] == "procedure"
        ]

        if (
            incident_items
            and procedure_items
        ):

            incident_count = min(
                len(incident_items),
                max(
                    2,
                    top_k // 2,
                ),
            )

            procedure_count = min(
                len(procedure_items),
                top_k - incident_count,
            )

            selected = (
                incident_items[
                    :incident_count
                ]
                +
                procedure_items[
                    :procedure_count
                ]
            )

        else:

            selected = evidence[:top_k]

        # -----------------------------------------------------
        # Final ranking
        # -----------------------------------------------------

        selected.sort(
            key=lambda x: x.get(
                "distance",
                float("inf"),
            )
        )

        for rank, item in enumerate(
            selected,
            start=1,
        ):

            item[
                "final_rank"
            ] = rank

        return selected

    # =========================================================
    # PROMPT
    # =========================================================

    @staticmethod
    def build_prompt(
        event,
        evidence,
    ):

        evidence_blocks = []

        for item in evidence:

            block = f"""
--- EVIDENCE {item["final_rank"]} ---

Retrieval Role:
{item.get("retrieval_role")}

Source:
{item.get("source")}

Document Type:
{item.get("document_type")}

Evidence Type:
{item.get("evidence_type")}

Incident ID:
{item.get("incident_id")}

Procedure ID:
{item.get("procedure_id")}

Event Type:
{item.get("event_type")}

Severity:
{item.get("severity")}

Synthetic:
{item.get("synthetic")}

Content:
{item.get("content", "")}

--- END EVIDENCE ---
"""

            evidence_blocks.append(
                block
            )

        evidence_text = (
            "\n".join(
                evidence_blocks
            )
        )

        event_text = json.dumps(
            event,
            indent=2,
        )

        return f"""
You are the ForgeShield Industrial Safety
Incident Report Generator.

Generate a structured incident report using
ONLY the EVENT DATA and RETRIEVED EVIDENCE.

==================================================
EVENT DATA
==================================================

{event_text}

==================================================
RETRIEVED EVIDENCE
==================================================

{evidence_text}

==================================================
ABSOLUTE GROUNDING RULES
==================================================

1. EVENT DATA is the ONLY source for the current
   incident's machine ID, event type, risk values,
   sensor values, product type, and failure flags.

2. Retrieved evidence may provide historical observations,
   hypotheses, procedures, corrective actions, and
   preventive actions.

3. NEVER invent facts, measurements, timestamps,
   identifiers, procedures, actions, thresholds,
   operating ranges, causes, or sources.

4. NEVER invent an Incident ID.

5. NEVER invent a Procedure ID.

6. ONLY cite an Incident ID that appears explicitly
   in the RETRIEVED EVIDENCE.

7. ONLY cite a Procedure ID that appears explicitly
   in the RETRIEVED EVIDENCE.

8. If an ID is not present in retrieved evidence,
   DO NOT mention it.

9. Do not cite evidence that is unrelated to the
   current event type.

10. Historical incidents are historical evidence only.
    They do NOT prove the current event occurred in
    the same way.

11. Historical incidents marked synthetic MUST remain
    explicitly identified as synthetic.

12. Procedure documents describe recommended procedures.
    They do NOT prove that those procedures were performed.

13. Do not treat a procedure as evidence that an incident
    occurred.

14. Do not treat a historical incident as an official SOP.

15. Every current sensor value must come from EVENT DATA.

16. Every historical sensor value must come from
    RETRIEVED EVIDENCE.

17. Never modify, calculate, reinterpret, or extrapolate
    numerical values from evidence.

18. Do not call a value high, low, excessive, abnormal,
    normal, safe, or unsafe unless retrieved evidence
    explicitly supports that interpretation.

19. Possible contributing factors MUST be hypotheses
    unless directly established by evidence.

20. Potential root causes MUST be hypotheses unless
    directly established by evidence.

21. Corrective actions may ONLY be taken from explicit
    retrieved procedure evidence or explicit corrective
    actions in retrieved incident evidence.

22. Preventive actions may ONLY be taken from explicit
    retrieved procedure evidence or explicit preventive
    actions in retrieved incident evidence.

23. If no supported corrective action exists, write:
    "No evidence-supported corrective actions are
    available in the retrieved evidence."

24. If no supported preventive action exists, write:
    "No evidence-supported preventive actions are
    available in the retrieved evidence."

25. Do not use outside knowledge.

26. Do not create recommendations from general safety knowledge.

27. Do not combine unrelated evidence into a new claim.

28. Do not invent timestamps.

29. If timestamps are absent, state exactly:
    "No timestamps or chronological event sequence
    are provided in the event data."

30. Prefer omission over unsupported claims.

31. Keep FACTS, HYPOTHESES, and ACTIONS separate.

==================================================
REPORT STRUCTURE
==================================================

Use EXACTLY these sections:

INCIDENT SUMMARY

TIMELINE OF EVENTS

ABNORMAL SENSOR BEHAVIOR

POSSIBLE CONTRIBUTING FACTORS

POTENTIAL ROOT CAUSES

EVIDENCE

CORRECTIVE ACTIONS

PREVENTIVE ACTIONS

LIMITATIONS

==================================================
SECTION INSTRUCTIONS
==================================================

INCIDENT SUMMARY
- Use EVENT DATA only.
- Include machine ID, event type, risk band,
  risk score, failure probability, anomaly score,
  and available sensor values.

TIMELINE OF EVENTS
- Use timestamps only if explicitly available.
- Otherwise use the exact insufficiency statement.

ABNORMAL SENSOR BEHAVIOR
- Report provided sensor values.
- Do not independently classify them as abnormal
  without evidence.

POSSIBLE CONTRIBUTING FACTORS
- Use only evidence-supported hypotheses.
- Label every hypothesis clearly.

POTENTIAL ROOT CAUSES
- Use only evidence-supported hypotheses.
- Label every hypothesis clearly.

EVIDENCE
Separate:
1. Current EVENT DATA
2. Historical incident evidence
3. Safety procedure evidence

For historical incidents:
- Cite only retrieved Incident IDs.
- Identify them as synthetic where applicable.

For procedures:
- Cite only retrieved Procedure IDs.
- Do not claim the procedure was performed.

CORRECTIVE ACTIONS
- Only explicit evidence-supported actions.
- Do not invent actions.

PREVENTIVE ACTIONS
- Only explicit evidence-supported actions.
- Do not invent actions.

LIMITATIONS
Mention:
- missing timestamps,
- synthetic evidence,
- unconfirmed hypotheses,
- evidence limitations.

Return ONLY the incident report.
"""

    # =========================================================
    # REPORT GENERATION
    # =========================================================

    def generate_report(
        self,
        event,
        event_type=None,
        severity=None,
        top_k=6,
    ):

        evidence = (
            self.retrieve_evidence(
                event_type=event_type,
                severity=severity,
                top_k=top_k,
            )
        )

        prompt = (
            self.build_prompt(
                event=event,
                evidence=evidence,
            )
        )

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0.0,
            },
        )

        report = (
            response[
                "message"
            ][
                "content"
            ]
        )

        return {
            "event": event,
            "report": report,
            "evidence": evidence,
            "model": OLLAMA_MODEL,
        }

    # =========================================================
    # CITATION VALIDATION
    # =========================================================

    @staticmethod
    def validate_citations(
        report,
        evidence,
    ):

        """
        Deterministically verify that every cited
        Incident ID and Procedure ID exists in the
        retrieved evidence.

        This does not prove that the generated claim is
        semantically correct. It catches unsupported IDs.
        """

        valid_incident_ids = {
            item.get("incident_id")
            for item in evidence
            if item.get("incident_id")
        }

        valid_procedure_ids = {
            item.get("procedure_id")
            for item in evidence
            if item.get("procedure_id")
        }

        cited_incident_ids = set(
            re.findall(
                r"\bINC-\d{4}\b",
                report,
            )
        )

        cited_procedure_ids = set(
            re.findall(
                r"\bFSP-SOP-\d{3}\b",
                report,
            )
        )

        invalid_incident_ids = (
            cited_incident_ids
            - valid_incident_ids
        )

        invalid_procedure_ids = (
            cited_procedure_ids
            - valid_procedure_ids
        )

        valid = (
            not invalid_incident_ids
            and not invalid_procedure_ids
        )

        return {
            "valid": valid,
            "valid_incident_ids": sorted(
                valid_incident_ids
            ),
            "valid_procedure_ids": sorted(
                valid_procedure_ids
            ),
            "cited_incident_ids": sorted(
                cited_incident_ids
            ),
            "cited_procedure_ids": sorted(
                cited_procedure_ids
            ),
            "invalid_incident_ids": sorted(
                invalid_incident_ids
            ),
            "invalid_procedure_ids": sorted(
                invalid_procedure_ids
            ),
        }

    # =========================================================
    # SAVE REPORT
    # =========================================================

    @staticmethod
    def save_result(
        result,
        citation_validation,
    ):

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path = (
            OUTPUT_DIR
            / "incident_report.md"
        )

        report_path.write_text(
            result["report"],
            encoding="utf-8",
        )

        validation_path = (
            OUTPUT_DIR
            / "incident_report_validation.json"
        )

        validation_path.write_text(
            json.dumps(
                citation_validation,
                indent=4,
            ),
            encoding="utf-8",
        )

        return (
            report_path,
            validation_path,
        )


# =============================================================
# MAIN
# =============================================================

def main():

    generator = (
        ForgeShieldIncidentReportGenerator()
    )

    # ---------------------------------------------------------
    # Load the REAL event produced by Part 01
    # ---------------------------------------------------------

    event = (
        generator.load_event()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FORGESHIELD | END-TO-END INCIDENT INTELLIGENCE"
    )

    print(
        "=" * 70
    )

    print(
        "\nEvent loaded from:"
    )

    print(
        EVENT_PATH
    )

    print(
        "\nMachine ID:"
    )

    print(
        event["machine_id"]
    )

    print(
        "\nEvent Type:"
    )

    print(
        event["event_type"]
    )

    print(
        "\nRisk:"
    )

    print(
        f'{event["risk_score"]:.6f} '
        f'({event["risk_band"]})'
    )

    # ---------------------------------------------------------
    # Generate report
    # ---------------------------------------------------------

    result = (
        generator.generate_report(
            event=event,
            event_type=event.get(
                "event_type"
            ),
            top_k=6,
        )
    )

    # ---------------------------------------------------------
    # Validate generated citations
    # ---------------------------------------------------------

    citation_validation = (
        generator.validate_citations(
            report=result["report"],
            evidence=result["evidence"],
        )
    )

    # ---------------------------------------------------------
    # Print report
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATED INCIDENT REPORT"
    )

    print(
        "=" * 70
    )

    print(
        result["report"]
    )

    # ---------------------------------------------------------
    # Print evidence
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EVIDENCE USED"
    )

    print(
        "=" * 70
    )

    for item in result["evidence"]:

        print(
            f"\n[{item['final_rank']}] "
            f"{item.get('source')}"
        )

        print(
            f"    Role: "
            f"{item.get('retrieval_role')}"
        )

        print(
            f"    Incident ID: "
            f"{item.get('incident_id')}"
        )

        print(
            f"    Procedure ID: "
            f"{item.get('procedure_id')}"
        )

        print(
            f"    Event Type: "
            f"{item.get('event_type')}"
        )

        print(
            f"    Severity: "
            f"{item.get('severity')}"
        )

        print(
            f"    Distance: "
            f"{item.get('distance')}"
        )

    # ---------------------------------------------------------
    # Citation validation
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CITATION VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nValid citations: "
        f"{citation_validation['valid']}"
    )

    print(
        "Cited incident IDs: "
        f"{citation_validation['cited_incident_ids']}"
    )

    print(
        "Cited procedure IDs: "
        f"{citation_validation['cited_procedure_ids']}"
    )

    print(
        "Invalid incident IDs: "
        f"{citation_validation['invalid_incident_ids']}"
    )

    print(
        "Invalid procedure IDs: "
        f"{citation_validation['invalid_procedure_ids']}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    (
        report_path,
        validation_path,
    ) = generator.save_result(
        result=result,
        citation_validation=citation_validation,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "END-TO-END INTEGRATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved incident report:"
    )

    print(
        report_path
    )

    print(
        "\nSaved citation validation:"
    )

    print(
        validation_path
    )


if __name__ == "__main__":

    main()