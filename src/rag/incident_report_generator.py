import json
import ollama

from retriever import ForgeShieldRetriever


OLLAMA_MODEL = "qwen3:8b"


class ForgeShieldIncidentReportGenerator:

    def __init__(self):
        print("Loading ForgeShield incident report generator...")

        self.retriever = ForgeShieldRetriever()

        print(f"Ollama model: {OLLAMA_MODEL}")

    def retrieve_evidence(
        self,
        event_type=None,
        severity=None,
        top_k=6,
    ):
        """
        Retrieve evidence through two independent retrieval
        paths:

        1. Historical incident retrieval
        2. Safety procedure retrieval

        Separating the queries prevents incident records from
        dominating procedure retrieval.
        """

        # -----------------------------------------------------
        # Query construction
        # -----------------------------------------------------

        if event_type:

            readable_event = (
                event_type.replace("_", " ")
            )

            incident_query = (
                f"historical industrial safety incident "
                f"machine {readable_event}"
            )

            procedure_query = (
                f"safety procedure SOP "
                f"machine {readable_event} "
                f"operator response corrective actions "
                f"preventive actions"
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
        # 1. Historical incident retrieval
        # -----------------------------------------------------

        incident_evidence = self.retriever.retrieve(
            query=incident_query,
            top_k=top_k,
            document_type="synthetic_incident",
            event_type=event_type,
            severity=severity,
        )

        # -----------------------------------------------------
        # Incident fallback
        # -----------------------------------------------------

        if len(incident_evidence) < 2:

            fallback_incidents = self.retriever.retrieve(
                query=incident_query,
                top_k=top_k,
                document_type="synthetic_incident",
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
                    incident_evidence.append(item)
                    existing_ids.add(incident_id)

                if len(incident_evidence) >= top_k:
                    break

        # -----------------------------------------------------
        # 2. Safety procedure retrieval
        # -----------------------------------------------------

        procedure_evidence = self.retriever.retrieve(
            query=procedure_query,
            top_k=top_k,
            document_type="safety_procedure",
        )

        # -----------------------------------------------------
        # 3. Event-specific procedure filtering
        # -----------------------------------------------------

        if event_type:

            event_terms = [
                term
                for term in (
                    event_type
                    .replace("_", " ")
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
                    source + " " + content
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
        # 4. Deduplicate procedure chunks
        # -----------------------------------------------------

        unique_procedures = []
        seen = set()

        for item in procedure_evidence:

            key = (
                item.get("procedure_id"),
                item.get("source"),
                item.get("content"),
            )

            if key not in seen:

                unique_procedures.append(item)
                seen.add(key)

        procedure_evidence = (
            unique_procedures
        )

        # -----------------------------------------------------
        # 5. Sort evidence by retrieval distance
        # -----------------------------------------------------

        incident_evidence.sort(
            key=lambda x: x["distance"]
        )

        procedure_evidence.sort(
            key=lambda x: x["distance"]
        )

        # -----------------------------------------------------
        # 6. Limit each evidence type
        # -----------------------------------------------------

        incident_evidence = (
            incident_evidence[:top_k]
        )

        procedure_evidence = (
            procedure_evidence[:top_k]
        )

        # -----------------------------------------------------
        # 7. Combine evidence
        # -----------------------------------------------------

        evidence = []

        for item in incident_evidence:

            item = dict(item)

            item["retrieval_role"] = (
                "incident"
            )

            evidence.append(item)

        for item in procedure_evidence:

            item = dict(item)

            item["retrieval_role"] = (
                "procedure"
            )

            evidence.append(item)

        # -----------------------------------------------------
        # 8. Ensure both evidence types are represented
        # -----------------------------------------------------

        # If procedures exist, keep at least one procedure.
        # If incidents exist, keep at least one incident.

        incident_items = [
            item
            for item in evidence
            if item["retrieval_role"]
            == "incident"
        ]

        procedure_items = [
            item
            for item in evidence
            if item["retrieval_role"]
            == "procedure"
        ]

        # -----------------------------------------------------
        # 9. Balanced final evidence set
        # -----------------------------------------------------

        if incident_items and procedure_items:

            incident_count = max(
                2,
                top_k // 2,
            )

            procedure_count = (
                top_k - incident_count
            )

            selected = (
                incident_items[:incident_count]
                + procedure_items[:procedure_count]
            )

        else:

            selected = evidence[:top_k]

        # -----------------------------------------------------
        # 10. Final ranking
        # -----------------------------------------------------

        selected.sort(
            key=lambda x: x["distance"]
        )

        for rank, item in enumerate(
            selected,
            start=1,
        ):
            item["final_rank"] = rank

        return selected

    @staticmethod
    def build_prompt(
        event,
        evidence,
    ):
        """
        Build a strictly grounded incident-report prompt.
        """

        evidence_blocks = []

        for item in evidence:

            content = item.get(
                "content",
                "",
            )

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
{content}

--- END EVIDENCE ---
"""

            evidence_blocks.append(
                block
            )

        evidence_text = "\n".join(
            evidence_blocks
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
STRICT GROUNDING RULES
==================================================

1. EVENT DATA is the primary source for the
   current incident.

2. Values explicitly present in EVENT DATA are
   observed values.

3. Do not invent facts, measurements, timestamps,
   events, procedures, causes, sources, identifiers,
   actions, thresholds, or operating ranges.

4. Do not describe a sensor value as normal,
   abnormal, high, low, safe, unsafe, excessive,
   or within an operational range unless the
   retrieved evidence explicitly supports that
   interpretation.

5. Historical incident records describe previous
   observations. They do NOT prove that the same
   event occurred in the current EVENT DATA.

6. Procedure documents describe recommended
   procedures. They do NOT prove that a procedure
   was performed during the current incident.

7. Keep current incident observations separate
   from historical incident observations.

8. Possible contributing factors must be presented
   as hypotheses unless directly established.

9. Potential root causes must be presented as
   hypotheses unless directly established.

10. Corrective actions may ONLY come from:
    - retrieved procedure evidence, OR
    - explicit corrective actions in retrieved
      incident evidence.

11. Preventive actions may ONLY come from:
    - retrieved procedure evidence, OR
    - explicit preventive actions in retrieved
      incident evidence.

12. If no evidence supports an action, do not
    invent one.

13. Do not combine unrelated evidence to create
    a new safety recommendation.

14. Do not use outside knowledge.

15. Do not invent numerical values.

16. Every numerical value must come directly from
    EVENT DATA or RETRIEVED EVIDENCE.

17. Do not invent timestamps.

18. Do not invent Incident IDs or Procedure IDs.

19. Every cited ID must exist in the retrieved
    evidence.

20. Do not cite unrelated evidence.

21. Synthetic incident records must remain explicitly
    identified as synthetic.

22. Never claim that a synthetic incident represents
    a real-world incident.

23. If evidence is insufficient for a claim,
    explicitly state that evidence is insufficient.

24. Prefer omission over unsupported claims.

25. Keep FACTS, HYPOTHESES, and ACTIONS separate.

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
- Summarize the current EVENT DATA only.
- Include machine ID, event type, risk information,
  and provided sensor values.

TIMELINE OF EVENTS
- Use only explicitly provided timestamps/events.
- If unavailable, state:
  "No timestamps or chronological event sequence
  are provided in the event data."

ABNORMAL SENSOR BEHAVIOR
- Report sensor values from EVENT DATA.
- Do not independently determine whether a value
  is abnormal unless evidence supports it.

POSSIBLE CONTRIBUTING FACTORS
- Use only evidence-supported factors.
- Clearly label hypotheses.

POTENTIAL ROOT CAUSES
- Use only evidence-supported hypotheses.
- Never present an inferred cause as confirmed.

EVIDENCE
Separate:
1. Current EVENT DATA
2. Historical incident evidence
3. Safety procedure evidence

Cite relevant Incident IDs and Procedure IDs.

CORRECTIVE ACTIONS
- Only include actions explicitly supported by
  retrieved evidence.

PREVENTIVE ACTIONS
- Only include actions explicitly supported by
  retrieved evidence.

LIMITATIONS
- Mention missing information.
- Mention synthetic evidence.
- Mention unconfirmed causes.
- Do not claim real-world validation.

Return ONLY the incident report.
"""

    def generate_report(
        self,
        event,
        event_type=None,
        severity=None,
        top_k=6,
    ):

        evidence = self.retrieve_evidence(
            event_type=event_type,
            severity=severity,
            top_k=top_k,
        )

        prompt = self.build_prompt(
            event=event,
            evidence=evidence,
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

        report = response[
            "message"
        ]["content"]

        return {
            "event": event,
            "report": report,
            "evidence": evidence,
            "model": OLLAMA_MODEL,
        }


def main():

    generator = (
        ForgeShieldIncidentReportGenerator()
    )

    # ---------------------------------------------------------
    # Demonstration high-risk overheating event
    # ---------------------------------------------------------

    event = {
        "machine_id": "MACHINE-DEMO-001",
        "event_type": "machine_overheating",
        "risk_score": 0.91,
        "risk_band": "Critical",
        "failure_probability": 0.87,
        "anomaly_score": 0.97,
        "air_temperature": 304.1,
        "process_temperature": 313.2,
        "rotational_speed": 1420,
        "torque": 61.4,
        "tool_wear": 188,
        "temperature_differential": 9.1,
        "mechanical_power": 9125.0,
    }

    print(
        "\n" + "=" * 70
    )

    print(
        "FORGESHIELD | INCIDENT REPORT GENERATOR"
    )

    print(
        "=" * 70
    )

    result = generator.generate_report(
        event=event,
        event_type="machine_overheating",
        severity="high",
        top_k=6,
    )

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


if __name__ == "__main__":
    main()