from pathlib import Path
import json
import random


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "docs"
    / "synthetic_incidents"
)

RANDOM_SEED = 42


EVENT_TYPES = [
    "machine_overheating",
    "vibration_anomaly",
    "gas_leakage",
    "guard_open",
    "unsafe_behavior",
    "slip_fall_hazard",
    "high_noise_level",
    "normal_operation",
]


MACHINE_IDS = [
    "MACHINE-001",
    "MACHINE-002",
    "MACHINE-003",
    "MACHINE-004",
    "MACHINE-005",
]


INCIDENT_TEMPLATES = {

    "machine_overheating": {
        "title": "Machine Overheating Incident",
        "summary": (
            "Elevated temperature readings were observed during "
            "machine operation, indicating a potential overheating condition."
        ),
        "contributing_factors": [
            "Elevated process temperature",
            "Reduced cooling effectiveness",
            "Extended machine operation",
            "Increased mechanical load",
        ],
        "possible_root_causes": [
            "Cooling system degradation",
            "Insufficient heat dissipation",
            "Excessive mechanical load",
            "Maintenance requirement",
        ],
        "corrective_actions": [
            "Inspect the cooling system",
            "Check process and air temperature sensors",
            "Reduce machine load if operationally safe",
            "Inspect the machine for heat-related damage",
        ],
        "preventive_actions": [
            "Schedule periodic cooling-system inspections",
            "Monitor temperature trends",
            "Maintain temperature sensors",
            "Review preventive maintenance intervals",
        ],
    },

    "vibration_anomaly": {
        "title": "Vibration Anomaly Incident",
        "summary": (
            "Abnormal machine vibration was detected during operation. "
            "The condition may indicate mechanical degradation."
        ),
        "contributing_factors": [
            "Abnormal rotational behavior",
            "Mechanical imbalance",
            "Component wear",
            "Increased operating load",
        ],
        "possible_root_causes": [
            "Bearing degradation",
            "Shaft imbalance",
            "Mechanical misalignment",
            "Component wear",
        ],
        "corrective_actions": [
            "Inspect bearings and rotating components",
            "Check shaft alignment",
            "Inspect for loose mechanical components",
            "Reduce operation if vibration continues to increase",
        ],
        "preventive_actions": [
            "Perform periodic vibration inspections",
            "Maintain rotating components",
            "Track vibration trends",
            "Include alignment checks in preventive maintenance",
        ],
    },

    "gas_leakage": {
        "title": "Gas Leakage Safety Incident",
        "summary": (
            "A potential gas leakage condition was identified in the "
            "operating environment and requires immediate safety assessment."
        ),
        "contributing_factors": [
            "Abnormal gas concentration",
            "Possible equipment leakage",
            "Insufficient ventilation",
            "Potential seal degradation",
        ],
        "possible_root_causes": [
            "Damaged pipe or hose",
            "Failed seal",
            "Loose connection",
            "Equipment malfunction",
        ],
        "corrective_actions": [
            "Follow the applicable emergency shutdown procedure",
            "Restrict access to the affected area",
            "Inspect the suspected leakage source",
            "Verify the atmosphere is safe before resuming operation",
        ],
        "preventive_actions": [
            "Inspect gas-handling components regularly",
            "Test gas detection equipment",
            "Maintain ventilation systems",
            "Review emergency response procedures",
        ],
    },

    "guard_open": {
        "title": "Machine Guard Open Incident",
        "summary": (
            "A machine guard was found in an open or unsecured condition "
            "during operation."
        ),
        "contributing_factors": [
            "Guard not properly secured",
            "Maintenance activity",
            "Incorrect restart procedure",
            "Possible interlock issue",
        ],
        "possible_root_causes": [
            "Improper guard closure",
            "Interlock malfunction",
            "Maintenance procedure deviation",
            "Mechanical damage to the guard",
        ],
        "corrective_actions": [
            "Stop or isolate the machine according to the applicable procedure",
            "Secure the machine guard",
            "Verify interlock operation",
            "Do not resume operation until the guard condition is verified",
        ],
        "preventive_actions": [
            "Inspect guards and interlocks regularly",
            "Verify guard status before startup",
            "Train operators on safe restart procedures",
            "Document guard-related maintenance",
        ],
    },

    "unsafe_behavior": {
        "title": "Unsafe Operator Behavior Incident",
        "summary": (
            "An unsafe operating behavior was observed that could increase "
            "the likelihood of personnel injury or equipment damage."
        ),
        "contributing_factors": [
            "Deviation from operating procedure",
            "Insufficient use of protective equipment",
            "Unsafe interaction with machinery",
            "Inadequate procedure compliance",
        ],
        "possible_root_causes": [
            "Procedure not followed",
            "Insufficient safety training",
            "Unclear operating instructions",
            "Workplace safety oversight gap",
        ],
        "corrective_actions": [
            "Stop the unsafe activity",
            "Ensure the operator follows the applicable safety procedure",
            "Verify required protective equipment",
            "Report and document the safety event",
        ],
        "preventive_actions": [
            "Provide periodic safety training",
            "Review operating procedures",
            "Conduct routine safety observations",
            "Reinforce PPE requirements",
        ],
    },

    "slip_fall_hazard": {
        "title": "Slip and Fall Hazard Incident",
        "summary": (
            "A potential slip or fall hazard was identified in the "
            "machine operating area."
        ),
        "contributing_factors": [
            "Wet or contaminated floor",
            "Obstruction in walking path",
            "Poor housekeeping",
            "Insufficient hazard marking",
        ],
        "possible_root_causes": [
            "Fluid leakage",
            "Inadequate housekeeping",
            "Unresolved floor obstruction",
            "Insufficient hazard controls",
        ],
        "corrective_actions": [
            "Isolate the affected area if necessary",
            "Remove the immediate hazard",
            "Clean and dry the affected surface",
            "Document the safety condition",
        ],
        "preventive_actions": [
            "Maintain housekeeping schedules",
            "Inspect work areas regularly",
            "Mark temporary hazards",
            "Investigate recurring sources of floor contamination",
        ],
    },

    "high_noise_level": {
        "title": "High Noise Level Incident",
        "summary": (
            "A high noise condition was identified in the operating area. "
            "The source should be investigated and appropriate hearing "
            "protection requirements followed."
        ),
        "contributing_factors": [
            "High machine operating speed",
            "Mechanical vibration",
            "Worn components",
            "Insufficient acoustic control",
        ],
        "possible_root_causes": [
            "Mechanical component degradation",
            "Loose component",
            "Unbalanced rotating equipment",
            "Equipment operating outside expected conditions",
        ],
        "corrective_actions": [
            "Inspect the machine for abnormal mechanical noise",
            "Verify required hearing protection",
            "Inspect rotating components",
            "Measure the noise level if required",
        ],
        "preventive_actions": [
            "Perform periodic noise assessments",
            "Maintain rotating machinery",
            "Provide appropriate hearing protection",
            "Investigate recurring abnormal noise",
        ],
    },

    "normal_operation": {
        "title": "Normal Machine Operation",
        "summary": (
            "Machine operation remained within the expected operating "
            "conditions during the observed period."
        ),
        "contributing_factors": [
            "Stable operating conditions",
            "Expected machine behavior",
            "No significant safety deviation observed",
        ],
        "possible_root_causes": [
            "No abnormal root cause identified",
        ],
        "corrective_actions": [
            "No immediate corrective action indicated",
            "Continue routine monitoring",
        ],
        "preventive_actions": [
            "Continue scheduled preventive maintenance",
            "Continue routine safety inspections",
            "Monitor operating trends",
        ],
    },
}


def make_sensor_context(event_type, index):

    random.seed(
        RANDOM_SEED + index
    )

    air_temperature = round(
        random.uniform(
            296.0,
            304.0,
        ),
        2,
    )

    process_temperature = round(
        air_temperature
        + random.uniform(
            8.0,
            13.0,
        ),
        2,
    )

    rotational_speed = random.randint(
        1200,
        2200,
    )

    torque = round(
        random.uniform(
            25.0,
            65.0,
        ),
        2,
    )

    tool_wear = random.randint(
        10,
        240,
    )

    # Introduce plausible abnormal values
    # for selected event categories.

    if event_type == "machine_overheating":

        air_temperature = round(
            random.uniform(
                301.5,
                304.5,
            ),
            2,
        )

        process_temperature = round(
            air_temperature
            + random.uniform(
                12.0,
                14.0,
            ),
            2,
        )

    elif event_type == "vibration_anomaly":

        rotational_speed = random.randint(
            2300,
            2800,
        )

        torque = round(
            random.uniform(
                55.0,
                75.0,
            ),
            2,
        )

    elif event_type == "high_noise_level":

        rotational_speed = random.randint(
            2200,
            2800,
        )

    elif event_type == "normal_operation":

        rotational_speed = random.randint(
            1400,
            1800,
        )

        torque = round(
            random.uniform(
                30.0,
                50.0,
            ),
            2,
        )

        tool_wear = random.randint(
            10,
            120,
        )

    return {
        "air_temperature": air_temperature,
        "process_temperature":
            process_temperature,
        "rotational_speed":
            rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }


def generate_incident(
    incident_id,
    event_type,
):

    template = (
        INCIDENT_TEMPLATES[
            event_type
        ]
    )

    machine_id = (
        MACHINE_IDS[
            (incident_id - 1)
            % len(MACHINE_IDS)
        ]
    )

    sensor_context = (
        make_sensor_context(
            event_type,
            incident_id,
        )
    )

    severity_map = {
        "normal_operation":
            "Low",
        "high_noise_level":
            "Medium",
        "slip_fall_hazard":
            "High",
        "unsafe_behavior":
            "High",
        "guard_open":
            "High",
        "vibration_anomaly":
            "High",
        "machine_overheating":
            "High",
        "gas_leakage":
            "Critical",
    }

    severity = severity_map[
        event_type
    ]

    incident = {
        "incident_id":
            f"INC-{incident_id:04d}",

        "machine_id":
            machine_id,

        "event_type":
            event_type,

        "severity":
            severity,

        "is_synthetic":
            True,

        "title":
            template["title"],

        "incident_summary":
            template["summary"],

        "sensor_context":
            sensor_context,

        "possible_contributing_factors":
            template[
                "contributing_factors"
            ],

        "potential_root_causes":
            template[
                "possible_root_causes"
            ],

        "corrective_actions":
            template[
                "corrective_actions"
            ],

        "preventive_actions":
            template[
                "preventive_actions"
            ],
    }

    return incident


def write_markdown(
    incident,
):

    filename = (
        OUTPUT_DIR
        / (
            f"{incident['incident_id']}_"
            f"{incident['event_type']}.md"
        )
    )

    sensor = incident[
        "sensor_context"
    ]

    lines = [
        f"# {incident['title']}",
        "",
        f"**Incident ID:** "
        f"{incident['incident_id']}",
        "",
        f"**Machine ID:** "
        f"{incident['machine_id']}",
        "",
        f"**Event Type:** "
        f"{incident['event_type']}",
        "",
        f"**Severity:** "
        f"{incident['severity']}",
        "",
        f"**Synthetic Record:** "
        f"{incident['is_synthetic']}",
        "",
        "## Incident Summary",
        "",
        incident[
            "incident_summary"
        ],
        "",
        "## Sensor Context",
        "",
        "| Sensor | Value |",
        "|---|---:|",
        (
            f"| Air temperature | "
            f"{sensor['air_temperature']} |"
        ),
        (
            f"| Process temperature | "
            f"{sensor['process_temperature']} |"
        ),
        (
            f"| Rotational speed | "
            f"{sensor['rotational_speed']} |"
        ),
        (
            f"| Torque | "
            f"{sensor['torque']} |"
        ),
        (
            f"| Tool wear | "
            f"{sensor['tool_wear']} |"
        ),
        "",
        "## Possible Contributing Factors",
        "",
    ]

    for item in incident[
        "possible_contributing_factors"
    ]:

        lines.append(
            f"- {item}"
        )

    lines.extend(
        [
            "",
            "## Potential Root Causes",
            "",
        ]
    )

    for item in incident[
        "potential_root_causes"
    ]:

        lines.append(
            f"- {item}"
        )

    lines.extend(
        [
            "",
            "## Corrective Actions",
            "",
        ]
    )

    for item in incident[
        "corrective_actions"
    ]:

        lines.append(
            f"- {item}"
        )

    lines.extend(
        [
            "",
            "## Preventive Actions",
            "",
        ]
    )

    for item in incident[
        "preventive_actions"
    ]:

        lines.append(
            f"- {item}"
        )

    lines.extend(
        [
            "",
            "## Data Provenance",
            "",
            (
                "This document is a synthetic safety-event "
                "record generated for the ForgeShield research "
                "proof of concept."
            ),
            "",
            (
                "It must not be interpreted as a real-world "
                "incident report or as evidence of an actual "
                "industrial event."
            ),
        ]
    )

    filename.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():

    print("\n" + "=" * 70)

    print(
        "FORGESHIELD | SYNTHETIC SAFETY INCIDENT GENERATION"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(
        RANDOM_SEED
    )

    # ---------------------------------------------------------
    # Generate balanced knowledge-base examples
    # ---------------------------------------------------------

    incidents = []

    incident_id = 1

    incidents_per_type = 10

    for event_type in EVENT_TYPES:

        for _ in range(
            incidents_per_type
        ):

            incident = generate_incident(
                incident_id,
                event_type,
            )

            incidents.append(
                incident
            )

            write_markdown(
                incident
            )

            incident_id += 1

    # ---------------------------------------------------------
    # Save machine-readable index
    # ---------------------------------------------------------

    index = {
        "dataset_name":
            "ForgeShield Synthetic Safety Incident Knowledge Base",

        "is_synthetic":
            True,

        "random_seed":
            RANDOM_SEED,

        "total_incidents":
            len(incidents),

        "event_types":
            EVENT_TYPES,

        "incidents_per_event_type":
            incidents_per_type,

        "source":
            "Generated locally for ForgeShield research POC",

        "usage":
            (
                "Synthetic records for RAG retrieval, "
                "incident interpretation, and dashboard demonstration."
            ),
    }

    index_path = (
        OUTPUT_DIR
        / "index.json"
    )

    index_path.write_text(
        json.dumps(
            index,
            indent=4,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        f"\nGenerated "
        f"{len(incidents)} synthetic incident records."
    )

    print(
        "\nEvent distribution:"
    )

    for event_type in EVENT_TYPES:

        count = sum(
            1
            for incident in incidents
            if incident[
                "event_type"
            ] == event_type
        )

        print(
            f"  {event_type:25s} "
            f"{count:2d}"
        )

    print(
        "\nSynthetic provenance:"
    )

    print(
        "  is_synthetic = True"
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SYNTHETIC INCIDENT GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()