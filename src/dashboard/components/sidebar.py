from __future__ import annotations

import streamlit as st

from dashboard.components.theme import (
    BORDER,
    BRAND,
    LOW,
)
from dashboard.components.ui import render_html


def render_sidebar() -> str:
    """Render all navigation chrome inside the actual Streamlit sidebar."""

    navigation = [
        "Command Center",
        "Live Monitoring",
        "Machine Intelligence",
        "Predictive Health",
        "Anomaly Detection",
        "Explainability",
        "Incident Intelligence",
        "Safety Copilot",
        "Knowledge Base",
        "Documentation",
        "Settings",
    ]

    with st.sidebar:

        render_html(
            f"""
            <div style="
                padding:28px 0 25px;
                border-bottom:1px solid {BORDER};
                margin-bottom:23px;
            ">
                <div style="
                    display:flex;
                    align-items:center;
                    gap:10px;
                ">
                    <span style="
                        display:inline-flex;
                        width:15px;
                        height:15px;
                        border:2px solid {BRAND};
                        transform:rotate(45deg);
                        box-sizing:border-box;
                    "></span>

                    <span style="
                        color:#F4F5F7;
                        font-size:19px;
                        font-weight:700;
                        letter-spacing:-0.04em;
                    ">
                        ForgeShield
                    </span>
                </div>

                <div style="
                    color:#858C96;
                    font-size:9px;
                    font-weight:500;
                    letter-spacing:0.05em;
                    margin:8px 0 0 25px;
                ">
                    INDUSTRIAL SAFETY INTELLIGENCE
                </div>
            </div>

            <div style="
                color:#68707A;
                font-size:9px;
                font-weight:600;
                letter-spacing:0.11em;
                text-transform:uppercase;
                margin-bottom:9px;
            ">
                Navigation
            </div>
            """
        )

        selected = st.radio(
            "Navigation",
            navigation,
            index=0,
            key="forge_navigation",
            label_visibility="collapsed",
        )

        render_html(
            f"""
            <div style="
                margin-top:30px;
                padding-top:18px;
                border-top:1px solid {BORDER};
            ">
                <div style="
                    color:#68707A;
                    font-size:9px;
                    font-weight:600;
                    letter-spacing:0.11em;
                    text-transform:uppercase;
                    margin-bottom:12px;
                ">
                    System Status
                </div>

                <div style="display:grid;gap:12px;">

                    <div>
                        <div style="
                            display:flex;
                            align-items:center;
                            gap:8px;
                            color:#8B929C;
                            font-size:11px;
                        ">
                            <span style="
                                width:6px;
                                height:6px;
                                border-radius:50%;
                                background:{LOW};
                            "></span>
                            AI4I + C-MAPSS
                        </div>

                        <div style="
                            color:{LOW};
                            font-family:'JetBrains Mono',monospace;
                            font-size:9px;
                            margin:3px 0 0 14px;
                        ">
                            Online
                        </div>
                    </div>

                    <div>
                        <div style="
                            display:flex;
                            align-items:center;
                            gap:8px;
                            color:#8B929C;
                            font-size:11px;
                        ">
                            <span style="
                                width:6px;
                                height:6px;
                                border-radius:50%;
                                background:{LOW};
                            "></span>
                            RAG Knowledge Base
                        </div>

                        <div style="
                            color:{LOW};
                            font-family:'JetBrains Mono',monospace;
                            font-size:9px;
                            margin:3px 0 0 14px;
                        ">
                            Online
                        </div>
                    </div>

                    <div>
                        <div style="
                            display:flex;
                            align-items:center;
                            gap:8px;
                            color:#8B929C;
                            font-size:11px;
                        ">
                            <span style="
                                width:6px;
                                height:6px;
                                border-radius:50%;
                                background:{LOW};
                            "></span>
                            Local LLM (Ollama)
                        </div>

                        <div style="
                            color:{LOW};
                            font-family:'JetBrains Mono',monospace;
                            font-size:9px;
                            margin:3px 0 0 14px;
                        ">
                            Online
                        </div>
                    </div>

                </div>
            </div>

            <div style="
                color:#4D535C;
                font-family:'JetBrains Mono',monospace;
                font-size:9px;
                margin-top:48px;
            ">
                v0.1.0
            </div>
            """
        )

    return selected
