from pathlib import Path

# ForgeShield design tokens
BACKGROUND = "#0A0C10"
SURFACE = "#12151A"
SURFACE_2 = "#151920"

BORDER = "#1C2028"
BORDER_STRONG = "#292F38"

TEXT = "#E7E9EC"
MUTED = "#868D97"
MUTED_2 = "#646B75"

BRAND = "#FF6B35"

CRITICAL = "#EF4B5C"
HIGH = "#F0A63C"
MEDIUM = "#E8C547"
LOW = "#38C98F"

PAGE_RADIUS = "8px"
PANEL_RADIUS = "10px"
SMALL_RADIUS = "4px"

RISK_COLORS = {
    "Critical": CRITICAL,
    "High": HIGH,
    "Medium": MEDIUM,
    "Low": LOW,
}


def load_global_css() -> str:
    """Return the shared ForgeShield dashboard stylesheet."""
    return r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --fs-bg: {BG};
    --fs-surface: {SURFACE};
    --fs-surface-2: {SURFACE_2};
    --fs-border: {BORDER};
    --fs-border-strong: {BORDER_STRONG};
    --fs-text: {TEXT};
    --fs-muted: {MUTED};
    --fs-muted-2: {MUTED_2};
    --fs-brand: {BRAND};
    --fs-critical: {CRITICAL};
    --fs-high: {HIGH};
    --fs-medium: {MEDIUM};
    --fs-low: {LOW};
}}

html, body, [class*="css"] {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.stApp {{
    background:
        radial-gradient(
            circle at 78% 0%,
            rgba(255, 107, 53, 0.035),
            transparent 34%
        ),
        var(--fs-bg);
    color: var(--fs-text);
}}

[data-testid="stAppViewContainer"] {{
    background: transparent;
}}

[data-testid="stHeader"] {{
    background: rgba(10, 12, 16, 0.94);
    border-bottom: 1px solid rgba(28, 32, 40, 0.75);
}}

[data-testid="stToolbar"] {{
    visibility: visible;
}}

[data-testid="stDecoration"] {{
    display: none;
}}

/* -----------------------------------------------------------------------
   Sidebar
   ----------------------------------------------------------------------- */

section[data-testid="stSidebar"] {{
    background: #0D1116;
    border-right: 1px solid var(--fs-border);
}}

section[data-testid="stSidebar"] > div {{
    padding-top: 1.15rem;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    margin: 0;
}}

section[data-testid="stSidebar"] .stRadio > label {{
    display: none;
}}

section[data-testid="stSidebar"] .stRadio > div {{
    gap: 3px;
}}

section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
    gap: 3px;
}}

section[data-testid="stSidebar"] .stRadio label {{
    min-height: 34px;
    padding: 7px 10px;
    border-radius: 6px;
    border: 1px solid transparent;
    color: #AAB0B9;
    transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
}}

section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(255,255,255,0.025);
    color: var(--fs-text);
}}

section[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background: rgba(255, 107, 53, 0.075);
    border-color: rgba(255, 107, 53, 0.18);
    color: var(--fs-text);
}}

section[data-testid="stSidebar"] .stRadio label > div:first-child {{
    display: none;
}}

section[data-testid="stSidebar"] .stRadio label p {{
    font-size: 13px;
    font-weight: 500;
    line-height: 1.25;
}}

section[data-testid="stSidebar"] .stRadio label:has(input:checked) p::before {{
    content: "◆";
    color: var(--fs-brand);
    font-size: 8px;
    margin-right: 9px;
    vertical-align: 2px;
}}

section[data-testid="stSidebar"] .stRadio label:not(:has(input:checked)) p::before {{
    content: "·";
    color: #4B525C;
    font-size: 15px;
    margin-right: 9px;
    vertical-align: -1px;
}}

/* Keep Streamlit's collapse affordance, but make it part of the chrome. */
button[data-testid="stSidebarCollapseButton"] {{
    border: 1px solid var(--fs-border) !important;
    background: #10141A !important;
    border-radius: 5px !important;
    width: 30px !important;
    height: 30px !important;
    margin: 10px !important;
}}

button[data-testid="stSidebarCollapseButton"]:hover {{
    border-color: rgba(255,107,53,0.35) !important;
    background: #141920 !important;
}}

button[data-testid="stSidebarCollapseButton"] svg {{
    color: #AAB0B9 !important;
}}

/* -----------------------------------------------------------------------
   Top bar / search
   ----------------------------------------------------------------------- */

.fs-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 34px;
}}

.fs-search-wrap {{
    flex: 1;
    max-width: 100%;
}}

.fs-search-caption {{
    color: var(--fs-muted-2);
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    margin-top: 5px;
    text-align: right;
}}

.fs-refresh {{
    min-width: 165px;
    text-align: right;
    color: var(--fs-muted);
    font-size: 11px;
    line-height: 1.45;
}}

.fs-refresh strong {{
    color: var(--fs-text);
    font-weight: 500;
}}

div[data-testid="stTextInput"] {{
    margin-bottom: 0;
}}

div[data-testid="stTextInput"] label {{
    display: none;
}}

div[data-testid="stTextInput"] > div > div {{
    background: transparent !important;
    border: 1px solid var(--fs-border) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}}

div[data-testid="stTextInput"] input {{
    color: var(--fs-text) !important;
    background: transparent !important;
    font-size: 13px !important;
}}

div[data-testid="stTextInput"] input::placeholder {{
    color: #666E79 !important;
}}

div[data-testid="stTextInput"] > div > div:focus-within {{
    border-color: rgba(255,107,53,0.45) !important;
    box-shadow: 0 0 0 1px rgba(255,107,53,0.12) !important;
}}

/* -----------------------------------------------------------------------
   Main headings
   ----------------------------------------------------------------------- */

.fs-eyebrow {{
    color: #717984;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 9px;
}}

.fs-page-title {{
    color: var(--fs-text);
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.035em;
    line-height: 1.08;
    margin-bottom: 8px;
}}

.fs-page-subtitle {{
    color: var(--fs-muted);
    font-size: 13px;
    line-height: 1.5;
}}

.fs-section-spacer {{
    height: 26px;
}}

/* -----------------------------------------------------------------------
   Dataset selector
   ----------------------------------------------------------------------- */

div[data-testid="stSelectbox"] label {{
    display: none;
}}

div[data-testid="stSelectbox"] > div > div {{
    background: var(--fs-surface) !important;
    border: 1px solid var(--fs-border) !important;
    border-radius: 6px !important;
    color: var(--fs-text) !important;
    min-height: 38px;
}}

div[data-testid="stSelectbox"] svg {{
    color: var(--fs-muted) !important;
}}

/* -----------------------------------------------------------------------
   KPI cards
   ----------------------------------------------------------------------- */

.fs-kpi {{
    background: var(--fs-surface);
    border: 1px solid var(--fs-border);
    border-radius: {PAGE_RADIUS};
    min-height: 145px;
    padding: 21px 22px 18px;
}}

.fs-kpi-label {{
    color: #7F8792;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 15px;
}}

.fs-kpi-value {{
    color: var(--fs-text);
    font-family: "JetBrains Mono", monospace;
    font-size: 38px;
    font-weight: 400;
    letter-spacing: -0.045em;
    line-height: 1;
}}

.fs-kpi-delta {{
    color: #707781;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    margin-top: 12px;
}}

.fs-kpi-meta {{
    color: #737B85;
    font-size: 11px;
    line-height: 1.35;
    margin-top: 8px;
}}

.fs-kpi-critical {{
    color: var(--fs-critical);
}}

/* -----------------------------------------------------------------------
   Panels
   ----------------------------------------------------------------------- */

.fs-panel {{
    background: var(--fs-surface);
    border: 1px solid var(--fs-border);
    border-radius: {PANEL_RADIUS};
    overflow: hidden;
}}

.fs-panel-header {{
    min-height: 62px;
    padding: 18px 20px;
    border-bottom: 1px solid var(--fs-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}}

.fs-panel-title {{
    color: var(--fs-text);
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.015em;
}}

.fs-panel-meta {{
    color: #6D7580;
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
}}

.fs-chart-body {{
    padding: 8px 10px 5px;
}}

/* -----------------------------------------------------------------------
   Donut legend
   ----------------------------------------------------------------------- */

.fs-legend {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border-top: 1px solid var(--fs-border);
}}

.fs-legend-item {{
    padding: 12px 10px;
    text-align: center;
    border-right: 1px solid var(--fs-border);
}}

.fs-legend-item:last-child {{
    border-right: 0;
}}

.fs-legend-name {{
    color: #969DA6;
    font-size: 10px;
}}

.fs-legend-count {{
    color: var(--fs-text);
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    margin-top: 4px;
}}

/* -----------------------------------------------------------------------
   Right rail
   ----------------------------------------------------------------------- */

.fs-rail-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 14px 16px;
    border-bottom: 1px solid rgba(28,32,40,0.8);
}}

.fs-rail-item:last-child {{
    border-bottom: 0;
}}

.fs-machine-id {{
    color: var(--fs-text);
    font-family: "JetBrains Mono", monospace;
    font-size: 13px;
}}

.fs-risk-pill {{
    display: inline-flex;
    align-items: center;
    border-radius: 4px;
    padding: 3px 7px;
    font-size: 10px;
    font-weight: 600;
    line-height: 1.1;
    border: 1px solid currentColor;
}}

.fs-risk-low {{
    color: var(--fs-low);
    background: rgba(56,201,143,0.09);
}}

.fs-risk-medium {{
    color: var(--fs-medium);
    background: rgba(232,197,71,0.09);
}}

.fs-risk-high {{
    color: var(--fs-high);
    background: rgba(240,166,60,0.09);
}}

.fs-risk-critical {{
    color: var(--fs-critical);
    background: rgba(239,75,92,0.09);
}}

.fs-incident-panel {{
    border-top: 2px solid var(--fs-brand);
}}

.fs-incident-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    padding: 17px 16px;
}}

.fs-incident-label {{
    color: #737B85;
    font-size: 10px;
    margin-bottom: 4px;
}}

.fs-incident-value {{
    color: var(--fs-text);
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    overflow-wrap: anywhere;
}}

.fs-incident-summary {{
    color: #969DA6;
    font-size: 11px;
    line-height: 1.55;
    padding: 0 16px 17px;
}}

.fs-alert {{
    display: grid;
    grid-template-columns: 3px minmax(0,1fr);
    gap: 12px;
    padding: 13px 16px;
    border-bottom: 1px solid rgba(28,32,40,0.75);
}}

.fs-alert:last-child {{
    border-bottom: 0;
}}

.fs-alert-bar {{
    border-radius: 2px;
    min-height: 28px;
}}

.fs-alert-text {{
    color: #A8AEB7;
    font-size: 11px;
    line-height: 1.45;
}}

/* -----------------------------------------------------------------------
   Activity table
   ----------------------------------------------------------------------- */

.fs-table-wrap {{
    overflow-x: auto;
}}

table.fs-table {{
    width: 100%;
    border-collapse: collapse;
}}

table.fs-table th {{
    color: #737B85;
    background: #101319;
    border-bottom: 1px solid var(--fs-border);
    padding: 11px 13px;
    font-size: 10px;
    font-weight: 600;
    text-align: left;
    white-space: nowrap;
}}

table.fs-table th.num,
table.fs-table td.num {{
    text-align: right;
}}

table.fs-table td {{
    color: #C5CAD1;
    border-bottom: 1px solid rgba(28,32,40,0.75);
    padding: 10px 13px;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    white-space: nowrap;
}}

table.fs-table td.machine {{
    color: var(--fs-text);
    font-weight: 500;
}}

table.fs-table tr:last-child td {{
    border-bottom: 0;
}}

table.fs-table tbody tr {{
    transition: background 100ms ease;
}}

table.fs-table tbody tr:hover {{
    background: rgba(255,255,255,0.018);
}}

.fs-sort {{
    color: #555D68;
    font-size: 9px;
    margin-left: 4px;
}}

.fs-risk-cell {{
    text-align: left !important;
    font-family: "Inter", sans-serif !important;
}}

/* -----------------------------------------------------------------------
   Footer
   ----------------------------------------------------------------------- */

.fs-footer {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid var(--fs-border);
    margin-top: 36px;
    padding: 17px 0 24px;
    color: #59616C;
    font-size: 10px;
}}

.fs-footer-mono {{
    font-family: "JetBrains Mono", monospace;
}}

/* -----------------------------------------------------------------------
   Empty / error / loading states
   ----------------------------------------------------------------------- */

.fs-state {{
    padding: 42px 24px;
    text-align: center;
}}

.fs-state-mark {{
    color: var(--fs-brand);
    font-family: "JetBrains Mono", monospace;
    font-size: 18px;
    margin-bottom: 10px;
}}

.fs-state-title {{
    color: var(--fs-text);
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 6px;
}}

.fs-state-copy {{
    color: var(--fs-muted);
    font-size: 11px;
    line-height: 1.5;
}}

.fs-error {{
    border: 1px solid rgba(239,75,92,0.28);
    background: rgba(239,75,92,0.045);
    border-radius: 8px;
    padding: 14px 16px;
    color: #D3A1A7;
    font-size: 12px;
}}

/* -----------------------------------------------------------------------
   Streamlit native controls
   ----------------------------------------------------------------------- */

.stButton > button {{
    border: 1px solid var(--fs-border) !important;
    background: var(--fs-surface) !important;
    color: var(--fs-text) !important;
    border-radius: 5px !important;
    font-size: 11px !important;
}}

.stButton > button:hover {{
    border-color: rgba(255,107,53,0.38) !important;
    color: var(--fs-brand) !important;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--fs-border);
    border-radius: 8px;
}}

div[data-testid="stExpander"] {{
    background: var(--fs-surface);
    border: 1px solid var(--fs-border);
    border-radius: 8px;
}}


/* -----------------------------------------------------------------------
   Machine Intelligence
   ----------------------------------------------------------------------- */

.fs-machine-hero {{
    background:
        linear-gradient(135deg, rgba(255,107,53,0.055), transparent 42%),
        var(--fs-surface);
    border: 1px solid var(--fs-border);
    border-radius: 10px;
    padding: 22px;
    margin-bottom: 18px;
}}

.fs-machine-hero-top {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
}}

.fs-machine-kicker {{
    color: #737B85;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 7px;
}}

.fs-machine-title {{
    color: var(--fs-text);
    font-family: "JetBrains Mono", monospace;
    font-size: 25px;
    font-weight: 500;
    letter-spacing: -0.04em;
}}

.fs-machine-subtitle {{
    color: var(--fs-muted);
    font-size: 11px;
    margin-top: 7px;
}}

.fs-machine-risk {{
    min-width: 150px;
    text-align: right;
}}

.fs-machine-risk-label {{
    color: #737B85;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.fs-machine-risk-value {{
    font-family: "JetBrains Mono", monospace;
    font-size: 29px;
    line-height: 1;
    margin-top: 6px;
}}

.fs-machine-risk-bar {{
    height: 4px;
    background: #20252D;
    border-radius: 3px;
    margin-top: 10px;
    overflow: hidden;
}}

.fs-machine-risk-fill {{
    height: 100%;
    border-radius: 3px;
}}

.fs-detail-card {{
    background: var(--fs-surface);
    border: 1px solid var(--fs-border);
    border-radius: 9px;
    padding: 17px;
    min-height: 118px;
}}

.fs-detail-label {{
    color: #747C86;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 9px;
}}

.fs-detail-value {{
    color: var(--fs-text);
    font-family: "JetBrains Mono", monospace;
    font-size: 23px;
}}

.fs-detail-small {{
    color: var(--fs-muted);
    font-size: 10px;
    margin-top: 7px;
    line-height: 1.4;
}}

.fs-section-title {{
    color: var(--fs-text);
    font-size: 15px;
    font-weight: 600;
    margin: 24px 0 10px;
}}

.fs-sensor-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1px;
    background: var(--fs-border);
    border: 1px solid var(--fs-border);
    border-radius: 8px;
    overflow: hidden;
}}

.fs-sensor-cell {{
    background: var(--fs-surface);
    padding: 13px 15px;
}}

.fs-sensor-name {{
    color: #737B85;
    font-size: 9px;
    margin-bottom: 5px;
}}

.fs-sensor-value {{
    color: var(--fs-text);
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
}}

.fs-failure-modes {{
    display: grid;
    gap: 8px;
}}

.fs-mode {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    background: #101319;
    border: 1px solid var(--fs-border);
    border-radius: 6px;
}}

.fs-mode-name {{
    color: #A9AFB8;
    font-size: 11px;
}}

.fs-mode-state {{
    color: #5F6670;
    font-family: "JetBrains Mono", monospace;
    font-size: 9px;
}}

.fs-mode-active .fs-mode-name {{
    color: var(--fs-text);
}}

.fs-mode-active .fs-mode-state {{
    color: var(--fs-critical);
}}

.fs-insight {{
    border-left: 3px solid var(--fs-brand);
    background: rgba(255,107,53,0.045);
    border-top: 1px solid var(--fs-border);
    border-right: 1px solid var(--fs-border);
    border-bottom: 1px solid var(--fs-border);
    border-radius: 0 7px 7px 0;
    padding: 14px 16px;
    color: #A9AFB8;
    font-size: 11px;
    line-height: 1.6;
}}

.fs-machine-select {{
    margin-bottom: 15px;
}}

.fs-machine-table-note {{
    color: #666E79;
    font-size: 10px;
    margin-top: 7px;
}}

.fs-nav-note {{
    color: #626A74;
    font-size: 10px;
    line-height: 1.5;
    padding: 9px 2px 0;
}}


@media (max-width: 1100px) {{
    .fs-topbar {{
        display: block;
    }}

    .fs-refresh {{
        margin-top: 10px;
        text-align: left;
    }}

    .fs-kpi {{
        min-height: 135px;
    }}
}}
</style>
""".format(
        BG=BACKGROUND,
        SURFACE=SURFACE,
        SURFACE_2=SURFACE_2,
        BORDER=BORDER,
        BORDER_STRONG=BORDER_STRONG,
        TEXT=TEXT,
        MUTED=MUTED,
        MUTED_2=MUTED_2,
        BRAND=BRAND,
        CRITICAL=CRITICAL,
        HIGH=HIGH,
        MEDIUM=MEDIUM,
        LOW=LOW,
        PAGE_RADIUS=PAGE_RADIUS,
        PANEL_RADIUS=PANEL_RADIUS,
        SMALL_RADIUS=SMALL_RADIUS,
    )


