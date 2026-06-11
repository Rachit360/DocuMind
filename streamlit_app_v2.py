"""
DocuMind — Streamlit frontend wired to the real backend.

Integration changes vs placeholder version (everything else identical):
  ① Added:  sys.path insert + imports from main.py (top of file)
  ② Added:  session_state keys: result, analysis_error, recipient_email
  ③ Added:  build_display() helper — maps pipeline output → UI shape
  ④ Changed: Analyze button calls run_document_pipeline() via tmp file
  ⑤ Changed: chat user_input calls real answer_document_question()
  ⑥ Changed: Suggested-question buttons call answer_document_question()
  ⑦ Changed: Export download data pulled from live result dict
  All CSS, layout, HTML blocks unchanged from original placeholder version.
"""

import io
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ─── ① Backend import ────────────────────────────────────────────────────────
# streamlit_app.py must live in the same directory as main.py and rag.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import (
        run_document_pipeline,
        answer_document_question,
        send_summary_email,
        log_to_google_sheet,
    )
    BACKEND_AVAILABLE = True
except ImportError as _e:
    BACKEND_AVAILABLE = False
    _IMPORT_ERROR = str(_e)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind · AI Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS (full, identical to placeholder version) ────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:       #0d0f1a;
    --bg-surface:    #141627;
    --bg-card:       #1a1d2e;
    --bg-card-hover: #1f2338;
    --border:        #252840;
    --border-light:  #2e3155;
    --accent:        #6366f1;
    --accent-soft:   rgba(99,102,241,.15);
    --accent-glow:   rgba(99,102,241,.25);
    --green:         #22c55e;
    --green-soft:    rgba(34,197,94,.12);
    --amber:         #f59e0b;
    --amber-soft:    rgba(245,158,11,.12);
    --red:           #ef4444;
    --red-soft:      rgba(239,68,68,.12);
    --text-primary:  #f1f3f9;
    --text-secondary:#8b93b0;
    --text-muted:    #4e5578;
    --radius:        10px;
    --radius-lg:     14px;
}

/* ── Global reset ── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }
.sidebar-inner { padding: 24px 20px; }

/* ── Brand block ── */
.brand-block {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}
.brand-icon {
    width: 36px; height: 36px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.brand-name { font-size: 17px; font-weight: 700; letter-spacing: -.3px; color: var(--text-primary); }
.brand-tagline { font-size: 10px; color: var(--text-muted); letter-spacing: .3px; text-transform: uppercase; }
.ai-badge {
    margin-left: auto;
    background: var(--accent-soft);
    border: 1px solid var(--accent);
    color: #a5b4fc;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: .5px;
    text-transform: uppercase;
    padding: 3px 7px;
    border-radius: 20px;
}

/* ── Sidebar section label ── */
.sidebar-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .8px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 20px 0 10px;
}

/* ── Upload zone ── */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed var(--border-light);
    border-radius: var(--radius-lg);
    padding: 24px 16px;
    text-align: center;
    margin-bottom: 16px;
    transition: border-color .2s;
}
.upload-zone:hover { border-color: var(--accent); }
.upload-zone-icon { font-size: 28px; margin-bottom: 8px; }
.upload-zone-title { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.upload-zone-sub { font-size: 11px; color: var(--text-secondary); margin-top: 4px; }
.format-tags { display: flex; gap: 5px; justify-content: center; flex-wrap: wrap; margin-top: 10px; }
.fmt-tag {
    background: var(--bg-base);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 9px; font-weight: 600;
    padding: 2px 7px; border-radius: 4px;
    letter-spacing: .4px;
}

/* ── File info ── */
.file-info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 12px 14px;
    margin-bottom: 16px;
}
.file-info-name { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.file-info-meta { font-size: 11px; color: var(--text-secondary); margin-top: 2px; }

/* ── Analysis options ── */
.options-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    margin-bottom: 16px;
}
.options-title {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .7px;
    color: var(--text-secondary); margin-bottom: 12px;
    display: flex; align-items: center; gap: 6px;
}

/* ── Status dot ── */
.status-row {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 12px;
    background: var(--green-soft);
    border: 1px solid rgba(34,197,94,.2);
    border-radius: var(--radius);
    font-size: 12px; font-weight: 500; color: var(--green);
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex-shrink: 0; }

/* ── Error row (new) ── */
.error-row {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 12px;
    background: var(--red-soft);
    border: 1px solid rgba(239,68,68,.25);
    border-radius: var(--radius);
    font-size: 12px; font-weight: 500; color: var(--red);
    margin-top: 8px;
}

/* ── Main header bar ── */
.main-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}
.main-header-left h1 {
    font-size: 20px; font-weight: 700;
    color: var(--text-primary); margin: 0; letter-spacing: -.3px;
}
.main-header-left p {
    font-size: 12px; color: var(--text-secondary); margin: 4px 0 0;
}
.header-badges { display: flex; gap: 8px; }
.hbadge {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 11px; color: var(--text-secondary);
    padding: 4px 10px;
    display: flex; align-items: center; gap: 5px;
}
.hbadge-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; }
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s, background .2s;
}
.metric-card:hover { background: var(--bg-card-hover); border-color: var(--border-light); }
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent);
    opacity: .6;
}
.metric-card.green::before { background: var(--green); }
.metric-card.amber::before { background: var(--amber); }
.metric-value { font-size: 26px; font-weight: 700; color: var(--text-primary); letter-spacing: -.5px; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 10px; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .6px; margin-top: 4px; }
.metric-sub { font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.metric-icon { position: absolute; top: 14px; right: 16px; font-size: 18px; opacity: .35; }

/* ── Tab strip ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 0 !important;
    margin-bottom: 20px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 14px; margin-top: 4px;
}
.section-header-line {
    flex: 1; height: 1px;
    background: linear-gradient(to right, var(--border), transparent);
}
.section-title {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .9px;
    color: var(--text-muted);
}

/* ── Cards ── */
.content-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px;
    margin-bottom: 16px;
}
.content-card-title {
    font-size: 13px; font-weight: 600;
    color: var(--text-primary); margin-bottom: 12px;
    display: flex; align-items: center; gap: 7px;
}
.exec-summary {
    font-size: 14px; line-height: 1.7;
    color: var(--text-secondary);
    background: var(--bg-surface);
    border-radius: var(--radius);
    padding: 14px 16px;
    border-left: 3px solid var(--accent);
}

/* ── Insight items ── */
.insight-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px; color: var(--text-secondary); line-height: 1.5;
}
.insight-item:last-child { border-bottom: none; padding-bottom: 0; }
.insight-bullet {
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--accent-soft);
    border: 1px solid var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; color: #a5b4fc; flex-shrink: 0; margin-top: 1px;
}

/* ── Action items ── */
.action-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 8px;
    font-size: 13px;
}
.action-priority {
    font-size: 9px; font-weight: 700;
    letter-spacing: .5px; text-transform: uppercase;
    padding: 2px 7px; border-radius: 4px;
    flex-shrink: 0;
}
.priority-high { background: var(--red-soft); color: var(--red); border: 1px solid rgba(239,68,68,.25); }
.priority-med  { background: var(--amber-soft); color: var(--amber); border: 1px solid rgba(245,158,11,.25); }
.priority-low  { background: var(--green-soft); color: var(--green); border: 1px solid rgba(34,197,94,.2); }
.action-owner { margin-left: auto; font-size: 11px; color: var(--text-muted); }

/* ── Automation buttons ── */
.automation-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.auto-btn {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    text-align: center;
    cursor: pointer;
    transition: all .2s;
    display: flex; flex-direction: column; align-items: center; gap: 6px;
}
.auto-btn:hover { border-color: var(--accent); background: var(--accent-soft); }
.auto-btn-icon { font-size: 20px; }
.auto-btn-label { font-size: 11px; font-weight: 600; color: var(--text-primary); }
.auto-btn-sub { font-size: 10px; color: var(--text-muted); }

/* ── KV cards ── */
.kv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.kv-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 14px;
}
.kv-key { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: var(--text-muted); }
.kv-val { font-size: 14px; font-weight: 600; color: var(--text-primary); margin-top: 4px; }

/* ── RAG status ── */
.rag-status-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
}
.rag-status-row:last-child { border-bottom: none; }
.rag-label { color: var(--text-muted); font-size: 12px; }
.rag-val { color: var(--text-primary); font-weight: 500; font-size: 12px; }
.rag-badge {
    background: var(--green-soft);
    border: 1px solid rgba(34,197,94,.2);
    color: var(--green);
    font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
    letter-spacing: .3px;
}
.rag-badge-accent {
    background: var(--accent-soft);
    border: 1px solid rgba(99,102,241,.25);
    color: #a5b4fc;
    font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
}

/* ── System info bar ── */
.sys-info-bar {
    display: flex; gap: 8px; flex-wrap: wrap;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
    margin-top: 8px;
}
.sys-chip {
    display: flex; align-items: center; gap: 5px;
    font-size: 11px; color: var(--text-secondary);
}
.sys-chip-label { color: var(--text-muted); }
.sys-chip-val { font-weight: 600; color: var(--text-primary); }
.sys-divider { width: 1px; background: var(--border); margin: 0 4px; }

/* ── Chat UI ── */
.chat-msg {
    display: flex; gap: 10px;
    margin-bottom: 16px;
    align-items: flex-start;
}
.chat-avatar {
    width: 30px; height: 30px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}
.chat-avatar-ai { background: var(--accent); }
.chat-avatar-user { background: var(--border-light); }
.chat-bubble {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 0 var(--radius) var(--radius) var(--radius);
    padding: 12px 14px;
    font-size: 13px; color: var(--text-secondary); line-height: 1.6;
    max-width: 80%;
}
.chat-bubble-user {
    background: var(--accent-soft);
    border-color: rgba(99,102,241,.25);
    color: var(--text-primary);
    border-radius: var(--radius) 0 var(--radius) var(--radius);
    margin-left: auto;
}
.chat-meta { font-size: 10px; color: var(--text-muted); margin-top: 5px; }

.confidence-bar-wrap {
    display: flex; align-items: center; gap: 8px; margin-top: 8px;
}
.confidence-bar-bg {
    flex: 1; height: 4px; background: var(--border);
    border-radius: 2px; overflow: hidden;
}
.confidence-bar-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(to right, var(--accent), #818cf8);
}
.confidence-label { font-size: 10px; color: var(--text-muted); }

/* ── Suggested questions ── */
.sq-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.sq-btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 12px;
    font-size: 12px; color: var(--text-secondary);
    cursor: pointer; text-align: left;
    transition: all .2s;
    display: flex; align-items: center; gap: 6px;
}
.sq-btn:hover { border-color: var(--accent); color: var(--text-primary); }

/* ── Export buttons ── */
.export-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 16px; }
.exp-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 18px 14px;
    text-align: center;
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    transition: all .2s;
}
.exp-card:hover { border-color: var(--border-light); background: var(--bg-card-hover); }
.exp-icon { font-size: 24px; }
.exp-title { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.exp-desc { font-size: 11px; color: var(--text-muted); }

/* ── Streamlit overrides ── */
.stCheckbox label { font-size: 13px !important; color: var(--text-secondary) !important; }
.stCheckbox [data-testid="stCheckbox"] { padding: 4px 0 !important; }
div[data-testid="stFileUploader"] { background: transparent !important; }
div[data-testid="stFileUploader"] > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--text-secondary) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .88 !important; }
.stExpander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.stExpander summary { color: var(--text-secondary) !important; font-size: 13px !important; }
div[data-testid="stChatInput"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
}
div[data-testid="stChatInput"] textarea { color: var(--text-primary) !important; }

/* hide default metric styling conflicts */
div[data-testid="stMetric"] { display: none; }
div[data-testid="column"] { padding: 0 !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }

/* tip bar */
.tip-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: var(--radius);
    padding: 10px 14px;
    font-size: 12px; color: var(--text-secondary);
    display: flex; align-items: center; gap: 8px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─── ② Session state ─────────────────────────────────────────────────────────
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "analysis_error" not in st.session_state:        # ← NEW
    st.session_state.analysis_error = ""
if "result" not in st.session_state:                # ← NEW: full pipeline output
    st.session_state.result = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm DocuMind's AI assistant. Upload and analyze a document, then ask me anything about its contents.",
            "confidence": 0,
        }
    ]
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = "No document uploaded"


# ─── ③ build_display() — maps real pipeline output → UI shape ────────────────
def build_display(result):
    """
    Translate run_document_pipeline() output into the flat dict the UI renders.
    Returns safe defaults when result is None (pre-analysis state).
    """
    if result is None:
        return {
            "doc_type":        "—",
            "processing_time": "—",
            "indexed_chunks":  "—",
            "retrieved_chunks": "—",
            "pages":           "—",
            "confidence":      "—",
            "summary": "Upload a document and click <strong>Analyze Document</strong> to see results.",
            "insights":        [],
            "action_items":    [],
            "extracted":       {},
            "rag_index":       {},
            "rag_result":      {},
        }

    classification = result.get("classification") or {}
    doc_summary    = result.get("document_summary") or {}
    rag_index      = result.get("rag_index") or {}
    rag_result     = result.get("rag_result") or {}
    action_items   = result.get("action_items") or []
    structured     = result.get("structured_data") or {}

    # Document type
    doc_type = (
        classification.get("document_type", "generic_document")
        .replace("_", " ").title()
    )

    # Confidence
    conf_raw   = classification.get("confidence", 0)
    confidence = f"{int(float(conf_raw) * 100)}%" if conf_raw else "—"

    # RAG counts
    indexed_chunks   = rag_index.get("chunk_count", "—")
    retrieved_chunks = len(rag_result.get("chunks", []))

    # Executive summary
    summary_text = (
        doc_summary.get("summary")
        or classification.get("summary")
        or "No summary available."
    )

    # Key insights — document_summary returns list under several possible keys
    raw_insights = (
        doc_summary.get("key_insights")
        or doc_summary.get("key_points")
        or doc_summary.get("major_conclusions")
        or []
    )
    if isinstance(raw_insights, str):
        raw_insights = [raw_insights]
    insights = [str(i) for i in raw_insights if i]

    # Action items (meeting_notes only) → UI shape
    ui_action_items = []
    for item in action_items:
        if not isinstance(item, dict):
            continue
        p_raw   = str(item.get("priority") or "").strip().lower()
        p_css   = "high" if "high" in p_raw else "med" if "med" in p_raw else "low"
        ui_action_items.append({
            "task":     item.get("task") or "",
            "priority": p_css,
            "owner":    item.get("owner") or "Unassigned",
            "due":      item.get("deadline") or "—",
        })

    # Extracted KV pairs from structured_data (all non-meeting-notes docs)
    extracted = {}
    if structured and isinstance(structured, dict):
        for k, v in structured.items():
            if v is None:
                continue
            label = k.replace("_", " ").title()
            extracted[label] = (
                ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
            )
    elif classification.get("document_type") == "meeting_notes" and action_items:
        from collections import Counter
        priorities = Counter(
            str(i.get("priority", "")).strip().lower()
            for i in action_items if isinstance(i, dict)
        )
        extracted = {
            "High Priority Tasks":   str(priorities.get("high", 0)),
            "Medium Priority Tasks":  str(priorities.get("medium", 0)),
            "Low Priority Tasks":     str(priorities.get("low", 0)),
            "Total Action Items":     str(len(action_items)),
        }

    return {
        "doc_type":        doc_type,
        "processing_time": result.get("extraction_method") or "~2–4s",
        "indexed_chunks":  indexed_chunks,
        "retrieved_chunks": retrieved_chunks,
        "pages":           result.get("page_count", "—"),
        "confidence":      confidence,
        "summary":         summary_text,
        "insights":        insights,
        "action_items":    ui_action_items,
        "extracted":       extracted,
        "rag_index":       rag_index,
        "rag_result":      rag_result,
    }


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-inner">
        <div class="brand-block">
            <div class="brand-icon">🧠</div>
            <div>
                <div class="brand-name">DocuMind</div>
                <div class="brand-tagline">Document Intelligence</div>
            </div>
            <div class="ai-badge">AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 20px;">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">📄 Document Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop file or click to browse",
        type=["pdf", "docx", "txt", "png", "jpg"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.session_state.uploaded_file_name = uploaded_file.name
        sz  = f"{uploaded_file.size / 1024:.1f} KB"
        ext = uploaded_file.name.split(".")[-1].upper()
        st.markdown(f"""
        <div class="file-info-card">
            <div class="file-info-name">📎 {uploaded_file.name}</div>
            <div class="file-info-meta">{sz} · {ext}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="file-info-card">
            <div class="file-info-name">📎 No document uploaded</div>
            <div class="file-info-meta">Upload a file to begin · Demo mode</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # # Analysis options
    # st.markdown('<div class="sidebar-label">⚙️ Analysis Options</div>', unsafe_allow_html=True)
    # opt_summary    = st.checkbox("Generate Summary",                  value=True)
    # opt_structured = st.checkbox("Extract Structured Data",           value=True)
    # opt_actions    = st.checkbox("Detect Action Items",               value=True)
    # opt_rag        = st.checkbox("Build RAG Index",                   value=True)
    # opt_workflow   = st.checkbox("Generate Workflow Recommendations",  value=False)

    # ── Recipient email — passed to run_document_pipeline ──
    st.markdown('<div class="sidebar-label">📧 Notification Email</div>', unsafe_allow_html=True)
    recipient_email = st.text_input(
        "Recipient email",
        placeholder="recipient@example.com",
        label_visibility="collapsed",
        key="recipient_email_input",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── ④ Analyze button — calls real backend ────────────────────────────────
    if st.button("✦ Analyze Document"):
        if not BACKEND_AVAILABLE:
            st.session_state.analysis_error = f"Backend unavailable: {_IMPORT_ERROR}"
        elif not uploaded_file:
            st.session_state.analysis_error = "Please upload a document first."
        else:
            st.session_state.analysis_error = ""
            # Write uploaded bytes to a temp file so the backend path-based API works
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner(
                "🤖 Analyzing document... This may take 10–20 seconds for larger files."
            ):
                result = run_document_pipeline(tmp_path, recipient_email or "")

            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            if result.get("status") == "error":
                st.session_state.analysis_error = result.get("message", "Unknown error.")
                st.session_state.analyzed = False
                st.session_state.result   = None
            else:
                print("DEBUG PAGE COUNT:", result.get("page_count"))
                st.session_state.result   = result
                st.session_state.analyzed = True

                # Reset chat for the new document
                st.session_state.messages = [
                    {
                        "role":    "assistant",
                        "content": "Document analyzed! Ask me anything about its contents.",
                        "confidence": 0,
                    }
                ]
            st.rerun()

    if st.session_state.analyzed:
        st.markdown("""
        <div class="status-row">
            <div class="dot"></div>
            Analysis complete
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.analysis_error:
        st.markdown(f"""
        <div class="error-row">⚠ {st.session_state.analysis_error}</div>
        """, unsafe_allow_html=True)

    if not BACKEND_AVAILABLE:
        st.warning(f"main.py not found: {_IMPORT_ERROR}", icon="⚠️")

    st.markdown("</div>", unsafe_allow_html=True)

    # System info
    st.markdown('<div style="padding: 0 20px; margin-top: 24px;">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">🔧 System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;">
        <div class="rag-status-row">
            <span class="rag-label">LLM</span>
            <span class="rag-val" style="font-size:11px;">Llama 3.3 70B</span>
        </div>
        <div class="rag-status-row">
            <span class="rag-label">Vector Store</span>
            <span class="rag-val" style="font-size:11px;">ChromaDB</span>
        </div>
        <div class="rag-status-row">
            <span class="rag-label">Embeddings</span>
            <span class="rag-val" style="font-size:11px;">MiniLM-L6-v2</span>
        </div>
        <div class="rag-status-row" style="padding-bottom:0;">
            <span class="rag-label">RAG</span>
            <span class="rag-badge">Enabled</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
# ③ Build display dict from real result (or empty defaults before analysis)
D     = build_display(st.session_state.result)
fname = st.session_state.uploaded_file_name

# Header
st.markdown(f"""
<div class="main-header">
    <div class="main-header-left">
        <h1>🧠 DocuMind</h1>
        <p>AI-Powered Document Intelligence · {fname}</p>
    </div>
    <div class="header-badges">
        <div class="hbadge"><div class="hbadge-dot"></div> {"Analysis Ready" if st.session_state.analyzed else "Awaiting Document"}</div>
        <div class="hbadge">⚡ Groq API</div>
        <div class="hbadge">🗄 ChromaDB</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Metric cards
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-icon">📄</div>
        <div class="metric-value" style="font-size:18px;">{D["doc_type"]}</div>
        <div class="metric-label">Document Type</div>
        <div class="metric-sub">{D["doc_type"]}</div>
    </div>
    <div class="metric-card green">
        <div class="metric-icon">⚡</div>
        <div class="metric-value" style="font-size:18px;">{D["processing_time"]}</div>
        <div class="metric-label">Extraction Method</div>
        <div class="metric-sub">Groq inference · Fast mode</div>
    </div>
    <div class="metric-card amber">
        <div class="metric-icon">🗂</div>
        <div class="metric-value">{D["indexed_chunks"]}</div>
        <div class="metric-label">Indexed Chunks</div>
        <div class="metric-sub">ChromaDB · MiniLM embeddings</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">🔍</div>
        <div class="metric-value">{D["retrieved_chunks"]}</div>
        <div class="metric-label">Retrieved Chunks</div>
        <div class="metric-sub">Semantic retrieval · Top-K=5</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">📑</div>
        <div class="metric-value">{D["pages"]}</div>
        <div class="metric-label">Pages</div>
        <div class="metric-sub">Full document parsed</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">🎯</div>
        <div class="metric-value">{D["confidence"]}</div>
        <div class="metric-label">Confidence</div>
        <div class="metric-sub">Classification accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_results, tab_chat, tab_export = st.tabs(["📊  Results", "💬  AI Assistant", "⬇  Export"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    col_left, col_right = st.columns([3, 2], gap="medium")

    with col_left:
        # Executive Summary
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Executive Summary</span>
            <div class="section-header-line"></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="content-card">
            <div class="exec-summary">{D["summary"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Key Insights
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Key Insights</span>
            <div class="section-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        if D["insights"]:
            items_html = "".join([
                f'<div class="insight-item"><div class="insight-bullet">✓</div><span>{i}</span></div>'
                for i in D["insights"]
            ])
            st.markdown(f'<div class="content-card">{items_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="content-card">
                <span style="color:var(--text-muted);font-size:13px;">
                    No insights extracted yet. Analyze a document to populate this section.
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Action Items (meeting_notes only — empty list for all other doc types)
        
        if D["action_items"]:
            st.markdown('<div class="section-header"><span class="section-title">Action Items</span><div class="section-header-line"></div></div>', unsafe_allow_html=True)
            ai_html = ""
            for item in D["action_items"]:
                p   = item["priority"]
                cls = "priority-high" if p == "high" else "priority-med" if p == "med" else "priority-low"
                ai_html += f"""
                <div class="action-item">
                    <span class="action-priority {cls}">{p.upper()}</span>
                    <span style="font-size:13px;color:var(--text-secondary);">{item["task"]}</span>
                    <span class="action-owner">👤 {item["owner"]} · {item["due"]}</span>
                </div>"""
            st.markdown(f'<div style="margin-bottom:16px">{ai_html}</div>', unsafe_allow_html=True)


        # Raw structured data JSON expander (non-meeting-notes docs)
        if st.session_state.result and st.session_state.result.get("structured_data"):
            with st.expander("🧩 Raw Structured Extraction (JSON)"):
                st.json(st.session_state.result["structured_data"])

    with col_right:

        # Workflow Recommendations
        st.markdown(
            '<div class="section-header"><span class="section-title">Workflow Recommendations</span><div class="section-header-line"></div></div>',
            unsafe_allow_html=True
        )

        if st.session_state.result and st.session_state.result.get("workflow_recommendations"):
            for workflow in st.session_state.result["workflow_recommendations"]:
                st.success(workflow)

        # Extracted Information
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Extracted Information</span>
            <div class="section-header-line"></div>
        </div>
        """, unsafe_allow_html=True)
        if D["extracted"]:
            kv_html = '<div class="kv-grid">'
            for k, v in D["extracted"].items():
                kv_html += f'<div class="kv-card"><div class="kv-key">{k}</div><div class="kv-val">{v}</div></div>'
            kv_html += "</div>"
            st.markdown(f'<div class="content-card">{kv_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="content-card">
                <span style="color:var(--text-muted);font-size:13px;">
                    Extracted fields will appear here after analysis.
                </span>
            </div>
            """, unsafe_allow_html=True)

        # RAG Status
        st.markdown("""
        <div class="section-header">
            <span class="section-title">RAG Pipeline Status</span>
            <div class="section-header-line"></div>
        </div>
        """, unsafe_allow_html=True)
        rag_idx  = D["rag_index"]
        rag_ret  = D["rag_result"]
        indexed  = rag_idx.get("chunk_count", "—")
        retrieved = len(rag_ret.get("chunks", []))
        if not st.session_state.analyzed:
            rag_status_badge = '<span class="rag-val">— Pending</span>'
        elif rag_idx.get("success"):
            rag_status_badge = '<span class="rag-badge">✓ Ready</span>'
        else:
            rag_status_badge = '<span style="color:var(--red);font-size:11px;">✗ Error</span>'

        st.markdown(f"""
        <div class="content-card">
            <div class="rag-status-row">
                <span class="rag-label">Indexed Chunks</span>
                <span class="rag-val">{indexed}</span>
            </div>
            <div class="rag-status-row">
                <span class="rag-label">Retrieved Chunks</span>
                <span class="rag-val">{retrieved}</span>
            </div>
            <div class="rag-status-row">
                <span class="rag-label">Embedding Model</span>
                <span class="rag-val">all-MiniLM-L6-v2</span>
            </div>
            <div class="rag-status-row">
                <span class="rag-label">Vector Store</span>
                <span class="rag-badge-accent">ChromaDB</span>
            </div>
            <div class="rag-status-row">
                <span class="rag-label">Retrieval Status</span>
                {rag_status_badge}
            </div>
            <div class="rag-status-row">
                <span class="rag-label">LLM Backend</span>
                <span class="rag-val">Groq · llama-3.3-70b</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if rag_idx:
            with st.expander("RAG Index Details"):
                st.json(rag_idx)

    # Tip bar
    st.markdown("""
    <div class="tip-bar">
        💡 <span>Switch to the <strong>AI Assistant</strong> tab to ask follow-up questions about this document using semantic search.</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    col_chat, col_sq = st.columns([2, 1], gap="medium")

    with col_chat:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Conversation</span>
            <div class="section-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "assistant":
                    conf = msg.get("confidence", 0)
                    conf_bar = ""
                    if conf and conf > 0:
                        conf_bar = f"""
                        <div class="confidence-bar-wrap">
                            <span class="confidence-label">Confidence</span>
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill" style="width:{conf}%;"></div>
                            </div>
                            <span class="confidence-label">{conf}%</span>
                        </div>"""
                    st.markdown(f"""
                    <div class="chat-msg">
                        <div class="chat-avatar chat-avatar-ai">🧠</div>
                        <div>
                            <div class="chat-bubble">{msg["content"]}{conf_bar}</div>
                            <div class="chat-meta">DocuMind AI · {datetime.now().strftime("%H:%M")}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if msg.get("sources"):
                        with st.expander(f"📚 View {len(msg['sources'])} source chunks"):
                            for src in msg["sources"]:
                                st.markdown(f"""
                                <div style="background:var(--bg-surface);border:1px solid var(--border);
                                     border-radius:8px;padding:10px 12px;margin-bottom:8px;
                                     font-size:12px;color:var(--text-secondary);">
                                    <strong style="color:var(--text-primary)">{src["source"]} · Score {src["score"]}</strong><br>
                                    {src["text"][:400]}{"…" if len(src["text"]) > 400 else ""}
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-msg" style="flex-direction:row-reverse;">
                        <div class="chat-avatar chat-avatar-user">👤</div>
                        <div style="text-align:right;">
                            <div class="chat-bubble chat-bubble-user">{msg["content"]}</div>
                            <div class="chat-meta">You · {datetime.now().strftime("%H:%M")}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ─── ⑤ Chat input → real answer_document_question() ─────────────────
        user_input = st.chat_input("Ask anything about this document…")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            if not st.session_state.analyzed or not BACKEND_AVAILABLE:
                answer_text = "Please upload and analyze a document first."
                sources     = []
            else:
                raw = answer_document_question(user_input)
                # Backend appends evidence after "\n\n---\n\n"
                if "\n\n---\n\n" in raw:
                    answer_text, _ = raw.split("\n\n---\n\n", 1)
                else:
                    answer_text = raw
                # Pull real chunks from last rag_result for the expander
                sources = [
                    {
                        "source": c.get("source", f"Chunk {c.get('chunk_index','?')}"),
                        "score":  c.get("score", "—"),
                        "text":   c.get("text", ""),
                    }
                    for c in (st.session_state.result or {})
                        .get("rag_result", {}).get("chunks", [])
                ]

            st.session_state.messages.append({
                "role":      "assistant",
                "content":   answer_text,
                "confidence": None,
                "sources":   sources,
            })
            st.rerun()

    with col_sq:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Suggested Questions</span>
            <div class="section-header-line"></div>
        </div>
        <div class="content-card">
        """, unsafe_allow_html=True)

        suggestions = [
            ("📋", "Summarize this document"),
            ("✅", "What action items were detected?"),
            ("📅", "What important dates exist?"),
            ("⚠️", "What are the key risks?"),
            ("👥", "Who are the involved parties?"),
            ("💰", "What are the financial terms?"),
        ]

        # ─── ⑥ Suggested-question buttons → real answer_document_question() ──
        for icon, q in suggestions:
            if st.button(f"{icon}  {q}", key=f"sq_{q}"):
                st.session_state.messages.append({"role": "user", "content": q})

                if not st.session_state.analyzed or not BACKEND_AVAILABLE:
                    ans  = "Please upload and analyze a document first."
                    srcs = []
                else:
                    raw = answer_document_question(q)
                    if "\n\n---\n\n" in raw:
                        ans, _ = raw.split("\n\n---\n\n", 1)
                    else:
                        ans = raw
                    srcs = [
                        {
                            "source": c.get("source", f"Chunk {c.get('chunk_index','?')}"),
                            "score":  c.get("score", "—"),
                            "text":   c.get("text", ""),
                        }
                        for c in (st.session_state.result or {})
                            .get("rag_result", {}).get("chunks", [])
                    ]

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": ans,
                    "sources": srcs,
                })
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Export & Distribute</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── CSS: make Streamlit buttons look exactly like .exp-card ──────────────
    # Each card column contains ONE button that fills the card dimensions.
    # The button background/border/text is stripped; the .exp-card HTML sits
    # above it visually via pointer-events:none so the button captures clicks.
    # Nothing in this <style> block affects any other tab.
    st.markdown("""
    <style>
    /* Wrapper that stacks the visual card and the invisible button */
    .exp-wrap {
        position: relative;
        margin-bottom: 0;
    }
    /* The visual card — pointer-events off so clicks fall through to button */
    .exp-wrap .exp-card {
        pointer-events: none;
        margin-bottom: 0;
        user-select: none;
    }
    /* The Streamlit button inside an exp-wrap col: stretch to fill card */
    div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"]
        .exp-wrap + div[data-testid="stVerticalBlock"] button,
    .exp-col-btn > div[data-testid="stVerticalBlock"] button {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: pointer;
        z-index: 10;
    }
    /* Simpler universal selector scoped to export tab columns */
    .exp-btn-overlay button {
        position: absolute !important;
        top: 0 !important; left: 0 !important;
        width: 100% !important; height: 100% !important;
        background: transparent !important;
        border: none !important;
        color: transparent !important;
        font-size: 0 !important;
        cursor: pointer !important;
        z-index: 10 !important;
        padding: 0 !important;
        min-height: unset !important;
        box-shadow: none !important;
        opacity: 0 !important;
    }
    /* Hover: let the card's own :hover CSS fire since pointer-events re-enabled on wrapper */
    .exp-wrap:hover .exp-card {
        border-color: var(--border-light) !important;
        background: var(--bg-card-hover) !important;
    }
    /* Disabled state: mute the card */
    .exp-wrap.exp-disabled .exp-card {
        opacity: 0.45;
        cursor: not-allowed;
    }
    /* Position context for the overlay */
    .exp-wrap { display: block; }
    div[data-testid="stVerticalBlock"]:has(.exp-wrap) {
        position: relative;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Shared data ───────────────────────────────────────────────────────────
    _result         = st.session_state.result
    _doc_summary    = (_result or {}).get("document_summary") or {}
    _classification = (_result or {}).get("classification") or {}
    _structured     = (_result or {}).get("structured_data") or {}
    _action_items   = (_result or {}).get("action_items") or []
    _doc_type       = _classification.get("document_type", "—")
    _recipient      = st.session_state.get("recipient_email_input", "")
    _ready          = st.session_state.analyzed and BACKEND_AVAILABLE

    # ── Pre-build all download payloads (no computation inside columns) ───────
    # 1. Summary TXT
    _summary_lines = [
        f"DocuMind Export — {fname}",
        f"Document Type: {_doc_type}",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "", "=" * 48, "SUMMARY", "=" * 48,
        _doc_summary.get("summary") or D["summary"], "",
    ]
    for _ins in (_doc_summary.get("key_insights") or _doc_summary.get("key_points") or []):
        _summary_lines.append(f"• {_ins}")
    _summary_txt = "\n".join(str(x) for x in _summary_lines)

    # 2. Structured JSON
    try:
        _structured_json = json.dumps(_structured, indent=2, default=str) if _structured else "{}"
    except Exception:
        _structured_json = "{}"

    # 3. Full pipeline JSON
    try:
        _full_json = json.dumps(_result or {}, indent=2, default=str)
    except Exception:
        _full_json = "{}"

    # 4. Action items CSV via pandas
    if _action_items:
        _csv_rows = [
            {"Task": i.get("task",""), "Priority": i.get("priority",""),
             "Owner": i.get("owner",""), "Deadline": i.get("deadline",""),
             "Confidence": i.get("confidence","")}
            for i in _action_items if isinstance(i, dict)
        ]
        _csv_buf = io.StringIO()
        pd.DataFrame(_csv_rows).to_csv(_csv_buf, index=False)
        _csv_data = _csv_buf.getvalue()
        _csv_label = f"Action Items CSV · {len(_csv_rows)} rows"
    else:
        _csv_data  = "Task,Priority,Owner,Deadline,Confidence\n"
        _csv_label = "Action Items CSV"

    # ── Feedback placeholders (rendered above cards so toasts stay in tab) ────
    _feedback = st.empty()

    # ── Six-column card grid — each column: card HTML + invisible overlay btn ─
    # The .exp-wrap div gives position:relative context. The button inside
    # .exp-btn-overlay is absolutely positioned to 0/0/100%/100% via the CSS
    # above, making the entire card surface the click target.
    c1, c2, c3, c4, c5= st.columns(5, gap="medium")

    # ── Card 1 — Download Summary (st.download_button, fires immediately) ─────
    with c1:
        _dis1 = not _ready
        st.markdown(f'<div class="exp-wrap{"exp-disabled" if _dis1 else ""}"><div class="exp-card"><div class="exp-icon">📄</div><div class="exp-title">Download Summary</div><div class="exp-desc">{"Analyze first" if _dis1 else "summary.txt"}</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="exp-btn-overlay">', unsafe_allow_html=True)
        st.download_button(
            label="Download Summary",
            data=_summary_txt,
            file_name="summary.txt",
            mime="text/plain",
            disabled=_dis1,
            key="exp_dl_summary",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 2 — Export Structured Data ──────────────────────────────────────
    with c2:
        _dis2 = not _ready or not _structured
        st.markdown(f'<div class="exp-wrap{"exp-disabled" if _dis2 else ""}"><div class="exp-card"><div class="exp-icon">🗂</div><div class="exp-title">Export Structured Data</div><div class="exp-desc">{"N/A for meeting notes" if _ready and _doc_type=="meeting_notes" else ("Analyze first" if _dis2 else "structured_data.json")}</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="exp-btn-overlay">', unsafe_allow_html=True)
        st.download_button(
            label="Export Structured Data",
            data=_structured_json,
            file_name="structured_data.json",
            mime="application/json",
            disabled=_dis2,
            key="exp_dl_structured",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 3 — Full Pipeline JSON ───────────────────────────────────────────
    with c3:
        _dis3 = not _ready
        st.markdown(f'<div class="exp-wrap{"exp-disabled" if _dis3 else ""}"><div class="exp-card"><div class="exp-icon">&#123; &#125;</div><div class="exp-title">Export JSON</div><div class="exp-desc">{"Analyze first" if _dis3 else "documind_analysis.json"}</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="exp-btn-overlay">', unsafe_allow_html=True)
        st.download_button(
            label="Export JSON",
            data=_full_json,
            file_name="documind_analysis.json",
            mime="application/json",
            disabled=_dis3,
            key="exp_dl_full",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 4 — Action Items CSV ─────────────────────────────────────────────
    with c4:
        _dis4 = not _ready or not _action_items
        st.markdown(f'<div class="exp-wrap{"exp-disabled" if _dis4 else ""}"><div class="exp-card"><div class="exp-icon">📊</div><div class="exp-title">Action Items CSV</div><div class="exp-desc">{"Meeting notes only" if _ready and not _action_items else ("Analyze first" if _dis4 else _csv_label)}</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="exp-btn-overlay">', unsafe_allow_html=True)
        st.download_button(
            label=_csv_label,
            data=_csv_data,
            file_name="action_items.csv",
            mime="text/csv",
            disabled=_dis4,
            key="exp_dl_csv",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 5 — Send Email Report (st.button — triggers on click → spinner) ──
    with c5:
        _dis5 = not _ready
        st.markdown(f'<div class="exp-wrap{"exp-disabled" if _dis5 else ""}"><div class="exp-card"><div class="exp-icon">📧</div><div class="exp-title">Send Email Report</div><div class="exp-desc">{"Analyze first" if _dis5 else ("No recipient set" if not _recipient else "via Gmail SMTP")}</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="exp-btn-overlay">', unsafe_allow_html=True)
        if st.button("Send Email Report", disabled=_dis5, key="exp_btn_email", use_container_width=True):
            if not _recipient:
                _feedback.error("Enter a recipient email in the sidebar first.", icon="⚠️")
            else:
                _payload = _structured if _structured else _action_items
                if not _payload:
                    _feedback.error("Nothing to send — no structured data or action items.", icon="⚠️")
                else:
                    with st.spinner("Sending email…"):
                        _er = send_summary_email(_payload, _classification, _recipient)
                    if _er.get("success"):
                        _feedback.success(f"Email sent to **{_er['email_sent_to']}**", icon="✅")
                    else:
                        _feedback.error(
                            _er.get("error","Email failed.")
                            + (" — " + _er["details"] if "details" in _er else ""),
                            icon="❌",
                        )
        st.markdown('</div>', unsafe_allow_html=True)

    # # ── Card 6 — Log to Google Sheets ─────────────────────────────────────────
    # with c6:
    #     _dis6 = not _ready or not _action_items
    #     st.markdown(f'<div class="exp-wrap{"exp-disabled" if _dis6 else ""}"><div class="exp-card"><div class="exp-icon">📋</div><div class="exp-title">Log to Google Sheets</div><div class="exp-desc">{"Meeting notes only" if _ready and not _action_items else ("Analyze first" if _dis6 else "Append rows")}</div></div></div>', unsafe_allow_html=True)
    #     st.markdown('<div class="exp-btn-overlay">', unsafe_allow_html=True)
    #     if st.button("Log to Google Sheets", disabled=_dis6, key="exp_btn_sheets", use_container_width=True):
    #         with st.spinner("Logging to Sheets…"):
    #             _sr = log_to_google_sheet(_action_items)
    #         if _sr.get("success"):
    #             _n = _sr.get("rows_logged", 0)
    #             _feedback.success(f"Logged **{_n}** row{'s' if _n != 1 else ''} to Google Sheets.", icon="✅")
    #         else:
    #             _feedback.error(
    #                 _sr.get("error","Sheets failed.")
    #                 + (" — " + _sr["details"] if "details" in _sr else ""),
    #                 icon="❌",
    #             )
    #     st.markdown('</div>', unsafe_allow_html=True)

  