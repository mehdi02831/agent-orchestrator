import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from agentlab import config
from agentlab.ui.client import ApiClient, ApiError, is_terminal

API_URL = os.getenv("ORCH_API_URL", "http://127.0.0.1:8010")

st.set_page_config(page_title="Agent orchestrator", layout="wide")
st.title("Agent orchestrator")

if not config.ORCH_API_KEY:
    st.error("ORCH_API_KEY is not set in .env")
    st.stop()
client = ApiClient(API_URL, config.ORCH_API_KEY)
try:
    client.health()
except ApiError as e:
    st.error(f"API not reachable at {API_URL} ({e.detail}). Start it with: docker compose up -d")
    st.stop()


def render_status(status):
    state = status["state"]
    if state == "SUCCESS":
        res = status["result"] or {}
        st.success(f"Outcome: {res.get('outcome')} (run {res.get('run_id')})")
        st.write(res.get("final", ""))
        if res.get("proposal") == "queued":
            st.info(f"A write proposal is waiting for review (pending action {res.get('pending_id')}). "
                    "Open the Review tab.")
        elif res.get("outcome") == "escalate":
            st.warning(f"Escalated to a human. Proposal status: {res.get('proposal')}")
        run = status.get("run") or {}
        with st.expander("Details"):
            st.write(f"steps {run.get('steps')} | latency {run.get('latency_s')} s | "
                     f"guard {run.get('guard_fired') or '-'}")
            for d in run.get("delegations", []):
                st.write(f"delegate {d['agent']}: {d['task']}")
                st.caption(d["result"])
            for t in run.get("tool_calls", []):
                st.write(f"tool {t['name']} [{t['kind']}]")
    elif state == "FAILURE":
        st.error(f"Task failed: {status.get('error')}")
    else:
        st.info(f"State: {state}")


def act(pid, decision, edits, note):
    try:
        r = client.decide(pid, decision, edits, note)
    except ApiError as e:
        st.session_state["flash"] = ("error", f"#{pid}: {e.detail}")
    else:
        level = "success" if r["status"] in ("approved", "rejected") else "warning"
        st.session_state["flash"] = (level, f"#{pid}: {r['status']} - {r['message']}")
    st.rerun()


tab_tasks, tab_review = st.tabs(["Tasks", "Review"])

with tab_tasks:
    prompt = st.text_area("Task", height=100)
    if st.button("Submit", type="primary", disabled=not prompt.strip()):
        try:
            st.session_state["task_id"] = client.submit(prompt)
        except ApiError as e:
            st.error(f"Could not submit: {e.detail}")
    tid = st.session_state.get("task_id")
    if tid:
        st.caption(f"Task {tid}")
        auto = st.checkbox("Auto-refresh", value=True)
        try:
            status = client.status(tid)
        except ApiError as e:
            st.error(f"Could not read status: {e.detail}")
        else:
            render_status(status)
            if auto and not is_terminal(status["state"]):
                time.sleep(2)
                st.rerun()

with tab_review:
    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])
    try:
        items = client.pending()["pending"]
    except ApiError as e:
        st.error(f"Could not load pending actions: {e.detail}")
        items = []
    st.write(f"{len(items)} waiting for review")
    for it in items:
        pid, prop = it["id"], it["proposed"]
        with st.container(border=True):
            st.subheader(f"#{pid}: {prop['path']}")
            st.caption(f"Task: {it['prompt']}")
            st.write(f"Why it paused: {it['reason']}")
            if it.get("last_error"):
                st.warning(f"Last attempt failed: {it['last_error']}")
            content = st.text_area("Content (read all of it before approving)", value=prop["content"],
                                   height=320, key=f"c{pid}")
            path = st.text_input("Path", value=prop["path"], key=f"p{pid}")
            overwrite = st.checkbox("Allow overwriting an existing file", value=bool(prop.get("overwrite")),
                                    key=f"o{pid}")
            note = st.text_input("Note", key=f"n{pid}")
            c1, c2, c3 = st.columns(3)
            if c1.button("Approve as drafted (ignores edits above)", key=f"a{pid}"):
                act(pid, "approve", None, note)
            if c2.button("Approve with my edits", key=f"e{pid}"):
                edits = {}
                if content != prop["content"]:
                    edits["content"] = content
                if path != prop["path"]:
                    edits["path"] = path
                if overwrite != bool(prop.get("overwrite")):
                    edits["overwrite"] = overwrite
                if not edits:
                    st.warning("Nothing was changed.")
                else:
                    act(pid, "edit", edits, note)
            if c3.button("Reject", key=f"r{pid}"):
                act(pid, "reject", None, note)
    st.button("Refresh")
