"""
Episode Store Dashboard.

EL-8: Streamlit dashboard — browse and inspect episodes

Run with:
    streamlit run dashboard.py

Requires the episode store to be running at EPISODE_STORE_URL
(default: http://localhost:8100).
"""

import os
import json

import requests
import streamlit as st

STORE_URL = os.environ.get("EPISODE_STORE_URL", "http://localhost:8100")


def api_get(path: str, params: dict | None = None) -> dict | list:
    """GET request to the episode store API."""
    resp = requests.get(f"{STORE_URL}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Episode Store",
    page_icon="🔬",
    layout="wide",
)

st.title("Episode Store Dashboard")
st.caption("Browse and inspect agent episodes")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")
agent_filter = st.sidebar.text_input("Agent ID", placeholder="All agents")
status_filter = st.sidebar.selectbox(
    "Status", ["All", "running", "success", "failure", "timeout", "killed"]
)
model_filter = st.sidebar.text_input("Model", placeholder="e.g. gpt-4")
tool_filter = st.sidebar.text_input("Tool", placeholder="e.g. web_search")
limit = st.sidebar.slider("Results", min_value=10, max_value=200, value=50, step=10)

# Build query params
params: dict = {"limit": limit}
if agent_filter:
    params["agent_id"] = agent_filter
if status_filter != "All":
    params["status"] = status_filter
if model_filter:
    params["model"] = model_filter
if tool_filter:
    params["tool"] = tool_filter

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

try:
    health = api_get("/v1/health")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", health["status"])
    col2.metric("Episodes Stored", health["episodes_stored"])
    col3.metric("Version", health["version"])
except Exception as e:
    st.error(f"Cannot connect to episode store at {STORE_URL}: {e}")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Episode list
# ---------------------------------------------------------------------------

episodes = api_get("/v1/episodes", params=params)

if not episodes:
    st.info("No episodes found matching your filters.")
    st.stop()

st.subheader(f"Episodes ({len(episodes)} results)")

# Summary table
table_data = []
for ep in episodes:
    table_data.append({
        "Episode ID": ep["episode_id"][:12] + "...",
        "Agent": ep["agent_id"],
        "Status": ep["status"],
        "Steps": ep["step_count"],
        "Tokens": ep["total_tokens"],
        "Cost ($)": f"${ep['total_cost_usd']:.4f}",
        "Duration (ms)": ep["total_duration_ms"],
        "Tools": ", ".join(ep["tools_used"]),
    })
st.dataframe(table_data, use_container_width=True)

# ---------------------------------------------------------------------------
# Episode detail
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Inspect Episode")

episode_ids = [ep["episode_id"] for ep in episodes]
selected_id = st.selectbox("Select an episode", episode_ids, format_func=lambda x: x[:12] + "...")

if selected_id:
    episode = api_get(f"/v1/episodes/{selected_id}")

    # Header metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", episode["status"])
    c2.metric("Total Tokens", episode["total_tokens"])
    c3.metric("Total Cost", f"${episode['total_cost_usd']:.4f}")
    c4.metric("Duration", f"{episode['total_duration_ms']}ms")

    # Steps
    st.write("**Steps:**")
    for step in episode["steps"]:
        icon = {"llm_call": "🤖", "tool_call": "🔧", "tool_result": "📋", "decision": "🧭", "error": "❌"}.get(step["step_type"], "▪️")
        with st.expander(f"{icon} Step {step['step_index']}: {step['step_type']} — {step.get('tool_name') or step.get('model') or 'N/A'}"):
            col_a, col_b, col_c = st.columns(3)
            col_a.write(f"**Tokens:** {step['tokens']}")
            col_b.write(f"**Cost:** ${step['cost_usd']:.4f}")
            col_c.write(f"**Duration:** {step['duration_ms']}ms")
            if step.get("input_summary"):
                st.write(f"**Input:** {step['input_summary']}")
            if step.get("output_summary"):
                st.write(f"**Output:** {step['output_summary']}")
            if step.get("error"):
                st.error(step["error"])
            if step.get("metadata"):
                st.json(step["metadata"])

    # Metadata
    if episode.get("metadata"):
        st.write("**Episode Metadata:**")
        st.json(episode["metadata"])

    # Raw JSON
    with st.expander("Raw JSON"):
        st.json(episode)

# ---------------------------------------------------------------------------
# Diff tool
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Compare Episodes")

col_left, col_right = st.columns(2)
left_id = col_left.selectbox("Baseline (left)", episode_ids, key="diff_left", format_func=lambda x: x[:12] + "...")
right_id = col_right.selectbox("Comparison (right)", episode_ids, key="diff_right", index=min(1, len(episode_ids) - 1), format_func=lambda x: x[:12] + "...")

if st.button("Compare"):
    if left_id == right_id:
        st.warning("Select two different episodes to compare.")
    else:
        diff = api_get("/v1/episodes/diff", params={"left": left_id, "right": right_id})
        c1, c2, c3 = st.columns(3)
        c1.metric("Token Delta", f"{diff['token_delta']:+d}")
        c2.metric("Cost Delta", f"${diff['cost_delta']:+.4f}")
        c3.metric("Duration Delta", f"{diff['duration_delta']:+d}ms")

        c4, c5, c6 = st.columns(3)
        c4.metric("Matching Steps", diff["matching_steps"])
        c5.metric("Differing Steps", diff["differing_steps"])
        c6.metric("Extra Steps", f"L:{diff['extra_left']} R:{diff['extra_right']}")

        if diff["step_diffs"]:
            st.write("**Step Differences:**")
            for sd in diff["step_diffs"]:
                st.write(f"Step {sd['step_index']} — **{sd['field']}**: `{sd['left']}` → `{sd['right']}`")
        else:
            st.success("No step-level differences found.")
