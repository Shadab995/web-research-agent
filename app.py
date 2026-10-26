"""Streamlit web UI for the Web Research Agent.

Run with:
    streamlit run app.py
"""

import streamlit as st

from research_agent import build_graph, initial_state

st.set_page_config(page_title="Web Research Agent", page_icon="🔎", layout="wide")
st.title("🔎 Web Research Agent")
st.caption("LangGraph + Tavily + Claude — ask a question, get a cited research report.")

with st.sidebar:
    st.header("Settings")
    max_iterations = st.slider("Max search rounds", 1, 4, 2)
    results_per_query = st.slider("Results per query", 2, 8, 4)

question = st.text_input("What do you want to research?", placeholder="e.g. What is the Model Context Protocol (MCP)?")

if st.button("Research", type="primary", disabled=not question):
    app = build_graph(results_per_query=results_per_query)
    report = ""
    with st.status("Researching...", expanded=True) as status:
        for step in app.stream(initial_state(question, max_iterations), stream_mode="updates"):
            for node, update in step.items():
                if node == "plan_queries":
                    st.write("🧭 **Planned searches:** " + " · ".join(update["queries"]))
                elif node == "web_search":
                    st.write(f"🔎 Search round {update['iteration']}: {len(update['sources'])} results")
                elif node == "reflect":
                    if update["queries"]:
                        st.write("🤔 **Follow-up searches:** " + " · ".join(update["queries"]))
                    else:
                        st.write("✅ Enough information gathered")
                elif node == "write_report":
                    report = update["report"]
        status.update(label="Done!", state="complete", expanded=False)

    st.markdown(report)
    st.download_button("Download report (.md)", report, file_name="research_report.md")
