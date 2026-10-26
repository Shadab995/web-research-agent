"""Command-line interface for the Web Research Agent.

Usage:
    python main.py "What are the latest advances in solid-state batteries?"
    python main.py "Compare LangGraph and CrewAI" --max-iterations 3 --save report.md
"""

import argparse

from research_agent import build_graph, initial_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Research a question on the web and get a cited report.")
    parser.add_argument("question", help="The question to research")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max search rounds (default: 2)")
    parser.add_argument("--model", default=None, help="Claude model (default: CLAUDE_MODEL env var)")
    parser.add_argument("--save", default=None, help="Optional path to save the report as Markdown")
    args = parser.parse_args()

    app = build_graph(model=args.model)
    final_report = ""

    # stream_mode="updates" yields {node_name: state_update} after each node runs,
    # so we can show the agent's progress step by step.
    for step in app.stream(initial_state(args.question, args.max_iterations), stream_mode="updates"):
        for node, update in step.items():
            if node == "plan_queries":
                print("🧭 Planned searches:")
                for q in update["queries"]:
                    print(f"   - {q}")
            elif node == "web_search":
                print(f"🔎 Search round {update['iteration']}: found {len(update['sources'])} results")
            elif node == "reflect":
                if update["queries"]:
                    print("🤔 Need more info, follow-up searches:")
                    for q in update["queries"]:
                        print(f"   - {q}")
                else:
                    print("✅ Enough information gathered")
            elif node == "write_report":
                final_report = update["report"]

    print("\n" + "=" * 80 + "\n")
    print(final_report)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\n💾 Saved to {args.save}")


if __name__ == "__main__":
    main()
