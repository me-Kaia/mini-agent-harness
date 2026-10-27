"""命令行入口：跑单个任务，或从断点恢复。

  python main.py "parts.txt 里所有元件总价是多少？" --base-url https://api.deepseek.com --model deepseek-chat
  python main.py --resume <run_id> ...
接本地 llama.cpp server：--base-url http://127.0.0.1:8080/v1 --model local
"""
import argparse
import os

from harness import Agent, AgentConfig, Checkpointer, OpenAICompatClient, ToolRegistry, Tracer
from harness.builtin_tools import calculator_tool, make_file_tools

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("goal", nargs="?")
    p.add_argument("--resume", help="从存档恢复的 run_id")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    p.add_argument("--workspace", default=os.path.join(ROOT, "sample_workspace"))
    p.add_argument("--max-turns", type=int, default=12)
    args = p.parse_args()

    reg = ToolRegistry()
    reg.register(calculator_tool())
    for t in make_file_tools(args.workspace):
        reg.register(t)
    agent = Agent(OpenAICompatClient(args.base_url, args.model, args.api_key), reg,
                  AgentConfig(max_turns=args.max_turns),
                  checkpointer=Checkpointer(os.path.join(ROOT, "runs", "checkpoints")),
                  tracer=Tracer(os.path.join(ROOT, "runs", "trace.jsonl")))
    state = agent.resume(args.resume) if args.resume else agent.run(args.goal)
    print(f"[run_id={state.run_id}] status={state.status} turns={state.turn} tokens={state.tokens_used}")
    print(state.final_answer)


if __name__ == "__main__":
    main()
