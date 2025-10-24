# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, cast

import dotenv
from langchain.agents import create_agent  # type: ignore
from langchain.chat_models import init_chat_model
from langchain.tools import tool  # type: ignore
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv()

# Define system prompt
SYSTEM_PROMPT = """\
あなたはオンライングラフ探索の専門家である教授です。

以下のツールを使えます：
- get_algorithm_info: アルゴリズムの詳細情報を取得
- get_complexity_analysis: 計算量や競合比の解析結果を取得
"""


# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""

    student_id: str


# Define state schema
@dataclass
class State:
    pass


# Define tools
@tool
def get_algorithm_info(algorithm_name: str) -> str:
    """Returns detailed information about the specified algorithm."""
    algorithms = {
        "BFS": """
幅優先探索（Breadth-First Search）：
- 概要：グラフを層ごとに探索するアルゴリズム
- 時間計算量：O(V + E)（V=頂点数、E=辺数）
- 空間計算量：O(V)
- 特徴：最短経路を保証（重みなしグラフ）
- 実装：キューを使用
        """,
        "DFS": """
深さ優先探索（Depth-First Search）：
- 概要：グラフを深く掘り下げながら探索
- 時間計算量：O(V + E)
- 空間計算量：O(V)（再帰スタック）
- 特徴：トポロジカルソート、強連結成分分解に使用
- 実装：スタックまたは再帰を使用
        """,
        "A*": """
A*探索アルゴリズム：
- 概要：ヒューリスティック関数を用いた最短経路探索
- 時間計算量：O(b^d)（b=分岐係数、d=深さ）
- 評価関数：f(n) = g(n) + h(n)（g=実コスト、h=ヒューリスティック）
- 特徴：許容的ヒューリスティックで最適解保証
- 応用：ナビゲーション、ゲームAI
        """,
        "k-server": """
k-サーバー問題：
- 概要：k個のサーバーで要求点を効率的にカバーする
- 競合比：最悪ケースでのオンライン/オフライン性能比
- 下界：k（どんなアルゴリズムでもk以上）
- 主要アルゴリズム：Greedy（2k-1競合）、Work Function Algorithm
- 応用：キャッシング、リソース配置
        """,
    }
    return algorithms.get(
        algorithm_name,
        f"申し訳ありません。{algorithm_name}の情報は登録されていません。",
    )


@tool
def get_complexity_analysis(problem_type: str) -> str:
    """Returns complexity and competitive ratio analysis for the problem type."""
    analyses = {
        "online_shortest_path": """
オンライン最短路問題の解析：
- オフライン最適：事前に全情報を知っている場合の最短路
- オンライン：辺の重みや存在が逐次的に判明
- 競合比：最悪ケースでのオンライン/オフライン比
- 主要結果：一般グラフではΩ(n)の下界
- 改善：特殊なグラフ構造（木、平面グラフ）で競合比改善可能
        """,
        "paging": """
ページング問題の解析：
- 問題：k個のページをキャッシュに保持、ページフォルト最小化
- LRU（Least Recently Used）：k競合
- FIFO（First In First Out）：k競合
- 下界：どんな決定性アルゴリズムもk競合
- ランダム化：Marking Algorithmで O(log k)競合比
        """,
        "metrical_task_system": """
距離空間タスクシステム（MTS）：
- k-サーバー問題の一般化
- 状態空間と遷移コストが与えられる
- Work Function Algorithm：2k-1競合
- ランダム化：O(log² k log n)競合比達成可能
- 応用：スケジューリング、動的最適化
        """,
    }
    return analyses.get(problem_type, f"{problem_type}の解析情報は登録されていません。")


# Configure model
model = init_chat_model("openai:gpt-4o", temperature=0)


# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""

    # Professor's explanation (required)
    explanation: str
    # Recommended next steps or references (optional)
    next_steps: str | None = None
    # Related technical terms or concepts (optional)
    related_concepts: str | None = None


# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_algorithm_info, get_complexity_analysis],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)


def main() -> None:
    """Run an interactive CLI session with the professor agent."""

    import uuid

    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    context = Context(student_id="1")

    print("Type 'exit' or 'quit' to end the conversation.")

    while True:
        try:
            user_message = input("You: ").strip()
        except EOFError:
            print("\nEOF received. Exiting.")
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break

        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = cast(
            dict[str, Any],
            agent.invoke(  # type: ignore[call-overload]
                {"messages": [{"role": "user", "content": user_message}]},
                config=config,
                context=context,
            ),
        )

        structured = response.get("structured_response")

        explanation = getattr(structured, "explanation", None)
        if explanation:
            print(f"Professor: {explanation}")
        else:
            print("Professor: Unable to provide a structured response.")

        next_steps = getattr(structured, "next_steps", None)
        if next_steps:
            print(f"Next steps: {next_steps}")

        related = getattr(structured, "related_concepts", None)
        if related:
            print(f"Related concepts: {related}")


if __name__ == "__main__":
    main()
