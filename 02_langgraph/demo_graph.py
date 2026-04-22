from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage

# 0. 封装好Agent，方便后续调用
@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model = init_chat_model(
    model="qwen2.5:7b-instruct-q4_K_M", model_provider="ollama", temperature=0.1
)
webster_agent = create_agent(
    model,
    tools=[get_weather],
    name="webster_agent"
)

# 1. 定义调用 Agent 节点逻辑
def webster_agent_node(state: MessagesState):
    """
    调用真正的 Webster Agent 节点。
    直接把整个 state 传入 agent，让 agent 自行决定是否调用工具、思考、输出。
    """
    # 真实调用 agent（推荐方式）
    result = webster_agent.invoke(state)
    
    # result 通常包含 "messages"，我们直接返回更新后的 messages
    return {"messages": result["messages"]}


# --- 2. 定义普通对话节点 ---
def general_chat_node(state: MessagesState):
    return {"messages": [AIMessage(content="你好！我是 Webster 的闲聊模式。")]}

# --- 3. 路由逻辑 ---
def router(state: MessagesState) -> Literal["call_agent", "just_chat"]:
    last_msg = state["messages"][-1].content.lower()
    if any(kw in last_msg for kw in ["查", "天气", "weather"]):
        return "call_agent"
    return "just_chat"

# --- 4. 构建图 ---
workflow = StateGraph(MessagesState)
# 添加节点
workflow.add_node("agent_worker", webster_agent_node)
workflow.add_node("chat_worker", general_chat_node)

# 设置路由
workflow.add_conditional_edges(
    START,
    router,
    {
        "call_agent": "agent_worker",
        "just_chat": "chat_worker"
    }
)
# 连接到结束
workflow.add_edge("agent_worker", END)
workflow.add_edge("chat_worker", END)
graph = workflow.compile()