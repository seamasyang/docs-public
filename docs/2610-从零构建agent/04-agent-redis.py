import os
from dotenv import load_dotenv
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.redis import RedisSaver

load_dotenv()


def call_model(state: MessagesState):
    SYSTEM_PROMPT = """
    你是一个思维清晰的AI助手，擅长应用思维链(CoT)解决问题
    在回应前，请遵循以下思考步骤
    1. 分析：识别问题的核心要素与约束条件
    2. 规划：构思解决问题的策略与步骤
    3. 评估：权衡不同方案的利弊
    4. 综合：基于评估，形成最终的、合理的答案

    请将你的推理过程置于“思考过程”标记之后，将最终答案置于“回答”标记之后
    """
    llm = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )
    cot_prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )
    chain = cot_prompt | llm
    response = chain.invoke(state)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("call_llm", call_model)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)

saver = RedisSaver(
    "redis://:redis2026@192.168.1.7:6379",
    ttl={
        "default_ttl": 60,  # 60 分钟后过期
        "refresh_on_read": True,  # 每次读取时重新计时
    },
)
saver.setup()
agent = workflow.compile(checkpointer=saver)


def call_agent(user_message: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    input_message = HumanMessage(content=user_message)
    response = agent.invoke({"messages": [input_message]}, config=config)
    return response["messages"][-1].content


if __name__ == "__main__":
    session_id = uuid4().hex
    messages = ["你好，我是李四", "请问我是谁"]
    for message in messages:
        result = call_agent(message, session_id)
        print(f"[user]:{message}\n[agent]:{result}\n\n")
