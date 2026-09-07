# main.py
from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env into environment



from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver



from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

cp = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are an assistant.",
    checkpointer=cp
)
agent_response = llm.invoke(input="hi there")
print(agent_response)
